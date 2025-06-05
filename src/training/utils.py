"""
Utility functions for training the GL-Fusion model.
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
import logging
import json # Added for loading node text
from tqdm import tqdm
from transformers import AutoTokenizer # Added for tokenizer loading
import random
import io
import gc
import datetime
from pathlib import Path
import torch.distributed as dist
from sklearn.preprocessing import MinMaxScaler # Added

# Get logger instance
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set seed to {seed}")

class TrajectoryDataset(Dataset):
    """
    Dataset for training with trajectory data.
    Generates samples dynamically to save memory.
    Adapts input format for GL-Fusion paper architecture.
    """
    
    def __init__(self, node_sequences, task_df, task_split_ids, task_id, split_name, tokenizer_path, node_text_path, graph_data_path, sequence_length=48, prediction_horizon=1, config=None, description="Building samples", subsample_ratio=1.0):
        """
        Initialize the dataset.
        
        Args:
            node_sequences (dict): Dictionary of node sequences mapped by user ID.
            task_df (pd.DataFrame): DataFrame containing the raw task data.
            task_split_ids (set or list): Set or list containing the indices for this split.
            task_id (int): Task ID (1 or 2).
            split_name (str): Name of the split ('train' or 'val').
            tokenizer_path (str): Path to the HuggingFace tokenizer.
            node_text_path (str): Path to the node text descriptions JSON file.
            graph_data_path (str): Path to the graph_data.pt file (for edge_index).
            sequence_length (int): Length of input *trajectory* sequence.
            prediction_horizon (int): Number of future steps to predict.
            config (dict): Configuration dictionary.
            description (str): Description for the progress bar during sample building.
            subsample_ratio (float): Fraction of samples to keep (e.g., 0.15 for 15%). Defaults to 1.0 (no subsampling).
        """
        self.node_sequences = node_sequences
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.description = description
        self.task_id = task_id
        self.split_name = split_name
        self.config = config
        self.subsample_ratio = subsample_ratio
        
        # Control logging - only log from main process
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        
        # Set memory-efficient flag based on job size
        self.memory_efficient = True
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size >= 8:  # For large jobs, be extra careful with memory
                self.memory_efficient = True
        
        # Load tokenizer and add special tokens
        if is_main_process: logger.info(f"Loading tokenizer for dataset from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        special_tokens_dict = {'additional_special_tokens': ['<graph_start>', '<graph_end>', '<node>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        if is_main_process: logger.info(f"Added {num_added_toks} special tokens to tokenizer.")
        self.graph_start_token_id = self.tokenizer.convert_tokens_to_ids('<graph_start>')
        self.graph_end_token_id = self.tokenizer.convert_tokens_to_ids('<graph_end>')
        self.node_token_id = self.tokenizer.convert_tokens_to_ids('<node>')
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
             self.pad_token_id = self.tokenizer.eos_token_id # Use EOS if PAD is not set
             logger.warning(f"Tokenizer has no PAD token, using EOS token ({self.pad_token_id}) for padding.")

        # --- (NEW) Load Coordinate Scalers ---
        data_conf = config['data']
        processed_dir = Path(data_conf.get('processed_dir', 'data/processed')) # Use a default if not in config
        x_scaler_path = processed_dir / 'x_coordinate_scaler.pkl'
        y_scaler_path = processed_dir / 'y_coordinate_scaler.pkl'

        try:
            if is_main_process: logger.info(f"Loading coordinate scalers: X from {x_scaler_path}, Y from {y_scaler_path}")
            with open(x_scaler_path, 'rb') as f:
                self.x_scaler = pickle.load(f)
            with open(y_scaler_path, 'rb') as f:
                self.y_scaler = pickle.load(f)
            if is_main_process: logger.info("Coordinate scalers loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Coordinate scaler files not found in {processed_dir}. Ensure preprocess.py has been run and saved them.")
            # Fallback: create dummy scalers that do nothing if files are not found
            # This allows dataset creation to proceed but will result in unscaled (likely poor) training.
            logger.warning("Using dummy identity scalers as scaler files were not found. Coordinates will NOT be scaled.")
            self.x_scaler = MinMaxScaler() # Dummy scaler
            self.x_scaler.fit(np.array([[0],[1]])) # Fit with dummy range to avoid errors if transform is called
            self.y_scaler = MinMaxScaler() # Dummy scaler
            self.y_scaler.fit(np.array([[0],[1]]))
        except Exception as e:
            logger.error(f"Error loading coordinate scalers: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load coordinate scalers: {e}")
        # --- End (NEW) Load Coordinate Scalers ---

        # Determine max sequence lengths
        self.max_llm_seq_length = config['llm'].get('sequence_length', 512) 
        # Max length for the tokenized node text descriptions
        self.max_node_text_len = config['llm'].get('node_text_sequence_length', 64) # Add this to config? Default 64
        
        # Load node text descriptions
        if is_main_process: logger.info(f"Loading node text descriptions from {node_text_path}")
        with open(node_text_path, 'r') as f:
            # Load JSON mapping node_id (as string) to text
            self.node_text_descriptions = json.load(f)
        if is_main_process: logger.info(f"Loaded text descriptions for {len(self.node_text_descriptions)} nodes.")

        # Load graph data for subgraph extraction
        if is_main_process: logger.info(f"Loading graph data from {graph_data_path}")
        graph_data = torch.load(graph_data_path, map_location='cpu', weights_only=False)
        self.edge_index = graph_data['edge_index']
        if is_main_process: logger.info(f"Loaded graph with edge_index shape: {self.edge_index.shape}")
        
        # Create adjacency list for efficient subgraph extraction
        self.adjacency_dict = {}
        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            if src not in self.adjacency_dict:
                self.adjacency_dict[src] = []
            if dst not in self.adjacency_dict:
                self.adjacency_dict[dst] = []
            self.adjacency_dict[src].append(dst)
            self.adjacency_dict[dst].append(src)  # For undirected graph
        
        if is_main_process: 
            logger.info(f"Built adjacency list with {len(self.adjacency_dict)} nodes")

        # Get the indices for this split
        self.indices = set(task_split_ids)
        
        # Filter task_df once for efficiency and store grouped by uid
        if is_main_process: logger.info(f"Filtering and grouping task_df for {split_name} split...")
        
        # --- Process Coordinates (Memory Efficient) ---
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main_process: logger.info(f"Processing coordinate data for {self.split_name} split...")
        
        # --- Apply early user subsampling for training data ---
        # This significantly reduces memory usage and processing time
        if self.subsample_ratio < 1.0 and self.split_name == 'train':
            # Get all unique users from the task dataframe
            all_users = task_df[task_df.index.isin(self.indices)]['uid'].unique()
            total_users_before = len(all_users)
            
            # Apply subsampling to users
            target_users = int(total_users_before * self.subsample_ratio)
            if target_users < total_users_before:
                # Randomly sample users
                np.random.seed(42)  # For reproducibility
                users_to_process = np.random.choice(all_users, target_users, replace=False)
                if is_main_process:
                    logger.info(f"Early subsampling: selected {len(users_to_process)} users out of {total_users_before} for training ({self.subsample_ratio*100:.1f}%)")
            else:
                users_to_process = all_users
        else:
            # For validation/test, optionally limit users for faster development
            all_users = task_df[task_df.index.isin(self.indices)]['uid'].unique()
            
            # Check if we should limit validation users for faster development
            max_val_users = int(os.environ.get('MAX_VAL_USERS', '0'))
            if max_val_users > 0 and len(all_users) > max_val_users:
                np.random.seed(42)  # For reproducibility
                users_to_process = np.random.choice(all_users, max_val_users, replace=False)
                if is_main_process:
                    logger.info(f"Limiting {self.split_name} dataset to {max_val_users} users (out of {len(all_users)}) for faster development")
            else:
                users_to_process = all_users
        
        # Get filtered dataframe based on selected users
        df_filtered = task_df[task_df['uid'].isin(users_to_process) & task_df.index.isin(self.indices)]
        total_users = len(users_to_process)
        
        logger.info(f"Processing coordinate data for {total_users} users")
        
        # Use a dictionary comprehension with optimized numpy operations
        # This is much faster than pandas groupby for large dataframes
        self.user_coord_data = {}
        
        # Release memory as soon as possible
        del df_filtered

        # Determine optimal chunk size based on system memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            # Adjust chunk size based on available memory (smaller chunks for less memory)
            if available_memory < 4 * 1024**3:  # Less than 4GB
                chunk_size = 500
            elif available_memory < 8 * 1024**3:  # Less than 8GB
                chunk_size = 1000
            else:  # More than 8GB
                chunk_size = 2000
            if is_main_process:
                logger.info(f"Setting chunk size to {chunk_size} based on available memory ({available_memory / 1024**3:.1f} GB)")
        except:
            # Default chunk size if psutil fails
            chunk_size = 1000
        
        # Process in chunks with vectorized operations where possible
        num_chunks = (total_users + chunk_size - 1) // chunk_size  # ceiling division
        
        # Create a shared cache directory for coordinates if specified
        cache_dir = os.environ.get('DATASET_CACHE_DIR')
        cache_loaded = False
        
        if cache_dir:
            # Generate a unique cache path based on the dataset parameters
            # Include subsample ratio in cache key for training data
            if self.split_name == 'train' and self.subsample_ratio < 1.0:
                cache_path = f"{cache_dir}/coords_cache_{split_name}_{task_id}_{len(self.indices)}_sub{int(self.subsample_ratio*100)}.pt"
            else:
                cache_path = f"{cache_dir}/coords_cache_{split_name}_{task_id}_{len(self.indices)}.pt"
                
            if os.path.exists(cache_path):
                if is_main_process:
                    logger.info(f"Loading coordinate data from cache: {cache_path}")
                try:
                    cache_data = torch.load(cache_path, weights_only=False)
                    self.user_coord_data = cache_data['user_coord_data']
                    self.user_ids = cache_data.get('user_ids', [])
                    
                    # Verify cache is valid
                    if len(self.user_coord_data) > 0 and len(self.user_ids) > 0:
                        cache_loaded = True
                        if is_main_process:
                            logger.info(f"Successfully loaded coordinate data for {len(self.user_coord_data)} users from cache")
                    else:
                        if is_main_process:
                            logger.warning(f"Cache appears to be empty or corrupted, regenerating...")
                except Exception as e:
                    if is_main_process:
                        logger.warning(f"Failed to load coordinate cache, regenerating: {e}")
        
        # Only process if cache was not loaded
        if not cache_loaded:
            # Process the coordinates
            self._process_user_coordinates(users_to_process, task_df, chunk_size, num_chunks)
            
            # Get list of valid users (those with sufficient sequence length)
            self.user_ids = sorted([
                uid for uid in users_to_process
                if uid in self.node_sequences and len(self.node_sequences[uid]) >= self.sequence_length + self.prediction_horizon
            ])
            
            # Save to cache if cache directory is specified
            if cache_dir and is_main_process:
                logger.info(f"Saving coordinate data to cache: {cache_path}")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                cache_data = {
                    'user_coord_data': self.user_coord_data,
                    'user_ids': self.user_ids
                }
                torch.save(cache_data, cache_path)
        
        # Log the final user count
        if is_main_process: 
            logger.info(f"Found {len(self.user_ids)} users with valid sequences in {split_name} split.")
        
        # Garbage collect to free memory
        gc.collect()
        
        # --- Build Sample Identifiers (Memory Efficient) ---
        self._build_sample_identifiers(is_main_process)
    
    def _process_user_coordinates(self, users_to_process, task_df, chunk_size, num_chunks):
        """Process user coordinates in chunks for memory efficiency"""
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        
        # Convert to numpy once and create efficient lookup
        if is_main_process:
            logger.info("Converting dataframe to numpy arrays for efficient lookup...")
        
        # Create indexed data structure for fast lookup
        task_array = task_df[['uid', 'x', 'y']].values
        indices_set = set(self.indices)
        
        # Build a dictionary mapping uid to their row indices for O(1) lookup
        uid_to_rows = {}
        for idx, (uid, x, y) in enumerate(task_array):
            if idx in indices_set:
                if uid not in uid_to_rows:
                    uid_to_rows[uid] = []
                uid_to_rows[uid].append((x, y))
        
        # Process users in chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(users_to_process))
            chunk_users = users_to_process[start_idx:end_idx]
            
            for uid in tqdm(chunk_users, desc=f"Processing {self.split_name} coords (chunk {chunk_idx+1}/{num_chunks})"):
                if uid in uid_to_rows:
                    self.user_coord_data[uid] = np.array(uid_to_rows[uid])
            
            # Force garbage collection between chunks
            gc.collect()
    
    def _build_sample_identifiers(self, is_main_process):
        """Build sample identifiers for the dataset"""
        if is_main_process: logger.info(f"Building sample identifiers for {self.split_name} split...")
        
        # In memory-efficient mode, process users in batches
        full_sample_identifiers = []
        
        # Use optimized chunk-based processing
        if self.memory_efficient:
            # Determine chunk size based on number of users
            total_users = len(self.user_ids)
            if total_users > 5000:
                chunk_size = 500
            elif total_users > 2000:
                chunk_size = 800
            else:
                chunk_size = 1000
                
            # Process in parallel if possible
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                import multiprocessing
                
                # Only use parallel processing if memory efficient and we have enough users
                if total_users > 1000:
                    # Use at most 4 workers to avoid memory issues
                    n_workers = min(4, multiprocessing.cpu_count())
                    
                    if is_main_process:
                        logger.info(f"Processing sample identifiers in parallel with {n_workers} workers")
                    
                    def process_user_chunk(users_chunk):
                        chunk_identifiers = []
                        for uid in users_chunk:
                            user_node_seq = self.node_sequences[uid]
                            user_coords = self.user_coord_data.get(uid)
                            if user_coords is None: continue
                            max_target_node_index = len(user_node_seq) - 1
                            max_coord_index = len(user_coords) - 1
                            num_possible_samples = len(user_node_seq) - (self.sequence_length + self.prediction_horizon) + 1
                            
                            # Further optimize by only taking every nth sample if we have too many
                            stride = max(1, num_possible_samples // 1000) if num_possible_samples > 2000 else 1
                            
                            for i in range(0, num_possible_samples, stride):
                                target_node_index = i + self.sequence_length + self.prediction_horizon - 1
                                if target_node_index <= max_target_node_index and target_node_index <= max_coord_index:
                                    chunk_identifiers.append((uid, i))
                        
                        return chunk_identifiers
                    
                    # Split users into chunks for parallel processing
                    user_chunks = [self.user_ids[i:i+chunk_size] for i in range(0, len(self.user_ids), chunk_size)]
                    
                    # Process chunks in parallel
                    with ProcessPoolExecutor(max_workers=n_workers) as executor:
                        futures = [executor.submit(process_user_chunk, chunk) for chunk in user_chunks]
                        
                        # Collect results as they complete
                        for i, future in enumerate(as_completed(futures)):
                            chunk_identifiers = future.result()
                            full_sample_identifiers.extend(chunk_identifiers)
                            if is_main_process:
                                logger.info(f"Processed chunk {i+1}/{len(user_chunks)}, added {len(chunk_identifiers)} identifiers")
                    
                else:
                    # Fallback to sequential processing for smaller datasets
                    for i in range(0, len(self.user_ids), chunk_size):
                        chunk_users = self.user_ids[i:i+chunk_size]
                        chunk_identifiers = []
                        
                        for uid in tqdm(chunk_users, desc=f"{self.description} (chunk {i//chunk_size+1}/{(len(self.user_ids)+chunk_size-1)//chunk_size})"):
                            user_node_seq = self.node_sequences[uid]
                            user_coords = self.user_coord_data.get(uid)
                            if user_coords is None: continue
                            max_target_node_index = len(user_node_seq) - 1
                            max_coord_index = len(user_coords) - 1
                            num_possible_samples = len(user_node_seq) - (self.sequence_length + self.prediction_horizon) + 1
                            
                            # Further optimize by only taking every nth sample if we have too many
                            stride = 5 if num_possible_samples > 1000 else 1
                            
                            for i in range(0, num_possible_samples, stride):
                                target_node_index = i + self.sequence_length + self.prediction_horizon - 1
                                if target_node_index <= max_target_node_index and target_node_index <= max_coord_index:
                                    chunk_identifiers.append((uid, i))
                        
                        full_sample_identifiers.extend(chunk_identifiers)
                        # Force garbage collection between chunks
                        gc.collect()
            
            except (ImportError, Exception) as e:
                # Fall back to sequential processing if parallel processing fails
                if is_main_process:
                    logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
                
                for i in range(0, len(self.user_ids), chunk_size):
                    chunk_users = self.user_ids[i:i+chunk_size]
                    chunk_identifiers = []
                    
                    for uid in tqdm(chunk_users, desc=f"{self.description} (chunk {i//chunk_size+1}/{(len(self.user_ids)+chunk_size-1)//chunk_size})"):
                        user_node_seq = self.node_sequences[uid]
                        user_coords = self.user_coord_data.get(uid)
                        if user_coords is None: continue
                        max_target_node_index = len(user_node_seq) - 1
                        max_coord_index = len(user_coords) - 1
                        num_possible_samples = len(user_node_seq) - (self.sequence_length + self.prediction_horizon) + 1
                        
                        # Further optimize by only taking every 5th sample if we have a lot
                        stride = 5 if num_possible_samples > 1000 else 1
                        
                        for i in range(0, num_possible_samples, stride):
                            target_node_index = i + self.sequence_length + self.prediction_horizon - 1
                            if target_node_index <= max_target_node_index and target_node_index <= max_coord_index:
                                chunk_identifiers.append((uid, i))
                    
                    full_sample_identifiers.extend(chunk_identifiers)
                    # Force garbage collection between chunks
                    gc.collect()
        else:
            # Original non-chunked approach
            for uid in tqdm(self.user_ids, desc=self.description):
                user_node_seq = self.node_sequences[uid]
                user_coords = self.user_coord_data.get(uid)
                if user_coords is None: continue
                max_target_node_index = len(user_node_seq) - 1
                max_coord_index = len(user_coords) - 1
                num_possible_samples = len(user_node_seq) - (self.sequence_length + self.prediction_horizon) + 1
                for i in range(num_possible_samples):
                    target_node_index = i + self.sequence_length + self.prediction_horizon - 1
                    if target_node_index <= max_target_node_index and target_node_index <= max_coord_index:
                        full_sample_identifiers.append((uid, i))
                      
        total_samples = len(full_sample_identifiers)
        if is_main_process: logger.info(f"Built {total_samples} potential sample identifiers for {self.split_name}.")
        
        # --- Apply Subsampling --- 
        if self.subsample_ratio < 1.0 and self.split_name == 'train': # Only subsample training data
            target_samples = int(total_samples * self.subsample_ratio)
            if is_main_process: logger.info(f"Subsampling training data to {target_samples} samples ({self.subsample_ratio*100:.1f}%)...")
            if target_samples < total_samples:
                # Use reservoir sampling for memory efficiency with huge datasets
                if total_samples > 1000000:  # If more than 1M samples
                    self.sample_identifiers = self._reservoir_sample(full_sample_identifiers, target_samples)
                else:
                    self.sample_identifiers = random.sample(full_sample_identifiers, target_samples)
                
                if is_main_process: logger.info(f"Selected {len(self.sample_identifiers)} samples after subsampling.")
            else:
                if is_main_process: logger.warning(f"Subsample ratio ({self.subsample_ratio}) resulted in {target_samples} samples, which is not less than total {total_samples}. Using all samples.")
                self.sample_identifiers = full_sample_identifiers
        else:
            # For validation, also apply sampling if MAX_VAL_USERS is set for faster development
            if self.split_name == 'val':
                max_val_users = int(os.environ.get('MAX_VAL_USERS', '0'))
                if max_val_users > 0 and total_samples > 10000:  # Only subsample if we have many samples
                    # Apply aggressive subsampling for development - aim for ~5000 samples max
                    target_val_samples = min(5000, total_samples)
                    if is_main_process:
                        logger.info(f"Subsampling validation data to {target_val_samples} samples for faster development (from {total_samples} total)")
                    self.sample_identifiers = random.sample(full_sample_identifiers, target_val_samples)
                else:
                    self.sample_identifiers = full_sample_identifiers
            else:
                self.sample_identifiers = full_sample_identifiers # Use all samples if ratio is 1.0 or not training split
                
            if is_main_process:
                if self.split_name == 'train':
                    logger.info(f"Using all {len(self.sample_identifiers)} training samples (subsample_ratio={self.subsample_ratio}).")
                else:
                    logger.info(f"Using all {len(self.sample_identifiers)} samples for {self.split_name} split (no subsampling applied).")
        
        del full_sample_identifiers # Clean up memory
        gc.collect()  # Force garbage collection
    
    def _reservoir_sample(self, population, k):
        """
        Reservoir sampling algorithm for memory-efficient random sampling.
        This is much more memory efficient than random.sample() for very large populations.
        """
        result = population[:k].copy()  # Fill the reservoir with the first k items
        for i in range(k, len(population)):
            # Random chance of replacing an item in the reservoir
            j = random.randrange(i + 1)
            if j < k:
                result[j] = population[i]
        return result
    
    def __len__(self):
        return len(self.sample_identifiers)
    
    @property
    def collate_fn(self):
        """Returns the custom collate function for this dataset."""
        return collate_trajectories
    
    def __getitem__(self, idx):
        uid, start_index = self.sample_identifiers[idx]
        user_node_seq = self.node_sequences[uid]
        
        # Extract the original trajectory node sequence segment
        trajectory_node_ids = user_node_seq[start_index : start_index + self.sequence_length]
        
        # Get unique nodes relevant to this trajectory segment
        unique_nodes_in_segment = sorted(list(set(trajectory_node_ids)))
        num_nodes = len(unique_nodes_in_segment)
        
        # Construct the LLM input sequence: [<graph_start>, <node> * num_nodes, <graph_end>]
        llm_input_ids_list = [
            self.graph_start_token_id
        ] + [
            self.node_token_id
        ] * num_nodes + [
            self.graph_end_token_id
        ]
        
        # --- Get and tokenize node text descriptions ---
        node_texts = [self.node_text_descriptions.get(str(node_id), f"Node {node_id} text missing") 
                      for node_id in unique_nodes_in_segment]
        tokenized_node_texts = self.tokenizer(
            node_texts, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_node_text_len, 
            return_tensors="pt"
        )
        node_text_input_ids = tokenized_node_texts['input_ids'] # Shape: [num_nodes, max_node_text_len]
        node_text_attention_mask = tokenized_node_texts['attention_mask'] # Shape: [num_nodes, max_node_text_len]
        # ---------------------------------------------

        # Truncate main sequence if exceeding max LLM sequence length
        if len(llm_input_ids_list) > self.max_llm_seq_length:
             max_nodes_allowed = self.max_llm_seq_length - 2
             llm_input_ids_list = llm_input_ids_list[:1] + llm_input_ids_list[1:1+max_nodes_allowed] + llm_input_ids_list[-1:]
             num_nodes = max_nodes_allowed # Update node count
             unique_nodes_in_segment = unique_nodes_in_segment[:num_nodes]
             # Truncate node text data as well
             node_text_input_ids = node_text_input_ids[:num_nodes]
             node_text_attention_mask = node_text_attention_mask[:num_nodes]
             logger.warning(f"LLM input sequence truncated for sample {idx} to {self.max_llm_seq_length} tokens.")
             
        # Convert main sequence to tensor
        llm_input_ids_tensor = torch.tensor(llm_input_ids_list, dtype=torch.long)
        attention_mask = torch.ones_like(llm_input_ids_tensor)
        
        # Target coordinates
        target_coord_index = start_index + self.sequence_length + self.prediction_horizon - 1
        target_coords_array = self.user_coord_data[uid][target_coord_index]
        target_coords = torch.tensor(target_coords_array, dtype=torch.float)
        
        # --- (NEW) Scale Target Coordinates ---
        # Scalers expect shape (n_samples, n_features), so reshape
        scaled_x = self.x_scaler.transform(target_coords_array[0].reshape(-1, 1))
        scaled_y = self.y_scaler.transform(target_coords_array[1].reshape(-1, 1))
        # Reconstruct the scaled target_coords tensor
        target_coords_scaled = torch.tensor([scaled_x[0,0], scaled_y[0,0]], dtype=torch.float)
        # --- End (NEW) Scale Target Coordinates ---
        
        # --- Get subgraph edge index --- 
        # Find edges where both source and target are in unique_nodes_in_segment
        # This requires mapping unique_nodes_in_segment back to their original IDs if necessary
        # Create a mapping from original node ID to its index within unique_nodes_in_segment
        node_id_to_local_idx = {node_id: i for i, node_id in enumerate(unique_nodes_in_segment)}
        
        subgraph_edge_indices = []
        node_id_set = set(unique_nodes_in_segment)
        
        # Use adjacency list for efficient edge lookup
        seen_edges = set()  # To avoid duplicate edges
        for node_id in unique_nodes_in_segment:
            if node_id in self.adjacency_dict:
                for neighbor in self.adjacency_dict[node_id]:
                    if neighbor in node_id_set:
                        # Create edge tuple (ensure consistent ordering to avoid duplicates)
                        edge = (min(node_id, neighbor), max(node_id, neighbor))
                        if edge not in seen_edges:
                            seen_edges.add(edge)
                            local_src = node_id_to_local_idx[node_id]
                            local_dst = node_id_to_local_idx[neighbor]
                            subgraph_edge_indices.append([local_src, local_dst])
        
        subgraph_edge_index = torch.tensor(subgraph_edge_indices, dtype=torch.long).t().contiguous()
        if subgraph_edge_index.numel() == 0: # Handle case with no edges within the subgraph
             subgraph_edge_index = torch.empty((2, 0), dtype=torch.long)
        # ----------------------------- 

        # --- Create LLM-aligned node sequence mapping --- START
        # Length should match llm_input_ids_tensor BEFORE padding
        llm_aligned_mapping = [-1] * len(llm_input_ids_tensor) # Initialize with -1 (padding/special token indicator)
        # Fill in the original node IDs for the <node> token positions
        for i in range(num_nodes): # num_nodes already reflects potential truncation
            llm_token_position = 1 + i # <node> tokens start at index 1
            if llm_token_position < len(llm_aligned_mapping): # Ensure we don't go out of bounds if truncated
                llm_aligned_mapping[llm_token_position] = unique_nodes_in_segment[i]
        llm_aligned_node_mapping = torch.tensor(llm_aligned_mapping, dtype=torch.long)
        # --- Create LLM-aligned node sequence mapping --- END

        return {
            'llm_input_ids': llm_input_ids_tensor,
            'attention_mask': attention_mask,
            'graph_node_original_ids': torch.tensor(unique_nodes_in_segment, dtype=torch.long),
            'node_token_indices': torch.arange(1, 1 + num_nodes, dtype=torch.long), 
            'node_text_input_ids': node_text_input_ids, # Added
            'node_text_attention_mask': node_text_attention_mask, # Added
            'subgraph_edge_index': subgraph_edge_index, # Added for structure-aware layer
            'llm_aligned_node_mapping': llm_aligned_node_mapping, # NEW: Mapping aligned with LLM sequence
            'target_coords': target_coords_scaled, # USE SCALED TARGETS
            'pad_token_id': self.pad_token_id # Pass pad token id for collate fn
        }

def collate_trajectories(batch):
    """
    Collate function for trajectory batches.
    Adapts to the new input format from TrajectoryDataset.
    Handles padding for llm_input_ids, node_text_input_ids, and the new llm_aligned_node_mapping.
    """
    max_llm_len = max(len(sample['llm_input_ids']) for sample in batch)
    # Max number of nodes in any sample in the batch
    max_num_nodes = max(len(sample['graph_node_original_ids']) for sample in batch)
    # Max length of node_token_indices (should be same as max_num_nodes)
    max_node_indices_len = max(len(sample['node_token_indices']) for sample in batch)
    # Max length of tokenized node text
    max_node_text_len = max(sample['node_text_input_ids'].shape[1] for sample in batch if sample['node_text_input_ids'].numel() > 0) if batch else 0
    
    batch_llm_input_ids = []
    batch_llm_attention_masks = []
    batch_graph_node_original_ids = [] # Keep as list for now
    batch_node_token_indices = []
    batch_node_text_input_ids = []
    batch_node_text_attention_mask = []
    batch_subgraph_edge_index = [] # Store as list
    batch_llm_aligned_node_mapping = [] # NEW: List for padded aligned mappings
    batch_target_coords = []
    
    pad_token_id = batch[0]['pad_token_id'] if batch else 0
    node_mapping_pad_value = -1 # Use -1 for padding the node mapping

    for sample in batch:
        # --- Pad LLM sequence --- 
        llm_ids = sample['llm_input_ids']
        llm_mask = sample['attention_mask']
        aligned_mapping = sample['llm_aligned_node_mapping'] # Get the new mapping
        current_llm_len = len(llm_ids)
        llm_padding_len = max_llm_len - current_llm_len
        
        if llm_padding_len > 0:
            pad_llm_ids = torch.full((llm_padding_len,), pad_token_id, dtype=torch.long)
            llm_ids = torch.cat([llm_ids, pad_llm_ids])
            pad_llm_mask = torch.zeros(llm_padding_len, dtype=torch.long)
            llm_mask = torch.cat([llm_mask, pad_llm_mask])
            # Pad the aligned mapping as well
            pad_aligned_mapping = torch.full((llm_padding_len,), node_mapping_pad_value, dtype=torch.long)
            aligned_mapping = torch.cat([aligned_mapping, pad_aligned_mapping])
        elif llm_padding_len < 0:
             llm_ids = llm_ids[:max_llm_len]
             llm_mask = llm_mask[:max_llm_len]
             aligned_mapping = aligned_mapping[:max_llm_len] # Truncate mapping too
            
        batch_llm_input_ids.append(llm_ids.unsqueeze(0))
        batch_llm_attention_masks.append(llm_mask.unsqueeze(0))
        batch_llm_aligned_node_mapping.append(aligned_mapping.unsqueeze(0)) # Add padded mapping
        
        # --- Pad Node Text sequence --- 
        node_text_ids = sample['node_text_input_ids']
        node_text_mask = sample['node_text_attention_mask']
        current_num_nodes = node_text_ids.shape[0]
        current_text_len = node_text_ids.shape[1]
        
        # Pad number of nodes
        node_padding_len = max_num_nodes - current_num_nodes
        if node_padding_len > 0:
             pad_node_text_ids = torch.full((node_padding_len, current_text_len), pad_token_id, dtype=torch.long)
             node_text_ids = torch.cat([node_text_ids, pad_node_text_ids], dim=0)
             pad_node_text_mask = torch.zeros((node_padding_len, current_text_len), dtype=torch.long)
             node_text_mask = torch.cat([node_text_mask, pad_node_text_mask], dim=0)
             
        # Pad text length (if needed, though usually handled by tokenizer max_length)
        text_padding_len = max_node_text_len - current_text_len
        if text_padding_len > 0:
             pad_text_ids = torch.full((max_num_nodes, text_padding_len), pad_token_id, dtype=torch.long)
             node_text_ids = torch.cat([node_text_ids, pad_text_ids], dim=1)
             pad_text_mask = torch.zeros((max_num_nodes, text_padding_len), dtype=torch.long)
             node_text_mask = torch.cat([node_text_mask, pad_text_mask], dim=1)
        elif text_padding_len < 0: # Truncate if somehow longer
             node_text_ids = node_text_ids[:, :max_node_text_len]
             node_text_mask = node_text_mask[:, :max_node_text_len]

        batch_node_text_input_ids.append(node_text_ids.unsqueeze(0))
        batch_node_text_attention_mask.append(node_text_mask.unsqueeze(0))
        
        # --- Pad Node Indices sequence --- 
        node_indices = sample['node_token_indices']
        current_node_indices_len = len(node_indices)
        node_indices_padding_len = max_node_indices_len - current_node_indices_len
        
        if node_indices_padding_len > 0:
            # Pad with 0, assuming 0 is not a valid index for node tokens
            pad_node_indices = torch.zeros(node_indices_padding_len, dtype=torch.long)
            node_indices = torch.cat([node_indices, pad_node_indices])
        elif node_indices_padding_len < 0:
             node_indices = node_indices[:max_node_indices_len]
             
        batch_node_token_indices.append(node_indices.unsqueeze(0))
        
        # --- Pad graph info (or keep as list) --- 
        # Padding graph_node_original_ids and node_token_indices might be needed depending on model usage
        # For now, keep as list, but if model needs tensors, padding similar to node_text is required.
        batch_graph_node_original_ids.append(sample['graph_node_original_ids']) 

        # --- Add edge index and target --- 
        batch_subgraph_edge_index.append(sample['subgraph_edge_index']) # Keep as list
        batch_target_coords.append(sample['target_coords'].unsqueeze(0))
    
    # Stack tensors
    batch_llm_input_ids = torch.cat(batch_llm_input_ids, dim=0)
    batch_llm_attention_masks = torch.cat(batch_llm_attention_masks, dim=0)
    batch_node_text_input_ids = torch.cat(batch_node_text_input_ids, dim=0) # Shape: [B, max_num_nodes, max_node_text_len]
    batch_node_text_attention_mask = torch.cat(batch_node_text_attention_mask, dim=0) # Shape: [B, max_num_nodes, max_node_text_len]
    batch_node_token_indices = torch.cat(batch_node_token_indices, dim=0) # Stacked tensor
    batch_llm_aligned_node_mapping = torch.cat(batch_llm_aligned_node_mapping, dim=0) # NEW: Stacked aligned mapping
    batch_target_coords = torch.cat(batch_target_coords, dim=0)
    
    # Note: graph_node_original_ids, subgraph_edge_index are still lists of tensors.
    # The model will need to handle this format or collate_fn needs further modification.
    
    return {
        'input_ids': batch_llm_input_ids, # Rename for consistency with model forward args
        'attention_mask': batch_llm_attention_masks,
        'graph_node_original_ids': batch_graph_node_original_ids, 
        'node_token_indices': batch_node_token_indices, 
        'node_text_input_ids': batch_node_text_input_ids,
        'node_text_attention_mask': batch_node_text_attention_mask,
        'subgraph_edge_index': batch_subgraph_edge_index,
        'node_sequence_mapping': batch_llm_aligned_node_mapping, # Rename for consistency with model forward args
        'target_positions': batch_target_coords # Rename for consistency with model forward args
    }


def file_exists_timeout(filepath, timeout=300):
    """
    Wait for a file to exist with a timeout, and verify it's complete.
    
    Args:
        filepath: Path to the file to check
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if file exists and is complete, False if timeout
    """
    import time
    start_time = time.time()
    last_size = -1
    stable_count = 0
    
    while time.time() - start_time < timeout:
        if os.path.exists(filepath):
            try:
                # Check if file size is stable (not being written to)
                current_size = os.path.getsize(filepath)
                if current_size > 0:  # File has content
                    if current_size == last_size:
                        stable_count += 1
                        if stable_count >= 3:  # File size stable for 3 checks (15 seconds)
                            # Try to load a small portion to verify it's a valid torch file
                            try:
                                # Quick check: try to read the file header
                                with open(filepath, 'rb') as f:
                                    header = f.read(8)  # Read first 8 bytes
                                    if len(header) == 8:  # File is readable and has content
                                        return True
                            except Exception:
                                pass  # File might still be being written
                    else:
                        stable_count = 0
                        last_size = current_size
            except OSError:
                pass  # File might be locked or still being written
        time.sleep(5)  # Check every 5 seconds
    return False


def load_data_for_training(config, task_id=1, rank=0, world_size=1):
    """
    Loads data for training, handling distributed setups and caching.
    Ensures that if data loading/creation fails, it raises an error rather than returning None datasets.
    """
    import os  # Import os at the beginning of function
    logger.info(f"[Rank {rank}/{world_size}] Initializing load_data_for_training for task {task_id}.")
    
    train_dataset, val_dataset, test_dataset = None, None, None
    train_sampler, val_sampler, test_sampler = None, None, None
    
    seed = config['training'].get('seed', 42)
    set_seed(seed + rank)
    
    data_config = config['data']
    raw_dir = Path(data_config['raw_dir'])
    processed_dir = Path(data_config['processed_dir'])
    is_main_process = (rank == 0)

    subsample_ratio_train = float(os.environ.get('DATA_SUBSAMPLE_RATIO', data_config.get('train_data_subsample_ratio', 1.0)))
    subsample_ratio_val = float(os.environ.get('VAL_DATA_SUBSAMPLE_RATIO', data_config.get('val_data_subsample_ratio', 1.0)))
    if is_main_process:
        logger.info(f"Using training data subsample ratio: {subsample_ratio_train}")
        logger.info(f"Using validation data subsample ratio: {subsample_ratio_val}")

    cache_base_dir = Path(os.environ.get("DATASET_CACHE_DIR", processed_dir))
    os.makedirs(cache_base_dir, exist_ok=True)
    
    train_cache_suffix = f"_task{task_id}_train_ratio{subsample_ratio_train:.4f}.pt"
    val_cache_suffix = f"_task{task_id}_val_ratio{subsample_ratio_val:.4f}.pt"
    test_cache_suffix = f"_task{task_id}_test_fulldata.pt"

    train_dataset_path = cache_base_dir / f"trajectory_dataset{train_cache_suffix}"
    val_dataset_path = cache_base_dir / f"trajectory_dataset{val_cache_suffix}"
    test_dataset_path = cache_base_dir / f"trajectory_dataset{test_cache_suffix}"

    node_text_path = processed_dir / 'node_text_descriptions.json'
    graph_data_path = processed_dir / 'graph_data.pt'
    tokenizer_path = config['llm'].get('local_model_path', config['llm']['model_name'])

    try:
        task_df_path = raw_dir / f'task{task_id}_dataset_kotae.csv'
        if not task_df_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {task_df_path}")
        task_df = pd.read_csv(task_df_path)

        train_val_split_path = processed_dir / f'train_val_split_task{task_id}.pkl'
        test_split_path = processed_dir / f'test_split_task{task_id}.pkl' 
        if not train_val_split_path.exists():
            raise FileNotFoundError(f"Train/Val split file not found: {train_val_split_path}. Run preprocess.py.")
        
        with open(train_val_split_path, 'rb') as f: 
            train_val_split = pickle.load(f)
        train_split_ids = train_val_split.get('train', [])
        val_split_ids = train_val_split.get('val', [])

        test_split_ids = [] 
        if test_split_path.exists():
            with open(test_split_path, 'rb') as f: 
                test_split_ids = pickle.load(f).get('test', [])
    except Exception as e:
        logger.error(f"[Rank {rank}] Failed to load essential raw data or split files: {e}", exc_info=True)
        raise RuntimeError(f"[Rank {rank}] Prerequisite data files missing or unreadable. Ensure preprocessing is complete.") from e
    
    node_sequences_data = None # Initialize node_sequences_data
    if is_main_process:
        logger.info(f"[Rank {rank}] Main process: attempting to create/cache datasets.")
        try:
            llm_prepared_dir = Path(data_config['llm_prepared_dir'])
            node_sequences_file_path = llm_prepared_dir / f'llm_sequences_train_val_task{task_id}.pt'
            if not node_sequences_file_path.exists():
                err_msg = f"[Rank {rank}] Node sequences file not found: {node_sequences_file_path}. Run preprocess.py to generate it."
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            logger.info(f"[Rank {rank}] Loading node sequences from {node_sequences_file_path}")
            try:
                node_sequences_data = torch.load(node_sequences_file_path, map_location='cpu', weights_only=False)
                if not isinstance(node_sequences_data, dict) or 'train' not in node_sequences_data or 'val' not in node_sequences_data:
                    err_msg = f"[Rank {rank}] Node sequences data from {node_sequences_file_path} is not a valid dictionary or missing 'train'/'val' keys."
                    logger.error(err_msg)
                    raise ValueError(err_msg)
                logger.info(f"[Rank {rank}] Node sequences loaded. Keys: {list(node_sequences_data.keys())}")
            except Exception as e:
                logger.error(f"[Rank {rank}] Failed to load node_sequences_data from {node_sequences_file_path}: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load node_sequences_data: {e}") from e

            logger.info(f"[Rank {rank}] Creating training dataset (Task {task_id}). Cache: {train_dataset_path}")
            train_dataset = TrajectoryDataset(
                node_sequences=node_sequences_data['train'], 
                task_df=task_df, 
                task_split_ids=train_split_ids, 
                task_id=task_id, 
                split_name='train',
                tokenizer_path=tokenizer_path, 
                node_text_path=node_text_path, 
                graph_data_path=graph_data_path,
                sequence_length=data_config['sequence_length'], 
                prediction_horizon=config['evaluation']['prediction_horizon'],
                config=config, 
                subsample_ratio=subsample_ratio_train, 
                description=f"[Rank {rank}] Building train samples"
            )
            if not train_dataset or len(train_dataset) == 0:
                logger.warning(f"[Rank {rank}] Training dataset is empty or invalid after creation.")
            logger.info(f"[Rank {rank}] Training dataset created with {len(train_dataset) if train_dataset else 0} samples. Saving to cache.")
            torch.save(train_dataset, train_dataset_path)
            with open(train_dataset_path, 'r+b') as f:
                os.fsync(f.fileno())
            logger.info(f"[Rank {rank}] Training dataset cache file written and synced to disk.")

            logger.info(f"[Rank {rank}] Creating validation dataset (Task {task_id}). Cache: {val_dataset_path}")
            val_dataset = TrajectoryDataset(
                node_sequences=node_sequences_data['val'], 
                task_df=task_df, 
                task_split_ids=val_split_ids, 
                task_id=task_id, 
                split_name='val',
                tokenizer_path=tokenizer_path, 
                node_text_path=node_text_path, 
                graph_data_path=graph_data_path,
                sequence_length=data_config['sequence_length'], 
                prediction_horizon=config['evaluation']['prediction_horizon'],
                config=config, 
                subsample_ratio=subsample_ratio_val, 
                description=f"[Rank {rank}] Building val samples"
            )
            if not val_dataset or len(val_dataset) == 0:
                logger.warning(f"[Rank {rank}] Validation dataset is empty or invalid after creation.")
            logger.info(f"[Rank {rank}] Validation dataset created with {len(val_dataset) if val_dataset else 0} samples. Saving to cache.")
            torch.save(val_dataset, val_dataset_path)
            with open(val_dataset_path, 'r+b') as f:
                os.fsync(f.fileno())
            logger.info(f"[Rank {rank}] Validation dataset cache file written and synced to disk.")
            
            if config['training'].get('prepare_test_dataset_too', False) and test_split_ids:
                logger.info(f"[Rank {rank}] Creating test dataset (Task {task_id}). Cache: {test_dataset_path}")
                test_dataset = TrajectoryDataset(
                    node_sequences=node_sequences_data.get('test', {}), 
                    task_df=task_df, 
                    task_split_ids=test_split_ids, 
                    task_id=task_id, 
                    split_name='test',
                    tokenizer_path=tokenizer_path, 
                    node_text_path=node_text_path, 
                    graph_data_path=graph_data_path,
                    sequence_length=data_config['sequence_length'], 
                    prediction_horizon=config['evaluation']['prediction_horizon'],
                    config=config, 
                    subsample_ratio=1.0, 
                    description=f"[Rank {rank}] Building test samples"
                )
                if test_dataset and len(test_dataset) > 0:
                    logger.info(f"[Rank {rank}] Test dataset created with {len(test_dataset)} samples. Saving to cache.")
                    torch.save(test_dataset, test_dataset_path)
                    with open(test_dataset_path, 'r+b') as f:
                        os.fsync(f.fileno())
                    logger.info(f"[Rank {rank}] Test dataset cache file written and synced to disk.")
                else:
                    logger.warning(f"[Rank {rank}] Test dataset is empty or invalid after creation. Not saving.")
                    test_dataset = None
            else:
                test_dataset = None 
            logger.info(f"[Rank {rank}] Datasets created and cached successfully by main process.")
        except Exception as e:
            logger.error(f"[Rank {rank}] Error during dataset creation/caching by main process: {e}", exc_info=True)
            train_dataset, val_dataset, test_dataset = None, None, None 
            raise RuntimeError(f"Rank {rank} (main process) failed to create/cache datasets: {e}") from e

    if world_size > 1:
        logger.info(f"[Rank {rank}] Waiting at barrier after dataset creation/before loading...")
        dist.barrier()
        logger.info(f"[Rank {rank}] Passed barrier after dataset creation.")

    if not is_main_process: # Non-main processes load from cache
        logger.info(f"[Rank {rank}] Non-main process: attempting to load datasets from cache.")
        # Ensure node_sequences_data is loaded if not already (e.g. if main process failed before caching it)
        # This part is tricky because node_sequences_data is needed by TrajectoryDataset
        # For simplicity here, we assume main process succeeded if we reach this point after barrier.
        # A more robust solution might involve broadcasting node_sequences_data or having all ranks load it.
        
        if not file_exists_timeout(train_dataset_path, timeout=600):
            err_msg = (f"[Rank {rank}] Training dataset cache file not found after waiting: {train_dataset_path}. "
                       f"Ensure rank 0 completes dataset creation and caching.")
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        if not file_exists_timeout(val_dataset_path, timeout=600):
            err_msg = (f"[Rank {rank}] Validation dataset cache file not found after waiting: {val_dataset_path}. "
                       f"Ensure rank 0 completes dataset creation and caching.")
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        try:
            logger.info(f"[Rank {rank}] Loading training dataset from cache: {train_dataset_path}")
            train_dataset = torch.load(train_dataset_path, map_location='cpu', weights_only=False)
            if not train_dataset or len(train_dataset) == 0: 
                logger.warning(f"[Rank {rank}] Loaded training dataset is empty or invalid.")
            else: 
                logger.info(f"[Rank {rank}] Training dataset loaded with {len(train_dataset)} samples.")

            logger.info(f"[Rank {rank}] Loading validation dataset from cache: {val_dataset_path}")
            val_dataset = torch.load(val_dataset_path, map_location='cpu', weights_only=False)
            if not val_dataset or len(val_dataset) == 0: 
                logger.warning(f"[Rank {rank}] Loaded validation dataset is empty or invalid.")
            else: 
                logger.info(f"[Rank {rank}] Validation dataset loaded with {len(val_dataset)} samples.")
            
            if config['training'].get('prepare_test_dataset_too', False) and test_split_path.exists() and test_dataset_path.exists(): # check if test dataset should be loaded
                logger.info(f"[Rank {rank}] Loading test dataset from cache: {test_dataset_path}")
                test_dataset = torch.load(test_dataset_path, map_location='cpu', weights_only=False)
                if test_dataset and len(test_dataset) > 0: 
                    logger.info(f"[Rank {rank}] Test dataset loaded with {len(test_dataset)} samples.")
                else: 
                    logger.warning(f"[Rank {rank}] Loaded test dataset is empty or invalid.")
            else:
                test_dataset = None # Ensure test_dataset is None if not loaded
            logger.info(f"[Rank {rank}] Datasets loaded from cache successfully by non-main process.")
        except Exception as e:
            logger.error(f"[Rank {rank}] Error loading datasets from cache by non-main process: {e}", exc_info=True)
            train_dataset, val_dataset, test_dataset = None, None, None
            raise RuntimeError(f"Rank {rank} (non-main process) failed to load datasets from cache: {e}") from e

    log_message_parts = [
        f"[Rank {rank}] Final check before sampler creation:",
        f"Train len: {len(train_dataset) if train_dataset else 'None/Empty'}",
        f"Val len: {len(val_dataset) if val_dataset else 'None/Empty'}",
        f"Test len: {len(test_dataset) if test_dataset else 'None/Empty'}."
    ]
    logger.info(" ".join(log_message_parts))

    shuffle_train = config['training'].get('shuffle_train', True)
    if world_size > 1: 
        if train_dataset and len(train_dataset) > 0:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle_train, seed=seed)
        else:
            logger.warning(f"[Rank {rank}] Train dataset is None or empty, cannot create DistributedSampler.") # Changed to warning
            train_sampler = None # Ensure sampler is None
        if val_dataset and len(val_dataset) > 0:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed)
        else:
            logger.warning(f"[Rank {rank}] Val dataset is None or empty, cannot create DistributedSampler.") # Changed to warning
            val_sampler = None # Ensure sampler is None
        if test_dataset and len(test_dataset) > 0:
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed)
        else: # Ensure sampler is None if dataset is None or empty
            test_sampler = None
    else: # Single process
        if train_dataset and len(train_dataset) > 0:
            train_sampler = RandomSampler(train_dataset) if shuffle_train else SequentialSampler(train_dataset)
        else:
            logger.warning(f"[Rank {rank}] Train dataset is None or empty, cannot create Sampler.") # Changed to warning
            train_sampler = None
        if val_dataset and len(val_dataset) > 0:
            val_sampler = SequentialSampler(val_dataset)
        else:
            logger.warning(f"[Rank {rank}] Val dataset is None or empty, cannot create Sampler.") # Changed to warning
            val_sampler = None
        if test_dataset and len(test_dataset) > 0:
            test_sampler = SequentialSampler(test_dataset)
        else:
            test_sampler = None


    logger.info(f"[Rank {rank}] load_data_for_training complete. Train sampler: {'Exists' if train_sampler else 'None'}. Val sampler: {'Exists' if val_sampler else 'None'}. Test sampler: {'Exists' if test_sampler else 'None'}")
    return train_dataset, val_dataset, test_dataset, train_sampler, val_sampler, test_sampler


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics.
    """
    mse = torch.mean(torch.sum((predictions - targets) ** 2, dim=1)).item()
    euclidean_dist = torch.mean(torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))).item()
    return {
        'mse': mse,
        'euclidean_dist': euclidean_dist
    } 

def log_model_info(model, config, rank):
    """
    Log model information.
    """
    # Check quantization config
    if hasattr(config, 'quantization_config') and config.quantization_config:
        is_quantized = True
        quant_config = config.quantization_config
        # Use getattr for safer access
        bits = '4bit' if getattr(quant_config, 'load_in_4bit', False) else \
               ('8bit' if getattr(quant_config, 'load_in_8bit', False) else 'Unknown')
        quant_type = getattr(quant_config, 'bnb_4bit_quant_type', 'nf4') # Default nf4 is common
        compute_dtype = getattr(quant_config, 'bnb_4bit_compute_dtype', 'float16') # Default float16

        quantization_info = f"{bits} (type: {quant_type}, compute: {compute_dtype})"
        logger.info(f"[Rank {rank}] Model is quantized: {quantization_info}")
    else:
        logger.info(f"[Rank {rank}] LLM Config details not fully accessible.")

    # Count trainable vs frozen parameters
    # ... existing code ... 