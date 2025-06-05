"""
Preprocessing script for Human Mobility data.
Implements steps 4.1-4.6 as described in the project plan.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import math # Added for ceil
from sklearn.preprocessing import MinMaxScaler # Added for coordinate scaling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/model_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    """
    Step 4.1: Load raw data files.
    
    Args:
        config: Dictionary containing configuration parameters.
        
    Returns:
        Tuple of pandas DataFrames for task1, task2, cell_poi, and poi_categories.
    """
    logger.info("Loading raw data files...")
    
    raw_dir = config['data']['raw_dir']
    
    # Load the mobility data for both tasks
    task1_path = os.path.join(raw_dir, 'task1_dataset_kotae.csv')
    task2_path = os.path.join(raw_dir, 'task2_dataset_kotae.csv')
    
    # Load the POI data
    poi_path = os.path.join(raw_dir, 'cell_POIcat.csv')
    poi_categories_path = os.path.join(raw_dir, 'POI_datacategories.csv')
    
    # Read the CSV files
    task1_df = pd.read_csv(task1_path)
    task2_df = pd.read_csv(task2_path)
    cell_poi_df = pd.read_csv(poi_path)
    poi_categories_df = pd.read_csv(poi_categories_path)
    
    logger.info(f"Loaded task1 data: {task1_df.shape} rows")
    logger.info(f"Loaded task2 data: {task2_df.shape} rows")
    logger.info(f"Loaded cell POI data: {cell_poi_df.shape} rows")
    logger.info(f"Loaded POI categories data: {poi_categories_df.shape} rows")
    
    return task1_df, task2_df, cell_poi_df, poi_categories_df

def define_data_splits(task1_df, task2_df, config):
    """
    Step 4.2: Define training/validation/test splits.
    
    Args:
        task1_df: DataFrame containing task1 data.
        task2_df: DataFrame containing task2 data.
        config: Dictionary containing configuration parameters.
        
    Returns:
        Dictionaries containing indices for train/val/test splits for both tasks.
    """
    logger.info("Defining data splits...")
    
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    # Task 1: Test set is for users 80000-99999, days 60-74
    task1_test_mask = (task1_df['uid'] >= 80000) & (task1_df['uid'] <= 99999) & (task1_df['d'] >= 60) & (task1_df['d'] <= 74)
    task1_test_indices = task1_df[task1_test_mask].index.tolist()
    
    # Task 2: Test set is for users 22500-24999, days 60-74
    task2_test_mask = (task2_df['uid'] >= 22500) & (task2_df['uid'] <= 24999) & (task2_df['d'] >= 60) & (task2_df['d'] <= 74)
    task2_test_indices = task2_df[task2_test_mask].index.tolist()
    
    # The remaining data is for training/validation
    task1_train_val_mask = ~task1_test_mask
    task1_train_val_indices = task1_df[task1_train_val_mask].index.tolist()
    
    task2_train_val_mask = ~task2_test_mask
    task2_train_val_indices = task2_df[task2_train_val_mask].index.tolist()
    
    # Further split into training and validation (80/20 split)
    # For simplicity, we'll use a random split here
    # In a more sophisticated approach, we might want to stratify by user or time period
    np.random.seed(42)  # For reproducibility
    
    # Task 1 train/val split
    n_val_task1 = int(len(task1_train_val_indices) * 0.2)
    val_indices_task1 = np.random.choice(task1_train_val_indices, size=n_val_task1, replace=False)
    val_indices_task1_set = set(val_indices_task1) # Convert to set for faster lookup
    train_indices_task1 = [idx for idx in task1_train_val_indices if idx not in val_indices_task1_set]
    
    # Task 2 train/val split
    n_val_task2 = int(len(task2_train_val_indices) * 0.2)
    val_indices_task2 = np.random.choice(task2_train_val_indices, size=n_val_task2, replace=False)
    val_indices_task2_set = set(val_indices_task2) # Convert to set for faster lookup
    train_indices_task2 = [idx for idx in task2_train_val_indices if idx not in val_indices_task2_set]
    
    # Create dictionaries to store the splits
    task1_splits = {
        'train': train_indices_task1,
        'val': val_indices_task1,
        'test': task1_test_indices
    }
    
    task2_splits = {
        'train': train_indices_task2,
        'val': val_indices_task2,
        'test': task2_test_indices
    }
    
    # Save the splits to disk
    with open(os.path.join(processed_dir, 'train_val_split_task1.pkl'), 'wb') as f:
        pickle.dump({'train': train_indices_task1, 'val': val_indices_task1}, f)
    
    with open(os.path.join(processed_dir, 'test_split_task1.pkl'), 'wb') as f:
        pickle.dump({'test': task1_test_indices}, f)
    
    with open(os.path.join(processed_dir, 'train_val_split_task2.pkl'), 'wb') as f:
        pickle.dump({'train': train_indices_task2, 'val': val_indices_task2}, f)
    
    with open(os.path.join(processed_dir, 'test_split_task2.pkl'), 'wb') as f:
        pickle.dump({'test': task2_test_indices}, f)
    
    logger.info(f"Task 1 - Training: {len(train_indices_task1)}, Validation: {len(val_indices_task1)}, Test: {len(task1_test_indices)}")
    logger.info(f"Task 2 - Training: {len(train_indices_task2)}, Validation: {len(val_indices_task2)}, Test: {len(task2_test_indices)}")
    
    return task1_splits, task2_splits

def form_trajectories(task1_df, task2_df, task1_splits, task2_splits):
    """
    Step 4.3: Form trajectories from the training/validation data.
    
    Args:
        task1_df: DataFrame containing task1 data.
        task2_df: DataFrame containing task2 data.
        task1_splits: Dictionary with indices for task1 train/val/test splits.
        task2_splits: Dictionary with indices for task2 train/val/test splits.
        
    Returns:
        Dictionaries containing trajectories for training and validation for both tasks.
    """
    logger.info("Forming trajectories...")
    
    # Helper function to form trajectories for a specific dataframe and indices
    def _form_trajectories_for_split(df, indices):
        # Filter the dataframe to include only the specified indices
        df_subset = df.iloc[indices].copy()
        
        # Group by user ID and sort by day and time
        trajectories = {}
        logger.info(f"Processing {df_subset['uid'].nunique()} users for trajectories...")
        for uid, user_data in tqdm(df_subset.groupby('uid'), desc="Forming trajectories"):
            # Sort by day and time
            user_data = user_data.sort_values(by=['d', 't'])
            
            # Extract the trajectory as a list of (d, t, x, y) tuples
            traj = user_data[['d', 't', 'x', 'y']].values.tolist()
            trajectories[uid] = traj
        
        return trajectories
    
    # Form trajectories for task 1
    task1_train_trajectories = _form_trajectories_for_split(task1_df, task1_splits['train'])
    task1_val_trajectories = _form_trajectories_for_split(task1_df, task1_splits['val'])
    
    # Form trajectories for task 2
    task2_train_trajectories = _form_trajectories_for_split(task2_df, task2_splits['train'])
    task2_val_trajectories = _form_trajectories_for_split(task2_df, task2_splits['val'])
    
    task1_trajectories = {
        'train': task1_train_trajectories,
        'val': task1_val_trajectories
    }
    
    task2_trajectories = {
        'train': task2_train_trajectories,
        'val': task2_val_trajectories
    }
    
    logger.info(f"Formed trajectories for {len(task1_train_trajectories)} users in task 1 training set")
    logger.info(f"Formed trajectories for {len(task1_val_trajectories)} users in task 1 validation set")
    logger.info(f"Formed trajectories for {len(task2_train_trajectories)} users in task 2 training set")
    logger.info(f"Formed trajectories for {len(task2_val_trajectories)} users in task 2 validation set")
    
    return task1_trajectories, task2_trajectories

def create_node_features(task1_df, task2_df, cell_poi_df, poi_categories_df, config):
    """
    Step 4.4: Create node features (numerical) and mapping.
    """
    logger.info("Creating numerical node features and mapping...")
    processed_dir = config['data']['processed_dir']
    
    # Combine and deduplicate unique (x, y) coordinates
    all_xy = pd.concat([
        task1_df[['x', 'y']], 
        task2_df[['x', 'y']]
    ]).drop_duplicates().reset_index(drop=True)
    
    # Create node mapping: (x, y) -> node_id
    node_mapping = {(row['x'], row['y']): idx for idx, row in all_xy.iterrows()}
    
    # Save the node mapping
    node_mapping_path = os.path.join(processed_dir, 'node_mapping.json')
    with open(node_mapping_path, 'w') as f:
        serializable_mapping = {f"{x},{y}": idx for (x, y), idx in node_mapping.items()}
        json.dump(serializable_mapping, f)
    logger.info(f"Saved node mapping to {node_mapping_path}")
    
    # Merge POI data
    merged_data = pd.merge(all_xy, cell_poi_df, on=['x', 'y'], how='left')
    
    # Fill missing POI data and get POI column names
    poi_columns = [col for col in cell_poi_df.columns if col not in ['x', 'y']]
    merged_data[poi_columns] = merged_data[poi_columns].fillna(0)
    
    # Create numerical node features from POI counts
    node_features = torch.tensor(merged_data[poi_columns].values, dtype=torch.float)
    
    logger.info(f"Created mapping for {len(node_mapping)} unique grid cells")
    logger.info(f"Numerical node features shape: {node_features.shape}")
    
    return node_mapping, node_features

def generate_node_text_features(node_mapping, cell_poi_df, poi_categories_df, config, max_pois_in_desc=5):
    """
    (NEW) Generate text descriptions for each node based on POI data.

    Args:
        node_mapping (dict): Mapping from (x, y) tuple to node_id.
        cell_poi_df (pd.DataFrame): DataFrame with POI category code and count per cell.
        poi_categories_df (pd.DataFrame): DataFrame listing POI category names (no header).
        config (dict): Configuration dictionary.
        max_pois_in_desc (int): Max number of top POIs to include in the description.

    Returns:
        dict: Dictionary mapping node_id to its text description.
    """
    logger.info("Generating node text descriptions from POI data...")
    processed_dir = config['data']['processed_dir']
    node_text_descriptions = {}

    # --- Prepare POI category mapping from POI_datacategories.csv ---
    # Assuming the file has no header and category names are in the first column
    try:
        # Create a mapping from 1-based line number to category name
        category_mapping = {idx + 1: name.strip() for idx, name in enumerate(poi_categories_df.iloc[:, 0])}
        logger.info(f"Loaded {len(category_mapping)} POI category descriptions from line numbers.")
    except Exception as e:
        logger.error(f"Error creating category mapping from POI_datacategories.csv: {e}. Cannot generate meaningful descriptions.", exc_info=True)
        # Fallback to generic description if mapping fails completely
        for node_id in range(len(node_mapping)):
            node_text_descriptions[node_id] = f"Node {node_id} (POI category names unavailable)"
        # Save the incomplete descriptions and return
        desc_path = os.path.join(processed_dir, 'node_text_descriptions.json')
        node_text_descriptions_str_keys = {str(k): v for k, v in node_text_descriptions.items()}
        with open(desc_path, 'w') as f:
            json.dump(node_text_descriptions_str_keys, f, indent=2)
        logger.warning(f"Saved placeholder node text descriptions to {desc_path}")
        return node_text_descriptions

    # --- Prepare POI Data (cell_poi_df) --- 
    # Rename columns for clarity if necessary (assuming they are poi_catcode, poi_count)
    if 'poi_catcode' not in cell_poi_df.columns or 'poi_count' not in cell_poi_df.columns:
        logger.warning("Expected columns 'poi_catcode' and 'poi_count' not found in cell_POIcat.csv. Trying to use first columns after x, y.")
        # Fallback: Use columns at index 2 and 3
        try:
             cell_poi_df = cell_poi_df.rename(columns={cell_poi_df.columns[2]: 'poi_catcode', cell_poi_df.columns[3]: 'poi_count'})
        except IndexError:
             logger.error("Cannot identify POI category code and count columns in cell_POIcat.csv. Aborting text generation.")
             return {}

    # Invert node_mapping for easy lookup: node_id -> (x, y)
    id_to_coords = {v: k for k, v in node_mapping.items()}

    # Group POI data by cell coordinates for efficient lookup
    # Keep catcode and count
    cell_poi_grouped = cell_poi_df.groupby(['x', 'y'])[['poi_catcode', 'poi_count']].apply(lambda g: g.values.tolist()).to_dict()
    logger.info(f"Grouped POI data for {len(cell_poi_grouped)} cells.")

    # --- Generate Descriptions --- 
    for node_id in tqdm(range(len(node_mapping)), desc="Generating node text"):
        coords = id_to_coords.get(node_id)
        if coords is None:
            node_text_descriptions[node_id] = f"Node {node_id} (Unknown Coordinates)"
            continue

        x, y = coords
        description_prefix = f"Location ({x}, {y}). "
        poi_parts = []
        
        try:
            # Get list of [catcode, count] pairs for this cell
            poi_data = cell_poi_grouped.get(coords, [])
            
            if poi_data:
                # Sort by count descending
                poi_data.sort(key=lambda item: item[1], reverse=True)
                
                for catcode, count in poi_data[:max_pois_in_desc]:
                    # Use catcode (which should be 1-based index) to lookup name
                    cat_name = category_mapping.get(int(catcode), f"Category {catcode}") # Fallback if catcode not in mapping
                    
                    # Make description slightly more natural
                    plural = "s" if int(count) > 1 else ""
                    count_str = str(math.ceil(count)) 
                    if len(cat_name) > 30:
                        cat_name = cat_name[:27] + "..."
                    poi_parts.append(f"{count_str} {cat_name}{plural}")
                
                if poi_parts:
                     description = description_prefix + "Features: " + ", ".join(poi_parts) + "."
                else:
                     # This case should be rare if poi_data was not empty
                     description = description_prefix + "No significant POIs found after filtering."
            else:
                description = description_prefix + "No POIs recorded."
                
        except Exception as e:
             # Log the specific error and node for easier debugging
             logger.warning(f"Unexpected error processing POIs for node {node_id} at ({x},{y}): {type(e).__name__} - {e}", exc_info=True)
             description = description_prefix + "Error processing POIs."

        node_text_descriptions[node_id] = description

    # --- Save Descriptions --- 
    desc_path = os.path.join(processed_dir, 'node_text_descriptions.json')
    # Ensure node IDs are strings in the output JSON keys
    node_text_descriptions_str_keys = {str(k): v for k, v in node_text_descriptions.items()}
    with open(desc_path, 'w') as f:
        json.dump(node_text_descriptions_str_keys, f, indent=2)
    logger.info(f"Saved node text descriptions to {desc_path}")
    
    return node_text_descriptions

def construct_graph(node_mapping, node_features, config):
    """
    Step 4.5: Construct the graph.
    
    Args:
        node_mapping: Dictionary mapping (x, y) coordinates to node IDs.
        node_features: Tensor of node features.
        config: Dictionary containing configuration parameters.
        
    Returns:
        PyTorch Geometric Data object representing the graph.
    """
    logger.info("Constructing graph...")
    
    processed_dir = config['data']['processed_dir']
    max_edge_distance = config['gnn']['max_edge_distance']
    
    # Extract all grid cell coordinates
    grid_cells = list(node_mapping.keys())
    
    # Create edges based on spatial proximity
    edges = []
    logger.info(f"Calculating edges based on max distance {max_edge_distance}...")
    for i, (x1, y1) in enumerate(tqdm(grid_cells, desc="Constructing graph edges")):
        for j, (x2, y2) in enumerate(grid_cells):
            if i == j:
                continue  # Skip self-loops
            
            # Calculate Manhattan distance
            distance = abs(x1 - x2) + abs(y1 - y2)
            
            # Add edge if within the maximum distance
            if distance <= max_edge_distance:
                source_node = node_mapping[(x1, y1)]
                target_node = node_mapping[(x2, y2)]
                edges.append([source_node, target_node])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    graph_data = Data(x=node_features, edge_index=edge_index)
    
    # Save the graph data
    torch.save(graph_data, os.path.join(processed_dir, 'graph_data.pt'))
    
    logger.info(f"Constructed graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    
    return graph_data

def prepare_llm_sequences(task1_trajectories, task2_trajectories, node_mapping, config):
    """
    Step 4.6: Prepare sequences for LLM input.
    Now stores (node_id, day, time) tuples to support time-aware models.
    """
    logger.info("Preparing sequences for LLM input (including time information)...")
    
    llm_prepared_dir = config['data']['llm_prepared_dir']
    os.makedirs(llm_prepared_dir, exist_ok=True)
    
    # Helper function to convert trajectories to sequences of (node_id, d, t) tuples
    def _trajectories_to_node_sequences(trajectories):
        node_sequences = {}
        for uid, traj in tqdm(trajectories.items(), desc="Converting trajectories to node sequences"):
            # Convert each (d, t, x, y) point to a (node_id, d, t) tuple
            node_seq = [(node_mapping[(x, y)], d, t) for d, t, x, y in traj]
            node_sequences[uid] = node_seq
        return node_sequences
    
    # Convert trajectories to node sequences
    task1_train_sequences = _trajectories_to_node_sequences(task1_trajectories['train'])
    task1_val_sequences = _trajectories_to_node_sequences(task1_trajectories['val'])
    
    task2_train_sequences = _trajectories_to_node_sequences(task2_trajectories['train'])
    task2_val_sequences = _trajectories_to_node_sequences(task2_trajectories['val'])
    
    # Save the sequences
    torch.save(
        {'train': task1_train_sequences, 'val': task1_val_sequences},
        os.path.join(llm_prepared_dir, 'llm_sequences_train_val_task1.pt')
    )
    
    torch.save(
        {'train': task2_train_sequences, 'val': task2_val_sequences},
        os.path.join(llm_prepared_dir, 'llm_sequences_train_val_task2.pt')
    )
    
    logger.info("Saved LLM sequences (with time info) for both tasks")

def main():
    """Main preprocessing function to run all steps."""
    config = load_config()
    task1_df, task2_df, cell_poi_df, poi_categories_df = load_data(config)
    task1_splits, task2_splits = define_data_splits(task1_df, task2_df, config)
    task1_trajectories, task2_trajectories = form_trajectories(task1_df, task2_df, task1_splits, task2_splits)
    node_mapping, node_features = create_node_features(task1_df, task2_df, cell_poi_df, poi_categories_df, config)
    
    # --- (NEW) Scale Coordinates and Save Scalers ---
    logger.info("Scaling coordinates...")
    processed_dir = config['data']['processed_dir']

    # Combine training data from both tasks to fit scalers
    # It's important to fit scalers ONLY on training data to prevent data leakage
    task1_train_df = task1_df.iloc[task1_splits['train']]
    task2_train_df = task2_df.iloc[task2_splits['train']]
    combined_train_df = pd.concat([task1_train_df[['x', 'y']], task2_train_df[['x', 'y']]], ignore_index=True)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Fit scalers on the combined training data
    combined_train_df['x_scaled'] = x_scaler.fit_transform(combined_train_df[['x']])
    combined_train_df['y_scaled'] = y_scaler.fit_transform(combined_train_df[['y']])
    
    # Save the fitted scalers
    x_scaler_path = os.path.join(processed_dir, 'x_coordinate_scaler.pkl')
    y_scaler_path = os.path.join(processed_dir, 'y_coordinate_scaler.pkl')
    with open(x_scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)
    logger.info(f"Saved X-coordinate scaler to {x_scaler_path}")
    logger.info(f"Saved Y-coordinate scaler to {y_scaler_path}")

    # --- Apply scaling to the original full dataframes ---
    # We need to do this carefully to ensure node_mapping uses original coordinates
    # but trajectories and target coordinates for model use scaled versions.
    # This part will require modification of how trajectories are formed and used.
    # For now, we save the scalers. The next step will be to integrate their use.
    # It might be better to scale coordinates *within* the TrajectoryDataset or a similar
    # data loading utility specifically when preparing batches for the model,
    # rather than globally altering the dataframes if node_mapping relies on original coords.

    # For now, we'll just log that scalers are ready.
    # Actual transformation of task1_df, task2_df for training/eval targets
    # will need to be handled in the data loading/batching part of the training/eval pipeline.
    logger.info("Coordinate scalers are now ready. Subsequent steps need to use these to scale targets for training and inverse_scale predictions for evaluation.")
    # --- End (NEW) Scale Coordinates ---
    
    # Generate and save node text descriptions (NEW STEP)
    generate_node_text_features(node_mapping, cell_poi_df, poi_categories_df, config)
    
    graph_data = construct_graph(node_mapping, node_features, config)
    prepare_llm_sequences(task1_trajectories, task2_trajectories, node_mapping, config)
    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main() 