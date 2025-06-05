"""
Evaluation script for the GL-Fusion model.
"""

import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import sys
from tqdm import tqdm
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add geobleu to path if not already installed in environment
# This assumes geobleu-2023 directory is at the project root
geobleu_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'geobleu-2023')
if geobleu_path not in sys.path:
    sys.path.insert(0, geobleu_path)
import geobleu # For GeoBLEU and DTW

from model.gl_fusion_model import GLFusionModel
from training.utils import calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the GL-Fusion model.')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--task', type=int, default=1, choices=[1, 2],
                       help='Task ID (1 or 2)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to evaluate on')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the evaluation results')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config, device):
    """
    Load the model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        config (dict): Configuration dictionary.
        device (str): Device to load the model onto.
        
    Returns:
        GLFusionModel: The loaded model.
    """
    # Initialize model
    model = GLFusionModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def load_test_data(config, task_id, split='val'):
    """
    Load test data for evaluation.
    
    Args:
        config (dict): Configuration dictionary.
        task_id (int): Task ID (1 or 2).
        split (str): Dataset split ('val' or 'test').
        
    Returns:
        tuple: (test_sequences, test_df, node_mapping)
    """
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    llm_prepared_dir = config['data']['llm_prepared_dir']
    
    # Load task data
    task_df = pd.read_csv(os.path.join(raw_dir, f'task{task_id}_dataset_kotae.csv'))
    
    # Load node mapping
    with open(os.path.join(processed_dir, 'node_mapping.json'), 'r') as f:
        # Convert string keys back to tuples: "x,y" -> (x, y)
        mapping_dict = json.load(f)
        node_mapping = {tuple(map(int, k.split(','))): v for k, v in mapping_dict.items()}
    
    # Load split indices
    if split == 'val':
        with open(os.path.join(processed_dir, f'train_val_split_task{task_id}.pkl'), 'rb') as f:
            import pickle
            split_data = pickle.load(f)
            indices = split_data['val']
    else:  # split == 'test'
        with open(os.path.join(processed_dir, f'test_split_task{task_id}.pkl'), 'rb') as f:
            import pickle
            split_data = pickle.load(f)
            indices = split_data['test']
    
    # Load node sequences if evaluating on validation set
    test_sequences = None
    if split == 'val':
        node_sequences = torch.load(os.path.join(llm_prepared_dir, f'llm_sequences_train_val_task{task_id}.pt'))
        test_sequences = node_sequences['val']
    
    return test_sequences, task_df, indices, node_mapping


def evaluate_validation_set(model, test_sequences, task_df, indices, node_mapping, device):
    """
    Evaluate the model on the validation set.
    
    Args:
        model (GLFusionModel): Model to evaluate.
        test_sequences (dict): Test node sequences.
        task_df (pd.DataFrame): Task data.
        indices (list): Indices of the validation set.
        node_mapping (dict): Node mapping dictionary.
        device (str): Device to evaluate on.
        
    Returns:
        tuple: (metrics, predictions_df)
    """
    # For validation, we already have the ground truth
    all_predictions = []
    all_targets = []
    
    # Track user, day, time, predicted and actual coordinates
    prediction_records = []
    
    # Lists to store per-user GeoBLEU and DTW scores
    user_geobleu_scores = []
    user_dtw_scores = []
    
    # Get unique users in the validation set
    val_users = set(task_df.iloc[indices]['uid'].values)
    
    for uid in tqdm(val_users, desc="Evaluating validation set"):
        if uid not in test_sequences:
            continue
        
        # Get the user's sequence
        node_seq = test_sequences[uid]
        
        # Get the user's data from the validation set
        user_df = task_df[(task_df['uid'] == uid) & task_df.index.isin(indices)].sort_values(by=['d', 't'])
        
        # Skip if not enough data
        if len(user_df) < 2:
            continue
        
        # For each day/time point, predict the next location
        # Collect predicted and target trajectories for the current user
        user_predicted_trajectory = []
        user_target_trajectory = []

        for i in range(len(user_df) - 1):
            # Get the current location's info
            current_row = user_df.iloc[i]
            current_d = current_row['d']
            current_t = current_row['t']
            
            # Get the target location's info
            target_row = user_df.iloc[i + 1]
            target_d = target_row['d']
            target_t = target_row['t']
            target_coords = np.array([target_row['x'], target_row['y']])
            
            # Extract the sequence up to the current point
            # This is simplified; in practice, we'd need to match the exact sequence
            # based on day and time, but for demonstration we'll just use a fixed length
            seq_len = 24  # Use the last 24 locations
            if i >= seq_len:
                input_seq = node_seq[i - seq_len + 1:i + 1]
            else:
                # Pad with zeros if needed
                input_seq = [0] * (seq_len - i - 1) + node_seq[:i + 1]
            
            # Prepare input for the model
            input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = (input_tensor > 0).long()
            node_seq_mapping = input_tensor.clone()
            
            # Forward pass
            with torch.no_grad():
                predictions = model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    node_sequence_mapping=node_seq_mapping
                )
            
            # Convert prediction to numpy
            pred_coords = predictions.cpu().numpy()[0]
            
            # Store predictions and targets
            all_predictions.append(pred_coords)
            all_targets.append(target_coords)
            
            # Append to user-specific trajectories for GeoBLEU/DTW
            # Format for geobleu: (d, t, x, y)
            user_predicted_trajectory.append((target_d, target_t, pred_coords[0], pred_coords[1]))
            user_target_trajectory.append((target_d, target_t, target_coords[0], target_coords[1]))
            
            # Store the prediction record
            prediction_records.append({
                'uid': uid,
                'current_d': current_d,
                'current_t': current_t,
                'target_d': target_d,
                'target_t': target_t,
                'pred_x': pred_coords[0],
                'pred_y': pred_coords[1],
                'target_x': target_coords[0],
                'target_y': target_coords[1]
            })
        
        # Calculate GeoBLEU and DTW for the current user if trajectories are not empty
        if user_predicted_trajectory and user_target_trajectory:
            try:
                # Ensure trajectories are long enough for default n-grams (min length 3 for n=3)
                if len(user_predicted_trajectory) >= 1 and len(user_target_trajectory) >=1: # geobleu internal check handles n-gram length
                    gb_score = geobleu.calc_geobleu_single(user_predicted_trajectory, user_target_trajectory)
                    user_geobleu_scores.append(gb_score)
                else:
                    logger.debug(f"Skipping GeoBLEU for UID {uid} due to short trajectory (pred: {len(user_predicted_trajectory)}, target: {len(user_target_trajectory)})")

                dtw_score = geobleu.calc_dtw_single(user_predicted_trajectory, user_target_trajectory)
                user_dtw_scores.append(dtw_score)
            except Exception as e:
                logger.warning(f"Could not calculate GeoBLEU/DTW for UID {uid}. Trajectories: Pred={len(user_predicted_trajectory)}, Target={len(user_target_trajectory)}. Error: {e}")

    # Calculate metrics
    all_predictions_np = np.array(all_predictions)
    all_targets_np = np.array(all_targets)
    if all_predictions_np.size > 0 and all_targets_np.size > 0:
        metrics = calculate_metrics(torch.tensor(all_predictions_np), torch.tensor(all_targets_np))
    else:
        metrics = {'mse': float('nan'), 'euclidean_dist': float('nan')}

    # Add average GeoBLEU and DTW to metrics
    if user_geobleu_scores:
        metrics['geo_bleu'] = np.mean(user_geobleu_scores)
    else:
        metrics['geo_bleu'] = float('nan') # Or 0.0, depending on desired behavior for no valid scores
        
    if user_dtw_scores:
        metrics['dtw'] = np.mean(user_dtw_scores)
    else:
        metrics['dtw'] = float('nan') # Or appropriate value

    # Create predictions dataframe
    predictions_df = pd.DataFrame(prediction_records)
    
    return metrics, predictions_df


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['task_id'] = args.task
    config['device'] = args.device
    
    # Set up logging
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'evaluation_log_task{args.task}_{args.split}.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting evaluation for Task {args.task} on {args.split} set")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    
    # Load model
    model = load_model(args.checkpoint, config, args.device)
    
    # Load test data
    test_sequences, task_df, indices, node_mapping = load_test_data(config, args.task, args.split)
    
    # Evaluate on appropriate split
    if args.split == 'val':
        metrics, predictions_df = evaluate_validation_set(
            model, test_sequences, task_df, indices, node_mapping, args.device
        )
        
        # Log metrics
        logger.info(f"Validation Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        # Save results if output path is provided
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # Save metrics
            metrics_path = f"{args.output}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save predictions
            predictions_path = f"{args.output}_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            
            logger.info(f"Saved metrics to {metrics_path}")
            logger.info(f"Saved predictions to {predictions_path}")
    else:
        # For test set, we would need to generate submission files
        # This is currently not implemented in this basic version
        logger.warning("Test set evaluation not implemented in this script. Use generate_submission.py instead.")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main() 