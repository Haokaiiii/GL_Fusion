"""
Enhanced evaluation script for the GL-Fusion model with coordinate scaling support.
This script properly handles tokenizer size mismatches and coordinate inverse-transformation.
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
import pickle
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings

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
    parser = argparse.ArgumentParser(description='Evaluate the GL-Fusion model with coordinate scaling.')
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


def load_coordinate_scalers(config):
    """Load the coordinate scalers for inverse transformation."""
    processed_dir = Path(config['data']['processed_dir'])
    x_scaler_path = processed_dir / 'x_coordinate_scaler.pkl'
    y_scaler_path = processed_dir / 'y_coordinate_scaler.pkl'
    
    try:
        logger.info(f"Loading coordinate scalers: X from {x_scaler_path}, Y from {y_scaler_path}")
        with open(x_scaler_path, 'rb') as f:
            x_scaler = pickle.load(f)
        with open(y_scaler_path, 'rb') as f:
            y_scaler = pickle.load(f)
        logger.info("Coordinate scalers loaded successfully.")
        return x_scaler, y_scaler
    except FileNotFoundError:
        logger.error(f"Coordinate scaler files not found in {processed_dir}. Cannot inverse-transform predictions.")
        logger.error("Make sure preprocessing has been run with the updated preprocess.py that saves scalers.")
        raise FileNotFoundError("Required coordinate scaler files not found.")
    except Exception as e:
        logger.error(f"Error loading coordinate scalers: {e}")
        raise


def load_model_with_scaling_support(checkpoint_path, config, device):
    """
    Load the model from checkpoint with support for tokenizer size mismatch.
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = GLFusionModel(config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path} with weights_only=False")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Handle tokenizer size mismatch
    current_vocab_size = model.llm.model.get_input_embeddings().weight.shape[0]
    checkpoint_vocab_size = checkpoint['model_state_dict']['llm.model.base_model.model.model.embed_tokens.weight'].shape[0]
    
    logger.info(f"Current model vocab size: {current_vocab_size}")
    logger.info(f"Checkpoint vocab size: {checkpoint_vocab_size}")
    
    if current_vocab_size != checkpoint_vocab_size:
        logger.info(f"Tokenizer size mismatch detected. Resizing model to match checkpoint...")
        model.llm.model.resize_token_embeddings(checkpoint_vocab_size)
        logger.info(f"Model resized to {checkpoint_vocab_size} tokens")
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load with strict=True, trying strict=False: {e}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Checkpoint loaded with strict=False")
    
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


def inverse_transform_predictions(predictions, x_scaler, y_scaler):
    """Inverse transform scaled predictions back to original coordinate space."""
    pred_x_scaled = predictions[0].reshape(-1, 1)
    pred_y_scaled = predictions[1].reshape(-1, 1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
        pred_x_original = x_scaler.inverse_transform(pred_x_scaled)
        pred_y_original = y_scaler.inverse_transform(pred_y_scaled)
    
    return np.array([pred_x_original[0,0], pred_y_original[0,0]])


def evaluate_validation_set(model, test_sequences, task_df, indices, node_mapping, device, x_scaler, y_scaler):
    """
    Evaluate the model on the validation set with proper coordinate scaling.
    """
    all_predictions = []
    all_targets = []
    prediction_records = []
    user_geobleu_scores = []
    user_dtw_scores = []
    
    # Get unique users in the validation set
    all_val_users = set(task_df.iloc[indices]['uid'].values)

    # Apply fast evaluation limits from environment variables for faster dev runs
    max_val_users = int(os.environ.get('MAX_VAL_USERS', '0'))
    if max_val_users > 0 and len(all_val_users) > max_val_users:
        import random
        random.seed(42)  # for reproducibility
        val_users = set(random.sample(list(all_val_users), max_val_users))
        logger.info(f'Fast evaluation: Limited to {max_val_users} users out of {len(all_val_users)} total')
    else:
        val_users = all_val_users
        logger.info(f"Evaluating on all {len(val_users)} users in the validation set.")

    for uid in tqdm(val_users, desc='Evaluating validation set'):
        if uid not in test_sequences:
            continue
        
        # Get the user's sequence
        node_seq = test_sequences[uid]
        
        # Get the user's data from the validation set
        user_df = task_df[(task_df['uid'] == uid) & task_df.index.isin(indices)].sort_values(by=['d', 't'])
        
        # Skip if not enough data
        if len(user_df) < 2:
            continue
        
        # Collect predicted and target trajectories for the current user
        user_predicted_trajectory = []
        user_target_trajectory = []

        for i in range(len(user_df) - 1):
            # Get the current and target location info
            current_row = user_df.iloc[i]
            current_d = current_row['d']
            current_t = current_row['t']
            
            target_row = user_df.iloc[i + 1]
            target_d = target_row['d']
            target_t = target_row['t']
            target_coords = np.array([target_row['x'], target_row['y']])
            
            # Extract the sequence up to the current point
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
                predictions_dict = model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    node_sequence_mapping=node_seq_mapping
                )
            
            # Extract predictions from dictionary (handle GLFusionModel output format)
            if isinstance(predictions_dict, dict) and 'predictions' in predictions_dict:
                predictions_tensor = predictions_dict['predictions']
            else:
                predictions_tensor = predictions_dict
            
            # Convert prediction to numpy and inverse-transform
            pred_coords_scaled = predictions_tensor.cpu().numpy()[0]
            pred_coords = inverse_transform_predictions(pred_coords_scaled, x_scaler, y_scaler)
            
            # Store predictions and targets
            all_predictions.append(pred_coords)
            all_targets.append(target_coords)
            
            # Append to user-specific trajectories for GeoBLEU/DTW
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
        
        # Calculate GeoBLEU and DTW for the current user
        if user_predicted_trajectory and user_target_trajectory:
            try:
                if len(user_predicted_trajectory) >= 1 and len(user_target_trajectory) >= 1:
                    gb_score = geobleu.calc_geobleu_single(user_predicted_trajectory, user_target_trajectory)
                    user_geobleu_scores.append(gb_score)

                dtw_score = geobleu.calc_dtw_single(user_predicted_trajectory, user_target_trajectory)
                user_dtw_scores.append(dtw_score)
            except Exception as e:
                logger.warning(f'Could not calculate GeoBLEU/DTW for UID {uid}. Error: {e}')

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
        metrics['geo_bleu'] = float('nan')
        
    if user_dtw_scores:
        metrics['dtw'] = np.mean(user_dtw_scores)
    else:
        metrics['dtw'] = float('nan')

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
    
    log_file = os.path.join(log_dir, f'evaluation_with_scaling_log_task{args.task}_{args.split}.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting evaluation for Task {args.task} on {args.split} set")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    
    # Load coordinate scalers
    x_scaler, y_scaler = load_coordinate_scalers(config)
    
    # Load model
    model = load_model_with_scaling_support(args.checkpoint, config, args.device)
    
    # Load test data
    test_sequences, task_df, indices, node_mapping = load_test_data(config, args.task, args.split)
    
    # Evaluate on appropriate split
    if args.split == 'val':
        metrics, predictions_df = evaluate_validation_set(
            model, test_sequences, task_df, indices, node_mapping, args.device, x_scaler, y_scaler
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
        logger.warning("Test set evaluation not implemented in this version.")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main() 