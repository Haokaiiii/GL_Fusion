"""
Script to generate prediction submissions for the HuMob Challenge.
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
import gzip

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gl_fusion_model import GLFusionModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate submission files for the HuMob Challenge.')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--task', type=int, default=1, choices=[1, 2],
                       help='Task ID (1 or 2)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to evaluate on')
    parser.add_argument('--output_dir', type=str, default='results/predictions',
                       help='Directory to save the submission files')
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


def load_test_data(config, task_id):
    """
    Load test data for submission generation.
    
    Args:
        config (dict): Configuration dictionary.
        task_id (int): Task ID (1 or 2).
        
    Returns:
        tuple: (task_df, test_indices, inv_node_mapping)
    """
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    
    # Load task data
    task_df = pd.read_csv(os.path.join(raw_dir, f'task{task_id}_dataset_kotae.csv'))
    
    # Load node mapping
    with open(os.path.join(processed_dir, 'node_mapping.json'), 'r') as f:
        # Convert string keys back to tuples: "x,y" -> (x, y)
        mapping_dict = json.load(f)
        node_mapping = {tuple(map(int, k.split(','))): v for k, v in mapping_dict.items()}
    
    # Create inverse node mapping: node_id -> (x, y)
    inv_node_mapping = {v: k for k, v in node_mapping.items()}
    
    # Load test indices
    with open(os.path.join(processed_dir, f'test_split_task{task_id}.pkl'), 'rb') as f:
        import pickle
        test_split = pickle.load(f)
        test_indices = test_split['test']
    
    return task_df, test_indices, inv_node_mapping


def generate_test_predictions(model, task_df, test_indices, inv_node_mapping, task_id, device):
    """
    Generate predictions for the test set.
    
    Args:
        model (GLFusionModel): Model to use for predictions.
        task_df (pd.DataFrame): Task data.
        test_indices (list): Test set indices.
        inv_node_mapping (dict): Inverse node mapping.
        task_id (int): Task ID (1 or 2).
        device (str): Device to use for predictions.
        
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    # Get the test data
    test_df = task_df.iloc[test_indices].copy()
    
    # For task 1, test users are 80000-99999 in days 60-74
    # For task 2, test users are 22500-24999 in days 60-74
    if task_id == 1:
        user_range = (80000, 99999)
    else:
        user_range = (22500, 24999)
    
    day_range = (60, 74)
    
    # Filter the test data to get only the users and days we need to predict
    test_df = test_df[
        (test_df['uid'] >= user_range[0]) & 
        (test_df['uid'] <= user_range[1]) & 
        (test_df['d'] >= day_range[0]) & 
        (test_df['d'] <= day_range[1])
    ]
    
    # Group by user
    logger.info(f"Generating predictions for {test_df['uid'].nunique()} users in task {task_id}")
    
    # Create a list to store prediction results
    predictions = []
    
    # Get unique users
    unique_users = test_df['uid'].unique()
    
    # For each user, predict their trajectory for days 60-74
    for uid in tqdm(unique_users, desc=f"Generating predictions for task {task_id}"):
        # Get the user's data from days 0-59
        history_df = task_df[
            (task_df['uid'] == uid) & 
            (task_df['d'] < day_range[0])
        ].sort_values(by=['d', 't'])
        
        # Skip if no history
        if len(history_df) == 0:
            logger.warning(f"No history data found for user {uid}")
            continue
        
        # Get the user's test data (for days 60-74)
        user_test_df = test_df[test_df['uid'] == uid].sort_values(by=['d', 't'])
        
        # Convert history trajectory to node IDs
        history_traj = []
        for _, row in history_df.iterrows():
            coord = (row['x'], row['y'])
            if coord in model.node_mapping:
                node_id = model.node_mapping[coord]
                history_traj.append((row['d'], row['t'], node_id))
        
        # Predict each time step for the user
        for i, (_, row) in enumerate(user_test_df.iterrows()):
            target_d = row['d']
            target_t = row['t']
            
            # Create input sequence based on history
            input_traj = history_traj.copy()
            
            # Add previously predicted positions to the trajectory
            for pred in predictions:
                if pred['uid'] == uid and (pred['d'] < target_d or (pred['d'] == target_d and pred['t'] < target_t)):
                    # Convert predicted x,y to node_id
                    pred_coord = (pred['pred_x'], pred['pred_y'])
                    if pred_coord in model.node_mapping:
                        pred_node_id = model.node_mapping[pred_coord]
                        input_traj.append((pred['d'], pred['t'], pred_node_id))
            
            # Sort by day and time
            input_traj.sort(key=lambda x: (x[0], x[1]))
            
            # If no input trajectory, skip
            if not input_traj:
                logger.warning(f"No input trajectory for user {uid} at day {target_d}, time {target_t}")
                continue
            
            # Extract just the node IDs
            node_seq = [x[2] for x in input_traj]
            
            # Prepare model input with a fixed sequence length
            seq_len = min(48, len(node_seq))  # Use at most the last 48 positions
            if len(node_seq) > seq_len:
                input_seq = node_seq[-seq_len:]
            else:
                # Pad with zeros if needed
                input_seq = [0] * (seq_len - len(node_seq)) + node_seq
            
            # Prepare input for the model
            input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = (input_tensor > 0).long()
            node_seq_mapping = input_tensor.clone()
            
            # Forward pass
            with torch.no_grad():
                pred_coords = model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    node_sequence_mapping=node_seq_mapping
                )
            
            # Convert prediction to numpy and round to integers
            pred_x, pred_y = map(int, np.round(pred_coords.cpu().numpy()[0]))
            
            # Store the prediction
            predictions.append({
                'uid': uid,
                'd': target_d,
                't': target_t,
                'pred_x': pred_x,
                'pred_y': pred_y
            })
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df


def save_submission_file(predictions_df, output_dir, team_name, task_id):
    """
    Save the submission file.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions.
        output_dir (str): Directory to save the submission file.
        team_name (str): Team name for the submission file.
        task_id (int): Task ID (1 or 2).
        
    Returns:
        str: Path to the saved submission file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the submission file
    submission_df = predictions_df[['uid', 'd', 't', 'pred_x', 'pred_y']].copy()
    submission_df.rename(columns={'pred_x': 'x', 'pred_y': 'y'}, inplace=True)
    
    # Sort by uid, d, t
    submission_df = submission_df.sort_values(by=['uid', 'd', 't']).reset_index(drop=True)
    
    # Save CSV file
    csv_path = os.path.join(output_dir, f"{team_name}_task{task_id}_humob.csv")
    submission_df.to_csv(csv_path, index=False)
    
    # Compress to gzip
    gz_path = f"{csv_path}.gz"
    with open(csv_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    # Remove the uncompressed file
    os.remove(csv_path)
    
    return gz_path


def validate_submission(submission_path, task_id):
    """
    Validate the submission file using the geobleu validator.
    
    Args:
        submission_path (str): Path to the submission file.
        task_id (int): Task ID (1 or 2).
        
    Returns:
        bool: True if validation is successful, False otherwise.
    """
    # First decompress the gzip file
    temp_csv_path = submission_path.replace('.gz', '')
    with gzip.open(submission_path, 'rb') as f_in:
        with open(temp_csv_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    try:
        # Import validator from geobleu
        import sys
        sys.path.append('geobleu-2023')
        from validator import main as validate_main
        
        # Run the validator
        result = validate_main([str(task_id), temp_csv_path])
        
        if result:
            logger.info("Submission validation successful")
            return True
        else:
            logger.error("Submission validation failed")
            return False
    except Exception as e:
        logger.error(f"Error validating submission: {e}")
        return False
    finally:
        # Remove the temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)


def main():
    """Main function to generate submission files."""
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
    
    log_file = os.path.join(log_dir, f'generate_submission_task{args.task}.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting submission generation for Task {args.task}")
    logger.info(f"Using device: {args.device}")
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, config, args.device)
    
    # Load test data
    logger.info("Loading test data")
    task_df, test_indices, inv_node_mapping = load_test_data(config, args.task)
    
    # Generate predictions
    logger.info("Generating predictions")
    predictions_df = generate_test_predictions(
        model, task_df, test_indices, inv_node_mapping, args.task, args.device
    )
    
    # Save submission file
    team_name = config['logging']['team_name']
    logger.info(f"Saving submission file for team: {team_name}")
    submission_path = save_submission_file(predictions_df, args.output_dir, team_name, args.task)
    logger.info(f"Submission saved to: {submission_path}")
    
    # Validate submission
    logger.info("Validating submission")
    is_valid = validate_submission(submission_path, args.task)
    
    if is_valid:
        logger.info("Submission is valid and ready for the HuMob Challenge")
    else:
        logger.error("Submission validation failed, please check the submission file")
    
    logger.info("Submission generation completed")


if __name__ == "__main__":
    main() 