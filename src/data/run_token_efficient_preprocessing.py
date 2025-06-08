#!/usr/bin/env python3
"""
Token-efficient enhanced preprocessing script using real POI categories.
This script generates compact, structured narratives for better LLM utilization.
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.append(str(Path(__file__).parent.parent))

from data.preprocess import (
    load_config, load_data, define_data_splits, form_trajectories,
    create_node_features, construct_graph
)
from data.enhanced_preprocess_refined import (
    load_real_poi_categories, create_detailed_poi_description,
    generate_enhanced_semantic_descriptions
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def abbreviate_time_period(hour):
    """Create abbreviated time period strings."""
    if 5 <= hour < 9: return "early_morn"
    if 9 <= hour < 12: return "morning"
    if 12 <= hour < 14: return "lunch"
    if 14 <= hour < 17: return "afternoon"
    if 17 <= hour < 20: return "evening"
    if 20 <= hour < 23: return "late_eve"
    return "late_night"

def extract_abbreviated_features(description):
    """Extract an abbreviated feature string from the description."""
    if "Primary functions:" in description:
        func_str = description.split("Primary functions:")[1].split(".")[0].strip()
        return func_str.replace(" and ", "&").replace(" services", "").replace(" ", "_")
    elif "residential" in description.lower():
        return "residential"
    return "unknown"

def create_token_efficient_sequences(trajectories, enhanced_descriptions, node_mapping):
    """Create token-efficient trajectory sequences."""
    logger.info("Creating token-efficient trajectory sequences...")
    
    efficient_sequences = {}
    
    for split in ['train', 'val']:
        efficient_sequences[split] = {}
        
        for uid, raw_trajectory in tqdm(trajectories[split].items(), desc=f"Processing {split}"):
            narrative_parts = []
            
            node_trajectory = []
            for d, t, x, y in raw_trajectory:
                node_id = node_mapping.get((x, y))
                if node_id is not None:
                    node_trajectory.append((node_id, d, t))

            if not node_trajectory:
                continue

            journey_day = node_trajectory[0][1]
            day_type = "WKD" if (journey_day - 1) % 7 < 5 else "WKE"
            intro = f"D{journey_day}({day_type}):"

            for i, (node_id, day, time) in enumerate(node_trajectory):
                location_desc = enhanced_descriptions.get(node_id, "")
                features = extract_abbreviated_features(location_desc)
                
                hour = time % 24
                minute = (time * 2) % 60
                time_str = f"{hour:02d}:{minute:02d}"
                time_period_abbr = abbreviate_time_period(hour)
                
                part = f"T{time_str}({time_period_abbr}) N{node_id}({features})"
                narrative_parts.append(part)
            
            efficient_sequences[split][uid] = {
                'tokens': node_trajectory,
                'narrative': intro + " -> ".join(narrative_parts)
            }
            
    return efficient_sequences

def run_token_efficient_preprocessing(task_id=2):
    """Run the token-efficient enhanced preprocessing pipeline."""
    
    logger.info("🚀 Starting Token-Efficient Enhanced Preprocessing")
    config = load_config()
    task1_df, task2_df, cell_poi_df, poi_categories_df = load_data(config)
    task1_splits, task2_splits = define_data_splits(task1_df, task2_df, config)
    
    trajectories = task2_trajectories if task_id == 2 else task1_trajectories
    
    node_mapping, node_features = create_node_features(task1_df, task2_df, cell_poi_df, poi_categories_df, config)
    
    enhanced_descriptions, category_mapping = generate_enhanced_semantic_descriptions(
        node_mapping, cell_poi_df, poi_categories_df, task1_df, task2_df, config
    )
    
    efficient_sequences = create_token_efficient_sequences(
        trajectories, enhanced_descriptions, node_mapping
    )
    
    # Save results
    processed_dir = Path(config['data']['processed_dir'])
    llm_prepared_dir = Path(config['data']['llm_prepared_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    llm_prepared_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sequences
    seq_path = llm_prepared_dir / f'token_efficient_sequences_task{task_id}.pt'
    torch.save(efficient_sequences, seq_path)
    logger.info(f"Saved token-efficient sequences to {seq_path}")
    
    # Save descriptions for reference
    desc_path = processed_dir / f'enhanced_node_descriptions_task{task_id}.json'
    with open(desc_path, 'w') as f:
        json.dump({str(k): v for k, v in enhanced_descriptions.items()}, f, indent=2)

    logger.info("✅ Token-efficient preprocessing completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run token-efficient enhanced preprocessing')
    parser.add_argument('--task', type=int, default=2, choices=[1, 2], help='Task ID (1 or 2)')
    
    args = parser.parse_args()
    run_token_efficient_preprocessing(args.task) 