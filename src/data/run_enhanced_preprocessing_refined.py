#!/usr/bin/env python3
"""
Enhanced preprocessing script using real POI categories for better LLM utilization.
This script uses the actual 86 POI categories from POI_datacategories.csv.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocess import (
    load_config, load_data, define_data_splits, form_trajectories,
    create_node_features, construct_graph
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_poi_categories(poi_categories_df):
    """Load the actual POI categories from POI_datacategories.csv."""
    category_mapping = {}
    for idx, category_name in enumerate(poi_categories_df.iloc[:, 0]):
        category_mapping[idx + 1] = category_name.strip()
    
    logger.info(f"Loaded {len(category_mapping)} real POI categories")
    return category_mapping

def create_detailed_poi_description(location_pois, category_mapping):
    """Create detailed POI description using real category names."""
    
    if location_pois.empty:
        return "Primarily residential or undeveloped area"
    
    # Sort by POI count
    location_pois_sorted = location_pois.sort_values('POI_count', ascending=False)
    
    poi_details = []
    food_count = 0
    shopping_count = 0
    service_count = 0
    entertainment_count = 0
    
    # Categorize POIs using real category keywords
    food_keywords = ['restaurant', 'cuisine', 'food', 'café', 'pizza', 'ramen', 'curry', 'bbq', 'bar', 'pub', 'diner', 'bakery', 'tea', 'wine', 'beer', 'fast food']
    shopping_keywords = ['store', 'shop', 'shopping', 'retail', 'grocery', 'electronics', 'clothes', 'convenience', 'drug store']
    service_keywords = ['bank', 'post office', 'hospital', 'pharmacy', 'hair salon', 'laundry', 'lawyer', 'accountant', 'real estate']
    entertainment_keywords = ['karaoke', 'casino', 'amusement', 'theme park', 'game arcade', 'swimming pool', 'disco', 'hotel', 'park']
    
    # Process top POIs
    for _, row in location_pois_sorted.head(8).iterrows():
        poi_category_id = int(row['POIcategory'])
        poi_count = int(row['POI_count'])
        
        if poi_category_id in category_mapping and poi_count > 0:
            category_name = category_mapping[poi_category_id].lower()
            
            # Categorize by type
            if any(keyword in category_name for keyword in food_keywords):
                food_count += poi_count
            elif any(keyword in category_name for keyword in shopping_keywords):
                shopping_count += poi_count
            elif any(keyword in category_name for keyword in service_keywords):
                service_count += poi_count
            elif any(keyword in category_name for keyword in entertainment_keywords):
                entertainment_count += poi_count
            
            # Add specific details for LLM understanding
            if poi_count > 1:
                poi_details.append(f"{poi_count} {category_name}s")
            else:
                poi_details.append(f"1 {category_name}")
    
    # Create rich description for LLM
    description_parts = []
    
    # Primary function classification
    total_count = food_count + shopping_count + service_count + entertainment_count
    if total_count > 0:
        categories = []
        if food_count / total_count > 0.4:
            categories.append("dining and food services")
        if shopping_count / total_count > 0.3:
            categories.append("retail and shopping")
        if service_count / total_count > 0.3:
            categories.append("professional services")
        if entertainment_count / total_count > 0.3:
            categories.append("entertainment and leisure")
        
        if categories:
            description_parts.append(f"Primary functions: {', '.join(categories)}")
    
    # Specific business details
    if poi_details:
        specific_desc = "Features: " + ", ".join(poi_details[:4])
        if len(poi_details) > 4:
            specific_desc += f" and {len(poi_details) - 4} other businesses"
        description_parts.append(specific_desc)
    
    return ". ".join(description_parts) if description_parts else "Mixed commercial area"

def generate_enhanced_node_descriptions(node_mapping, cell_poi_df, poi_categories_df, 
                                       task1_df, task2_df, config):
    """Generate enhanced node descriptions using real POI categories."""
    logger.info("Generating enhanced node descriptions with real POI categories...")
    
    # Load real POI categories
    category_mapping = load_real_poi_categories(poi_categories_df)
    
    enhanced_descriptions = {}
    id_to_coords = {v: k for k, v in node_mapping.items()}
    
    for node_id in range(len(node_mapping)):
        coords = id_to_coords.get(node_id)
        if coords is None:
            enhanced_descriptions[node_id] = f"Node {node_id} (unknown location)"
            continue
            
        x, y = coords
        
        # Get POI information for this location
        location_pois = cell_poi_df[(cell_poi_df['x'] == x) & (cell_poi_df['y'] == y)]
        
        # Build rich semantic description
        description_parts = [f"Location at grid coordinates ({x}, {y})"]
        
        # Detailed POI functionality using real categories
        poi_desc = create_detailed_poi_description(location_pois, category_mapping)
        description_parts.append(poi_desc)
        
        enhanced_descriptions[node_id] = ". ".join(description_parts) + "."
    
    logger.info(f"Generated enhanced descriptions for {len(enhanced_descriptions)} locations")
    return enhanced_descriptions, category_mapping

def create_enhanced_trajectory_sequences(trajectories, enhanced_descriptions, category_mapping, node_mapping):
    """Create enhanced trajectory sequences with rich narratives."""
    logger.info("Creating enhanced trajectory sequences with rich narratives...")
    
    enhanced_sequences = {}
    
    for split in ['train', 'val']:
        enhanced_sequences[split] = {}
        
        for uid, raw_trajectory in tqdm(trajectories[split].items(), desc=f"Processing {split}"):
            # Create narrative for LLM understanding
            narrative_parts = []
            
            # Convert raw trajectory (with x,y) to node-based trajectory
            node_trajectory = []
            for d, t, x, y in raw_trajectory:
                node_id = node_mapping.get((x, y))
                if node_id is not None:
                    node_trajectory.append((node_id, d, t))

            if not node_trajectory:
                continue

            # Get the day for the journey intro
            journey_day = node_trajectory[0][1]
            day_type = "weekend" if (journey_day - 1) % 7 >= 5 else "weekday"
            intro = f"User's journey on {day_type} day {journey_day}:"

            for i, (node_id, day, time) in enumerate(node_trajectory):
                location_desc = enhanced_descriptions.get(node_id, f"location {node_id}")
                
                # Extract key features for narrative
                if "Primary functions:" in location_desc:
                    features = location_desc.split("Primary functions:")[1].split(".")[0].strip()
                    features = f"an area with {features}"
                elif "Features:" in location_desc:
                    features = location_desc.split("Features:")[1].split(".")[0].strip()
                    features = f"an area featuring {features}"
                elif "residential" in location_desc.lower():
                    features = "a residential area"
                else:
                    features = "an unknown location"
                
                # Time context for LLM understanding
                hour = time % 24
                minute = (time * 2) % 60 # Assuming 30-min intervals
                time_str = f"{hour:02d}:{minute:02d}"

                if 5 <= hour < 9:
                    time_period = "early morning"
                elif 9 <= hour < 12:
                    time_period = "morning"
                elif 12 <= hour < 14:
                    time_period = "lunch time"
                elif 14 <= hour < 17:
                    time_period = "afternoon"
                elif 17 <= hour < 20:
                    time_period = "evening"
                elif 20 <= hour < 23:
                    time_period = "late evening"
                else:
                    time_period = "late night"
                
                if i == 0:
                    narrative_parts.append(
                        f"starts at {time_str} ({time_period}) at Node {node_id}, {features}"
                    )
                else:
                    narrative_parts.append(f"at {time_str} ({time_period}) at Node {node_id}, {features}")
            
            enhanced_sequences[split][uid] = {
                'tokens': node_trajectory,
                'narrative': intro + " " + "; ".join(narrative_parts) + ".",
                'node_descriptions': {node_id: enhanced_descriptions.get(node_id, "") for node_id, _, _ in node_trajectory}
            }
    
    return enhanced_sequences

def save_enhanced_results(enhanced_descriptions, enhanced_sequences, category_mapping, config, task_id):
    """Save all enhanced preprocessing results."""
    processed_dir = Path(config['data']['processed_dir'])
    llm_prepared_dir = Path(config['data']['llm_prepared_dir'])
    
    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    llm_prepared_dir.mkdir(parents=True, exist_ok=True)
    
    # Save enhanced descriptions
    desc_path = processed_dir / f'enhanced_node_descriptions_task{task_id}.json'
    with open(desc_path, 'w') as f:
        json.dump({str(k): v for k, v in enhanced_descriptions.items()}, f, indent=2)
    
    # Save enhanced sequences
    seq_path = llm_prepared_dir / f'enhanced_llm_sequences_task{task_id}.pt'
    torch.save(enhanced_sequences, seq_path)
    
    # Save category mapping for reference
    cat_path = processed_dir / 'poi_category_mapping.json'
    with open(cat_path, 'w') as f:
        json.dump({str(k): v for k, v in category_mapping.items()}, f, indent=2)
    
    logger.info(f"Saved enhanced preprocessing results for task {task_id}")
    logger.info(f"  Descriptions: {desc_path}")
    logger.info(f"  Sequences: {seq_path}")
    logger.info(f"  POI Categories: {cat_path}")

def run_enhanced_preprocessing(task_id=2):
    """Run the complete enhanced preprocessing pipeline."""
    
    logger.info("🚀 Starting Enhanced Preprocessing with Real POI Categories")
    logger.info("=" * 80)
    
    # Load configuration and data
    logger.info("Step 1: Loading configuration and raw data...")
    config = load_config()
    task1_df, task2_df, cell_poi_df, poi_categories_df = load_data(config)
    
    # Standard preprocessing steps
    logger.info("Step 2: Running standard preprocessing...")
    task1_splits, task2_splits = define_data_splits(task1_df, task2_df, config)
    task1_trajectories, task2_trajectories = form_trajectories(
        task1_df, task2_df, task1_splits, task2_splits
    )
    node_mapping, node_features = create_node_features(
        task1_df, task2_df, cell_poi_df, poi_categories_df, config
    )
    
    # Enhanced descriptions using real POI categories
    logger.info("Step 3: Generating enhanced descriptions with real POI categories...")
    enhanced_descriptions, category_mapping = generate_enhanced_node_descriptions(
        node_mapping, cell_poi_df, poi_categories_df, task1_df, task2_df, config
    )
    
    # Select appropriate trajectories for the task
    if task_id == 1:
        trajectories = task1_trajectories
    else:  # task_id == 2
        trajectories = task2_trajectories
    
    # Enhanced trajectory sequences
    logger.info("Step 4: Creating enhanced trajectory sequences...")
    enhanced_sequences = create_enhanced_trajectory_sequences(
        trajectories, enhanced_descriptions, category_mapping, node_mapping
    )
    
    # Construct graph (standard)
    logger.info("Step 5: Constructing spatial graph...")
    graph_data = construct_graph(node_mapping, node_features, config)
    
    # Save results
    logger.info("Step 6: Saving enhanced results...")
    save_enhanced_results(enhanced_descriptions, enhanced_sequences, category_mapping, config, task_id)
    
    logger.info("✅ Enhanced preprocessing completed successfully!")
    
    # Print sample results
    print_sample_results(enhanced_descriptions, enhanced_sequences)

def print_sample_results(descriptions, sequences):
    """Print sample results to show the enhancement quality."""
    print("\n📊 ENHANCED PREPROCESSING SAMPLE RESULTS")
    print("=" * 60)
    
    # Sample description
    sample_node = next(iter(descriptions.keys()))
    sample_desc = descriptions[sample_node]
    print(f"Sample Enhanced Description (Node {sample_node}):")
    print(f"  {sample_desc}")
    
    # Sample trajectory narrative
    sample_uid = next(iter(sequences['train'].keys()))
    sample_narrative = sequences['train'][sample_uid]['narrative']
    print(f"\nSample Trajectory Narrative (User {sample_uid}):")
    print(f"  {sample_narrative}")
    
    print(f"\nTotal enhanced descriptions: {len(descriptions)}")
    print(f"Training sequences: {len(sequences['train'])}")
    print(f"Validation sequences: {len(sequences['val'])}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced preprocessing with real POI categories')
    parser.add_argument('--task', type=int, default=2, choices=[1, 2], 
                       help='Task ID (1 or 2)')
    
    args = parser.parse_args()
    run_enhanced_preprocessing(args.task) 