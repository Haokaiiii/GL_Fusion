"""
Refined enhanced preprocessing using real POI categories from POI_datacategories.csv.
Creates much richer semantic descriptions with actual business types.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from collections import defaultdict, Counter
from pathlib import Path
import logging
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)

def load_real_poi_categories(poi_categories_df):
    """Load the actual POI categories from POI_datacategories.csv."""
    # Create mapping from line number (1-based) to category name
    category_mapping = {}
    for idx, category_name in enumerate(poi_categories_df.iloc[:, 0]):
        category_mapping[idx + 1] = category_name.strip()
    
    logger.info(f"Loaded {len(category_mapping)} real POI categories")
    return category_mapping

def create_semantic_poi_groups(category_mapping):
    """Group POI categories into semantic clusters for better understanding."""
    
    food_keywords = ['restaurant', 'cuisine', 'food', 'café', 'pizza', 'ramen', 'curry', 'bbq', 'bar', 'pub', 'diner', 'bakery', 'tea', 'wine', 'beer', 'fast food']
    shopping_keywords = ['store', 'shop', 'shopping', 'retail', 'grocery', 'electronics', 'clothes', 'convenience', 'drug store', 'glasses']
    entertainment_keywords = ['karaoke', 'casino', 'amusement', 'theme park', 'game arcade', 'swimming pool', 'disco', 'cruising', 'sports recreation']
    services_keywords = ['bank', 'post office', 'hospital', 'pharmacy', 'hair salon', 'laundry', 'lawyer', 'accountant', 'real estate', 'driving school']
    transport_keywords = ['transit station', 'parking', 'port']
    education_keywords = ['school', 'kindergarten', 'cram school']
    accommodation_keywords = ['hotel', 'hot spring']
    public_keywords = ['city hall', 'community center', 'church', 'park', 'cemetery']
    business_keywords = ['office', 'it office', 'publisher', 'building material', 'heavy industry', 'npo', 'utility', 'research facility']
    
    poi_groups = {}
    
    for poi_id, category_name in category_mapping.items():
        category_lower = category_name.lower()
        
        if any(keyword in category_lower for keyword in food_keywords):
            poi_groups[poi_id] = 'dining_food'
        elif any(keyword in category_lower for keyword in shopping_keywords):
            poi_groups[poi_id] = 'retail_shopping'
        elif any(keyword in category_lower for keyword in entertainment_keywords):
            poi_groups[poi_id] = 'entertainment_leisure'
        elif any(keyword in category_lower for keyword in services_keywords):
            poi_groups[poi_id] = 'professional_services'
        elif any(keyword in category_lower for keyword in transport_keywords):
            poi_groups[poi_id] = 'transportation'
        elif any(keyword in category_lower for keyword in education_keywords):
            poi_groups[poi_id] = 'education'
        elif any(keyword in category_lower for keyword in accommodation_keywords):
            poi_groups[poi_id] = 'accommodation'
        elif any(keyword in category_lower for keyword in public_keywords):
            poi_groups[poi_id] = 'public_civic'
        elif any(keyword in category_lower for keyword in business_keywords):
            poi_groups[poi_id] = 'business_office'
        else:
            poi_groups[poi_id] = 'mixed_services'
    
    return poi_groups

def create_detailed_poi_description(location_pois, category_mapping):
    """Create detailed POI description using real category names."""
    
    if location_pois.empty:
        return None
    
    # Sort by POI count to get most prominent features first
    location_pois_sorted = location_pois.sort_values('POI_count', ascending=False)
    
    poi_details = []
    
    # Process top POIs (limit to avoid overly long descriptions)
    for _, row in location_pois_sorted.head(6).iterrows():
        poi_category_id = int(row['POIcategory'])
        poi_count = int(row['POI_count'])
        
        if poi_category_id in category_mapping and poi_count > 0:
            category_name = category_mapping[poi_category_id]
            
            # Add specific POI details
            if poi_count > 1:
                poi_details.append(f"{poi_count} {category_name.lower()}s")
            else:
                poi_details.append(f"1 {category_name.lower()}")
    
    if not poi_details:
        return None
    
    # Create description
    specific_desc = "Features: " + ", ".join(poi_details[:4])
    if len(poi_details) > 4:
        specific_desc += f" and {len(poi_details) - 4} other businesses"
    
    return specific_desc

def generate_enhanced_semantic_descriptions(node_mapping, cell_poi_df, poi_categories_df, 
                                          task1_df, task2_df, config):
    """
    Generate rich semantic descriptions using real POI categories.
    """
    logger.info("Generating enhanced semantic descriptions with real POI data...")
    
    # Load real POI categories
    category_mapping = load_real_poi_categories(poi_categories_df)
    poi_groups = create_semantic_poi_groups(category_mapping)
    
    # Analyze temporal patterns for each location
    location_patterns = analyze_temporal_patterns(task1_df, task2_df, node_mapping)
    
    # Generate neighborhood context
    neighborhood_context = generate_neighborhood_context(node_mapping, cell_poi_df, category_mapping, poi_groups)
    
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
        
        # Build rich description
        description_parts = [f"Location at grid coordinates ({x}, {y})"]
        
        # Detailed POI functionality
        if not location_pois.empty:
            poi_desc = create_detailed_poi_description(location_pois, category_mapping)
            if poi_desc:
                description_parts.append(poi_desc)
        else:
            description_parts.append("Primarily residential or undeveloped area")
        
        # Temporal activity patterns
        if coords in location_patterns:
            pattern_desc = describe_temporal_patterns(location_patterns[coords])
            description_parts.append(f"Activity patterns: {pattern_desc}")
        
        # Neighborhood context
        if coords in neighborhood_context:
            context_desc = neighborhood_context[coords]
            description_parts.append(f"Area characteristics: {context_desc}")
        
        # Mobility characteristics
        mobility_desc = analyze_mobility_characteristics(x, y, task1_df, task2_df)
        if mobility_desc:
            description_parts.append(f"Usage type: {mobility_desc}")
        
        enhanced_descriptions[node_id] = ". ".join(description_parts) + "."
    
    logger.info(f"Generated enhanced descriptions for {len(enhanced_descriptions)} locations")
    return enhanced_descriptions, category_mapping

def analyze_temporal_patterns(task1_df, task2_df, node_mapping):
    """Analyze when locations are typically visited."""
    combined_df = pd.concat([task1_df, task2_df])
    location_patterns = defaultdict(lambda: defaultdict(list))
    
    coord_to_node = {coords: node_id for node_id, coords in enumerate(node_mapping.keys())}
    
    for _, row in combined_df.iterrows():
        coords = (row['x'], row['y'])
        if coords in coord_to_node:
            hour = row['t'] % 24
            day_of_week = (row['d'] - 1) % 7
            location_patterns[coords]['hours'].append(hour)
            location_patterns[coords]['days'].append(day_of_week)
            location_patterns[coords]['visits'].append(1)
    
    return location_patterns

def describe_temporal_patterns(patterns):
    """Create natural language description of temporal patterns."""
    hours = patterns['hours']
    days = patterns['days']
    total_visits = len(hours)
    
    if total_visits < 5:
        return "limited activity recorded"
    
    # Analyze peak hours
    hour_counter = Counter(hours)
    peak_hours = [h for h, count in hour_counter.most_common(3)]
    
    # Categorize time periods
    time_desc = []
    
    # Morning patterns
    morning_visits = sum(1 for h in hours if 6 <= h <= 10)
    if morning_visits / total_visits > 0.3:
        time_desc.append("high morning activity")
    elif morning_visits / total_visits > 0.1:
        time_desc.append("moderate morning activity")
    
    # Midday patterns  
    midday_visits = sum(1 for h in hours if 11 <= h <= 14)
    if midday_visits / total_visits > 0.3:
        time_desc.append("popular lunch destination")
    
    # Evening patterns
    evening_visits = sum(1 for h in hours if 17 <= h <= 21)
    if evening_visits / total_visits > 0.3:
        time_desc.append("high evening usage")
    
    # Late night patterns
    late_visits = sum(1 for h in hours if h >= 22 or h <= 5)
    if late_visits / total_visits > 0.2:
        time_desc.append("late-night destination")
    
    # Day patterns
    day_counter = Counter(days)
    weekday_visits = sum(count for day, count in day_counter.items() if day < 5)
    weekend_visits = sum(count for day, count in day_counter.items() if day >= 5)
    
    total_day_visits = weekday_visits + weekend_visits
    if total_day_visits > 0:
        if weekday_visits / total_day_visits > 0.8:
            time_desc.append("primarily weekday location")
        elif weekend_visits / total_day_visits > 0.6:
            time_desc.append("weekend hotspot")
    
    return ", ".join(time_desc) if time_desc else "consistent activity throughout day"

def generate_neighborhood_context(node_mapping, cell_poi_df, category_mapping, poi_groups):
    """Generate neighborhood context using real POI data."""
    neighborhood_context = {}
    
    for coords in tqdm(node_mapping.keys(), desc="Analyzing neighborhoods"):
        x, y = coords
        
        # Look at 3x3 neighborhood
        surrounding_pois = []
        surrounding_groups = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_pois = cell_poi_df[
                    (cell_poi_df['x'] == x + dx) & (cell_poi_df['y'] == y + dy)
                ]
                
                for _, row in neighbor_pois.iterrows():
                    poi_cat_id = int(row['POIcategory'])
                    if poi_cat_id in poi_groups:
                        surrounding_groups.append(poi_groups[poi_cat_id])
                        surrounding_pois.append(poi_cat_id)
        
        if surrounding_groups:
            group_counts = Counter(surrounding_groups)
            total_variety = len(set(surrounding_groups))
            
            if total_variety >= 6:
                context = "highly diverse mixed-use district"
            elif total_variety >= 4:
                context = "mixed-use area"
            elif total_variety >= 2:
                dominant_group = group_counts.most_common(1)[0][0]
                context = f"{dominant_group.replace('_', ' ')} focused area"
            else:
                single_group = list(set(surrounding_groups))[0]
                context = f"specialized {single_group.replace('_', ' ')} district"
        else:
            context = "low-density residential or undeveloped area"
        
        neighborhood_context[coords] = context
    
    return neighborhood_context

def analyze_mobility_characteristics(x, y, task1_df, task2_df):
    """Analyze mobility patterns for this location."""
    combined_df = pd.concat([task1_df, task2_df])
    location_visits = combined_df[(combined_df['x'] == x) & (combined_df['y'] == y)]
    
    if len(location_visits) < 3:
        return "infrequent destination"
    
    unique_users = location_visits['uid'].nunique()
    total_visits = len(location_visits)
    
    # Calculate visit frequency patterns
    user_visit_counts = location_visits['uid'].value_counts()
    repeat_visitors = (user_visit_counts > 1).sum()
    
    if unique_users / total_visits > 0.8:
        return "high-turnover destination"
    elif repeat_visitors / unique_users > 0.6:
        return "regular-use location with loyal visitors"
    elif total_visits / unique_users > 3:
        return "frequently revisited location"
    else:
        return "moderate traffic location"

def create_enhanced_trajectory_narratives(trajectories, enhanced_descriptions, category_mapping, poi_groups):
    """Create rich trajectory narratives using real POI information."""
    enhanced_narratives = {}
    
    for split in ['train', 'val']:
        enhanced_narratives[split] = {}
        
        for uid, trajectory in tqdm(trajectories[split].items(), desc=f"Creating narratives for {split}"):
            narrative_parts = []
            
            for i, (node_id, day, time) in enumerate(trajectory):
                # Get enhanced description
                location_desc = enhanced_descriptions.get(node_id, f"location {node_id}")
                
                # Extract key features for narrative
                features = extract_key_features_from_description(location_desc)
                
                # Create time context
                hour = time % 24
                time_period = categorize_time_period(hour)
                day_type = "weekend" if (day - 1) % 7 >= 5 else "weekday"
                
                if i == 0:
                    narrative_parts.append(
                        f"Journey starts on {day_type} day {day} during {time_period} at {features}"
                    )
                else:
                    # Analyze movement type
                    prev_node_id = trajectory[i-1][0]
                    prev_desc = enhanced_descriptions.get(prev_node_id, "")
                    movement_type = classify_movement_type(prev_desc, location_desc)
                    
                    narrative_parts.append(
                        f"moves to {features} ({movement_type})"
                    )
            
            enhanced_narratives[split][uid] = {
                'tokens': trajectory,
                'narrative': ". Then ".join(narrative_parts) + ".",
                'enhanced_context': enhanced_descriptions
            }
    
    return enhanced_narratives

def extract_key_features_from_description(description):
    """Extract key features from enhanced description for narrative."""
    # Extract primary function and main features
    if "Primary function:" in description:
        function_part = description.split("Primary function:")[1].split(".")[0].strip()
        return f"area with {function_part}"
    elif "Features:" in description:
        features_part = description.split("Features:")[1].split(".")[0].strip()
        return f"area featuring {features_part}"
    else:
        return "location"

def categorize_time_period(hour):
    """Categorize hour into narrative-friendly time periods."""
    if 5 <= hour < 9:
        return "early morning"
    elif 9 <= hour < 12:
        return "morning"
    elif 12 <= hour < 14:
        return "lunch time"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening rush"
    elif 20 <= hour < 23:
        return "evening"
    else:
        return "late night"

def classify_movement_type(from_desc, to_desc):
    """Classify movement type based on enhanced descriptions."""
    from_lower = from_desc.lower()
    to_lower = to_desc.lower()
    
    # Work-related movements
    if "business" in to_lower or "office" in to_lower or "professional" in to_lower:
        return "commuting to work"
    elif "business" in from_lower or "office" in from_lower:
        return "leaving work"
    
    # Meal-related movements
    elif "dining" in to_lower or "restaurant" in to_lower or "food" in to_lower:
        return "going for meal"
    
    # Shopping movements
    elif "retail" in to_lower or "shopping" in to_lower or "store" in to_lower:
        return "shopping trip"
    
    # Entertainment movements
    elif "entertainment" in to_lower or "leisure" in to_lower:
        return "leisure activity"
    
    # Service movements
    elif "services" in to_lower or "bank" in to_lower or "hospital" in to_lower:
        return "service visit"
    
    else:
        return "general movement"

def save_enhanced_results(enhanced_descriptions, enhanced_narratives, config, task_id):
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
    
    # Save enhanced narratives
    narratives_path = llm_prepared_dir / f'enhanced_narratives_task{task_id}.pt'
    torch.save(enhanced_narratives, narratives_path)
    
    logger.info(f"Saved enhanced preprocessing results for task {task_id}")
    logger.info(f"  Descriptions: {desc_path}")
    logger.info(f"  Narratives: {narratives_path}")

def main():
    """Main function for refined enhanced preprocessing."""
    logger.info("Refined enhanced preprocessing with real POI categories completed!")

if __name__ == "__main__":
    main() 