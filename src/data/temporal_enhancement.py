"""
Temporal context enhancement for better LLM utilization in trajectory prediction.
Adds temporal reasoning and pattern recognition to leverage LLM's temporal understanding.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TemporalContextEnhancer:
    """Enhances temporal context for better LLM understanding."""
    
    def __init__(self, config):
        self.config = config
        self.temporal_patterns = {}
        self.seasonal_patterns = {}
        
    def add_rich_temporal_features(self, task_df, task_id):
        """Add rich temporal features that help LLM understand time patterns."""
        
        logger.info(f"Adding rich temporal features for task {task_id}...")
        
        enhanced_df = task_df.copy()
        
        # Add cyclical time features
        enhanced_df['hour_sin'] = np.sin(2 * np.pi * (enhanced_df['t'] % 24) / 24)
        enhanced_df['hour_cos'] = np.cos(2 * np.pi * (enhanced_df['t'] % 24) / 24)
        enhanced_df['day_sin'] = np.sin(2 * np.pi * enhanced_df['d'] / 75)  # 75 days cycle
        enhanced_df['day_cos'] = np.cos(2 * np.pi * enhanced_df['d'] / 75)
        
        # Add day of week (assuming d=1 is day 1)
        enhanced_df['day_of_week'] = (enhanced_df['d'] - 1) % 7
        enhanced_df['is_weekend'] = enhanced_df['day_of_week'].isin([5, 6])
        
        # Add time period categories
        enhanced_df['time_period'] = enhanced_df['t'].apply(self._categorize_time_period)
        
        # Add temporal descriptions
        enhanced_df['temporal_context'] = enhanced_df.apply(
            lambda row: self._create_temporal_context(row), axis=1
        )
        
        return enhanced_df
    
    def _categorize_time_period(self, hour):
        """Categorize hour into meaningful time periods."""
        hour = hour % 24
        if 5 <= hour < 9:
            return "early_morning_commute"
        elif 9 <= hour < 12:
            return "morning_business"
        elif 12 <= hour < 14:
            return "lunch_period"
        elif 14 <= hour < 17:
            return "afternoon_activity"
        elif 17 <= hour < 20:
            return "evening_commute"
        elif 20 <= hour < 23:
            return "evening_leisure"
        else:
            return "late_night"
    
    def _create_temporal_context(self, row):
        """Create rich temporal context description."""
        day = row['d']
        hour = row['t'] % 24
        day_of_week = (day - 1) % 7
        
        # Day context
        if day_of_week < 5:  # Weekday
            day_context = "weekday"
        else:  # Weekend
            day_context = "weekend"
        
        # Time context
        time_context = self._categorize_time_period(hour)
        
        # Combine contexts
        context_parts = [day_context, time_context]
        
        # Add special temporal markers
        if day <= 7:
            context_parts.append("early_period")
        elif day >= 68:
            context_parts.append("late_period")
        
        return "_".join(context_parts)

    def analyze_user_temporal_patterns(self, task_df):
        """Analyze individual user temporal patterns."""
        logger.info("Analyzing user temporal patterns...")
        
        user_patterns = {}
        
        for uid in task_df['uid'].unique():
            user_data = task_df[task_df['uid'] == uid]
            
            # Analyze activity patterns
            hourly_activity = user_data.groupby(user_data['t'] % 24).size()
            daily_activity = user_data.groupby('d').size()
            
            # Find peak activity times
            peak_hours = hourly_activity.nlargest(3).index.tolist()
            
            # Classify user type based on patterns
            user_type = self._classify_user_type(hourly_activity, daily_activity)
            
            user_patterns[uid] = {
                'peak_hours': peak_hours,
                'user_type': user_type,
                'activity_variance': hourly_activity.std(),
                'total_locations': len(user_data[['x', 'y']].drop_duplicates())
            }
        
        return user_patterns
    
    def _classify_user_type(self, hourly_activity, daily_activity):
        """Classify user based on temporal activity patterns."""
        
        # Check for commuter pattern (peaks at 7-9 and 17-19)
        morning_peak = hourly_activity.loc[7:9].sum() if len(hourly_activity.loc[7:9]) > 0 else 0
        evening_peak = hourly_activity.loc[17:19].sum() if len(hourly_activity.loc[17:19]) > 0 else 0
        total_activity = hourly_activity.sum()
        
        if (morning_peak + evening_peak) / total_activity > 0.4:
            return "regular_commuter"
        
        # Check for irregular pattern
        if hourly_activity.std() > hourly_activity.mean():
            return "irregular_traveler"
        
        # Check for evening-focused activity
        evening_activity = hourly_activity.loc[18:23].sum() if len(hourly_activity.loc[18:23]) > 0 else 0
        if evening_activity / total_activity > 0.5:
            return "evening_active"
        
        return "regular_user"

    def create_temporal_prompts(self, trajectory_sequence, user_patterns, target_time):
        """Create temporal prompts that help LLM understand the prediction context."""
        
        if not trajectory_sequence:
            return "No trajectory history available."
        
        # Get user type
        uid = trajectory_sequence[0].get('uid', 0)
        user_type = user_patterns.get(uid, {}).get('user_type', 'regular_user')
        
        # Create temporal reasoning prompt
        prompt_parts = []
        
        # User context
        prompt_parts.append(f"User profile: {user_type.replace('_', ' ')}")
        
        # Recent activity pattern
        recent_locations = []
        for item in trajectory_sequence[-5:]:  # Last 5 locations
            hour = item['time'] % 24
            time_desc = self._categorize_time_period(hour)
            recent_locations.append(f"{item['description']} at {time_desc}")
        
        if recent_locations:
            prompt_parts.append(f"Recent activity: {' → '.join(recent_locations)}")
        
        # Prediction context
        target_hour = target_time % 24
        target_period = self._categorize_time_period(target_hour)
        prompt_parts.append(f"Predicting next location for {target_period}")
        
        # Temporal reasoning
        reasoning = self._generate_temporal_reasoning(
            trajectory_sequence, target_time, user_type
        )
        if reasoning:
            prompt_parts.append(f"Temporal context: {reasoning}")
        
        return ". ".join(prompt_parts) + "."
    
    def _generate_temporal_reasoning(self, trajectory, target_time, user_type):
        """Generate temporal reasoning to help LLM understand patterns."""
        
        if len(trajectory) < 2:
            return ""
        
        current_time = trajectory[-1]['time']
        target_hour = target_time % 24
        current_hour = current_time % 24
        
        reasoning_parts = []
        
        # Time progression reasoning
        if target_hour > current_hour:
            time_diff = target_hour - current_hour
            if time_diff <= 2:
                reasoning_parts.append("short-term continuation expected")
            else:
                reasoning_parts.append("significant time progression")
        
        # Pattern-based reasoning
        if user_type == "regular_commuter":
            if 7 <= target_hour <= 9:
                reasoning_parts.append("morning commute period")
            elif 17 <= target_hour <= 19:
                reasoning_parts.append("evening commute period")
        
        # Activity type reasoning
        if 12 <= target_hour <= 14:
            reasoning_parts.append("lunch period activity likely")
        elif 20 <= target_hour <= 23:
            reasoning_parts.append("evening leisure time")
        
        return ", ".join(reasoning_parts)

def enhance_tokenizer_with_temporal_tokens(tokenizer, config):
    """Add temporal-specific tokens to the tokenizer."""
    
    logger.info("Adding temporal tokens to tokenizer...")
    
    temporal_tokens = [
        # Time periods
        "<EARLY_MORNING>", "<MORNING>", "<AFTERNOON>", "<EVENING>", "<NIGHT>",
        
        # Day types
        "<WEEKDAY>", "<WEEKEND>", "<HOLIDAY>",
        
        # User types
        "<COMMUTER>", "<IRREGULAR>", "<EVENING_ACTIVE>", "<REGULAR>",
        
        # Movement types
        "<COMMUTE_HOME>", "<COMMUTE_WORK>", "<LEISURE>", "<SHOPPING>", "<MEAL>",
        
        # Temporal patterns
        "<REPEAT_VISIT>", "<NEW_LOCATION>", "<ROUTINE>", "<IRREGULAR_PATTERN>",
        
        # Spatial context
        "<NEARBY>", "<DISTANT>", "<RETURN>", "<EXPLORE>"
    ]
    
    # Add tokens to tokenizer
    num_added = tokenizer.add_tokens(temporal_tokens)
    logger.info(f"Added {num_added} temporal tokens to tokenizer")
    
    return tokenizer, temporal_tokens

def create_enhanced_trajectory_representation(trajectory, node_descriptions, 
                                           temporal_enhancer, user_patterns):
    """Create enhanced trajectory representation with rich temporal context."""
    
    if not trajectory:
        return []
    
    enhanced_repr = []
    
    for i, (node_id, day, time) in enumerate(trajectory):
        # Get base description
        base_desc = node_descriptions.get(node_id, f"Location {node_id}")
        
        # Add temporal context
        temporal_context = temporal_enhancer._create_temporal_context({
            'd': day, 't': time
        })
        
        # Add sequence context
        sequence_context = "start" if i == 0 else "continue"
        if i == len(trajectory) - 1:
            sequence_context = "end"
        
        # Create enhanced representation
        enhanced_item = {
            'node_id': node_id,
            'description': base_desc,
            'temporal_context': temporal_context,
            'sequence_context': sequence_context,
            'day': day,
            'time': time,
            'hour': time % 24,
            'position_in_sequence': i
        }
        
        # Add movement context for non-first items
        if i > 0:
            prev_node = trajectory[i-1][0]
            movement_type = classify_movement_pattern(
                prev_node, node_id, node_descriptions
            )
            enhanced_item['movement_type'] = movement_type
        
        enhanced_repr.append(enhanced_item)
    
    return enhanced_repr

def classify_movement_pattern(from_node, to_node, descriptions):
    """Classify movement pattern between nodes."""
    from_desc = descriptions.get(from_node, "").lower()
    to_desc = descriptions.get(to_node, "").lower()
    
    # Analyze movement based on POI types
    if "residential" in from_desc and "work" in to_desc:
        return "work_commute"
    elif "work" in from_desc and "residential" in to_desc:
        return "home_commute"
    elif "dining" in to_desc or "food" in to_desc:
        return "meal_trip"
    elif "shopping" in to_desc or "retail" in to_desc:
        return "shopping_trip"
    elif "entertainment" in to_desc or "recreation" in to_desc:
        return "leisure_trip"
    elif from_node == to_node:
        return "stationary"
    else:
        return "general_movement" 