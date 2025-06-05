"""
Configuration management for GL-Fusion training.
Handles loading configs and merging with command line arguments.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and merging."""
    
    def __init__(self, config_path: str):
        """Initialize with base config file path."""
        self.config_path = Path(config_path)
        self.config = self._load_yaml_config()
        
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded config from {self.config_path}")
        return config
    
    def merge_args(self, args: argparse.Namespace) -> None:
        """Merge command line arguments into config."""
        # Task ID
        if hasattr(args, 'task'):
            self.config['data']['task_id'] = int(args.task)
            
        # Debug mode
        if hasattr(args, 'debug') and args.debug:
            logger.info("Debug mode enabled - reducing dataset and epochs")
            self.config['training']['num_epochs'] = min(2, self.config['training']['num_epochs'])
            self.config['data']['max_samples'] = 100
            
        # Training parameters
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.config['training']['batch_size'] = args.batch_size
            
        if hasattr(args, 'gradient_accumulation_steps') and args.gradient_accumulation_steps is not None:
            self.config['training']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
            
        # Distributed training
        if hasattr(args, 'deepspeed'):
            self.config['training']['use_deepspeed'] = args.deepspeed
            
        if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
            self.config['training']['deepspeed_config'] = args.deepspeed_config
            
        if hasattr(args, 'use_ddp'):
            self.config['training']['use_ddp'] = args.use_ddp
            
        # Environment variables override
        self._merge_env_vars()
        
    def _merge_env_vars(self) -> None:
        """Merge environment variable overrides."""
        # Data subsampling
        if 'DATA_SUBSAMPLE_RATIO' in os.environ:
            self.config['data']['train_data_subsample_ratio'] = float(os.environ['DATA_SUBSAMPLE_RATIO'])
            
        if 'VAL_DATA_SUBSAMPLE_RATIO' in os.environ:
            self.config['data']['val_data_subsample_ratio'] = float(os.environ['VAL_DATA_SUBSAMPLE_RATIO'])
            
        if 'MAX_VAL_USERS' in os.environ:
            self.config['training']['max_val_users'] = int(os.environ['MAX_VAL_USERS'])
            
        # Cache directory
        if 'DATASET_CACHE_DIR' in os.environ:
            self.config['data']['cache_dir'] = os.environ['DATASET_CACHE_DIR']
            
    def get_config(self) -> Dict[str, Any]:
        """Get the final merged configuration."""
        return self.config
    
    def save_config(self, output_path: str) -> None:
        """Save the current configuration to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        logger.info(f"Saved config to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GL-Fusion model")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task", type=str, required=True, help="Task ID (1 or 2)")
    
    # Optional arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Override gradient accumulation")
    
    # Distributed training
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed config file")
    parser.add_argument("--use_ddp", action="store_true", help="Use PyTorch DDP")
    
    # Checkpoint
    parser.add_argument("--resume_checkpoint", type=str, help="Resume from checkpoint")
    
    # LoRA
    parser.add_argument("--finetuning_type", type=str, default="lora", 
                       help="Finetuning type: full, lora, qlora")
    
    return parser.parse_args() 