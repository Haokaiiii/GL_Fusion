"""
Data loader factory for GL-Fusion training.
Simplified data loading without complex caching logic.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_data_loaders(
        config: Dict[str, Any],
        distributed_manager: Any
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create train, validation, and optionally test data loaders.
        
        Args:
            config: Configuration dictionary
            distributed_manager: Distributed training manager
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Import the actual data loading function
        try:
            from src.training.utils import load_data_for_training
        except ImportError:
            from training.utils import load_data_for_training
            
        # Get task ID
        task_id = int(config['data']['task_id'])
        
        # Load datasets
        logger.info(f"Loading datasets for task {task_id}...")
        data_result = load_data_for_training(
            config, 
            task_id,
            rank=distributed_manager.rank,
            world_size=distributed_manager.world_size
        )
        
        # Handle different return formats
        if len(data_result) == 6:
            train_dataset, val_dataset, test_dataset, train_sampler, val_sampler, test_sampler = data_result
        else:
            # Older version returns only datasets
            train_dataset, val_dataset = data_result[:2]
            test_dataset = data_result[2] if len(data_result) > 2 else None
            train_sampler = val_sampler = test_sampler = None
            
        # Create samplers if not provided
        if train_sampler is None:
            train_sampler = DataLoaderFactory._create_sampler(
                train_dataset, 
                distributed_manager,
                shuffle=True
            )
            
        if val_sampler is None:
            val_sampler = DataLoaderFactory._create_sampler(
                val_dataset,
                distributed_manager,
                shuffle=False
            )
            
        if test_dataset is not None and test_sampler is None:
            test_sampler = DataLoaderFactory._create_sampler(
                test_dataset,
                distributed_manager,
                shuffle=False
            )
            
        # Create data loaders
        batch_size = config['training']['batch_size']
        num_workers = config['data'].get('num_workers', 0)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            drop_last=distributed_manager.is_distributed
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['evaluation'].get('batch_size', batch_size),
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
            drop_last=False
        )
        
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['evaluation'].get('batch_size', batch_size),
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=test_dataset.collate_fn,
                drop_last=False
            )
            
        if distributed_manager.is_main_process():
            logger.info(f"Created data loaders:")
            logger.info(f"  Train: {len(train_loader)} batches")
            logger.info(f"  Val: {len(val_loader)} batches")
            if test_loader:
                logger.info(f"  Test: {len(test_loader)} batches")
                
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def _create_sampler(dataset, distributed_manager, shuffle: bool):
        """Create appropriate sampler based on distributed setup."""
        if distributed_manager.is_distributed:
            return DistributedSampler(
                dataset,
                num_replicas=distributed_manager.world_size,
                rank=distributed_manager.rank,
                shuffle=shuffle,
                seed=42
            )
        else:
            if shuffle:
                return RandomSampler(dataset)
            else:
                return SequentialSampler(dataset) 