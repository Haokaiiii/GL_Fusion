"""
Centralized configuration for data processing parameters.
This module contains all the hardcoded values that were scattered throughout the codebase.
"""

import os
from typing import Optional


class DataConfig:
    """Configuration class for data processing parameters."""
    
    # Memory-based chunk sizes
    CHUNK_SIZE_LOW_MEM = 500      # For systems with < 4GB available memory
    CHUNK_SIZE_MED_MEM = 1000     # For systems with 4-8GB available memory  
    CHUNK_SIZE_HIGH_MEM = 2000    # For systems with > 8GB available memory
    
    # Memory thresholds (in bytes)
    LOW_MEMORY_THRESHOLD = 4 * 1024**3   # 4GB
    HIGH_MEMORY_THRESHOLD = 8 * 1024**3  # 8GB
    
    # Sampling parameters
    MAX_VAL_SAMPLES = 5000              # Maximum validation samples for fast development
    MAX_VAL_SAMPLES_THRESHOLD = 10000   # Only subsample validation if above this
    PARALLEL_PROCESSING_THRESHOLD = 1000 # Minimum samples for parallel processing
    MAX_PARALLEL_WORKERS = 4            # Maximum parallel workers
    LARGE_DATASET_THRESHOLD = 1000000   # Threshold for "large" datasets
    
    # Sample stride parameters
    SAMPLE_STRIDE_THRESHOLD = 1000      # Apply stride if more samples than this
    SAMPLE_STRIDE_LARGE = 5             # Stride for large datasets
    SAMPLE_STRIDE_VERY_LARGE = 1000     # Stride for very large datasets
    SAMPLE_STRIDE_HUGE_THRESHOLD = 2000 # Threshold for applying huge stride
    
    # Batch processing
    MAX_NODES_PER_BATCH = 50000        # Maximum nodes to process in a single batch
    
    # Cache configuration
    CACHE_ENABLED = True                # Enable/disable coordinate caching
    
    @classmethod
    def get_chunk_size(cls, available_memory: Optional[int] = None) -> int:
        """
        Get appropriate chunk size based on available memory.
        
        Args:
            available_memory: Available memory in bytes. If None, will try to detect.
            
        Returns:
            Appropriate chunk size for the system
        """
        if available_memory is None:
            try:
                import psutil
                available_memory = psutil.virtual_memory().available
            except ImportError:
                # Default to medium chunk size if we can't detect memory
                return cls.CHUNK_SIZE_MED_MEM
        
        if available_memory < cls.LOW_MEMORY_THRESHOLD:
            return cls.CHUNK_SIZE_LOW_MEM
        elif available_memory < cls.HIGH_MEMORY_THRESHOLD:
            return cls.CHUNK_SIZE_MED_MEM
        else:
            return cls.CHUNK_SIZE_HIGH_MEM
    
    @classmethod
    def get_sample_stride(cls, num_samples: int) -> int:
        """
        Get appropriate sample stride based on dataset size.
        
        Args:
            num_samples: Number of samples in the dataset
            
        Returns:
            Stride value to use for sampling
        """
        if num_samples > cls.SAMPLE_STRIDE_HUGE_THRESHOLD:
            return max(1, num_samples // cls.SAMPLE_STRIDE_VERY_LARGE)
        elif num_samples > cls.SAMPLE_STRIDE_THRESHOLD:
            return cls.SAMPLE_STRIDE_LARGE
        else:
            return 1
    
    @classmethod
    def should_use_parallel(cls, num_items: int) -> bool:
        """
        Determine if parallel processing should be used.
        
        Args:
            num_items: Number of items to process
            
        Returns:
            True if parallel processing should be used
        """
        return num_items > cls.PARALLEL_PROCESSING_THRESHOLD
    
    @classmethod
    def get_max_workers(cls) -> int:
        """
        Get the maximum number of parallel workers.
        
        Returns:
            Maximum number of workers to use
        """
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            return min(cls.MAX_PARALLEL_WORKERS, cpu_count)
        except Exception:
            return cls.MAX_PARALLEL_WORKERS
    
    @classmethod
    def should_subsample_validation(cls, num_samples: int) -> bool:
        """
        Determine if validation dataset should be subsampled.
        
        Args:
            num_samples: Number of validation samples
            
        Returns:
            True if subsampling should be applied
        """
        max_val_users = int(os.environ.get('MAX_VAL_USERS', '0'))
        return max_val_users > 0 and num_samples > cls.MAX_VAL_SAMPLES_THRESHOLD 