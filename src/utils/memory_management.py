"""
Memory management utilities for optimized garbage collection.
Replaces excessive gc.collect() calls with strategic memory management.
"""

import gc
import logging
import torch
import functools
from typing import Optional, Callable, Any
import weakref

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Context manager for strategic memory cleanup.
    Use this instead of manual gc.collect() calls.
    """
    
    # Class-level counter to track cleanup frequency
    _cleanup_counter = 0
    _cleanup_frequency = 10  # Only cleanup every N uses
    
    def __init__(self, force_cleanup: bool = False, gpu_cleanup: bool = True):
        """
        Initialize memory manager.
        
        Args:
            force_cleanup: Force cleanup regardless of frequency
            gpu_cleanup: Also clean GPU memory if available
        """
        self.force_cleanup = force_cleanup
        self.gpu_cleanup = gpu_cleanup
    
    def __enter__(self):
        """Entry point - no action needed."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit point - perform cleanup if needed."""
        MemoryManager._cleanup_counter += 1
        
        # Only cleanup at specified frequency or if forced
        if self.force_cleanup or MemoryManager._cleanup_counter >= MemoryManager._cleanup_frequency:
            self._perform_cleanup()
            MemoryManager._cleanup_counter = 0
    
    def _perform_cleanup(self):
        """Perform actual memory cleanup."""
        gc.collect()
        
        if self.gpu_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.debug("Memory cleanup performed")
    
    @classmethod
    def set_cleanup_frequency(cls, frequency: int):
        """
        Set how often cleanup should occur.
        
        Args:
            frequency: Cleanup every N context manager uses
        """
        cls._cleanup_frequency = max(1, frequency)
        logger.info(f"Memory cleanup frequency set to every {cls._cleanup_frequency} operations")


def strategic_gc(threshold_mb: float = 1000) -> Callable:
    """
    Decorator for strategic garbage collection based on memory usage.
    Only performs gc if memory usage increases by threshold amount.
    
    Args:
        threshold_mb: Memory increase threshold in MB to trigger gc
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Keep weak reference to avoid memory leaks
        _last_memory = weakref.ref(lambda: 0)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory before function
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / (1024**2)  # MB
            else:
                memory_before = 0
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Check memory after function
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / (1024**2)  # MB
                memory_increase = memory_after - memory_before
                
                # Only gc if memory increased significantly
                if memory_increase > threshold_mb:
                    logger.debug(f"Memory increased by {memory_increase:.1f}MB, triggering cleanup")
                    gc.collect()
                    torch.cuda.empty_cache()
            
            return result
        
        return wrapper
    return decorator


class MemoryMonitor:
    """Monitor memory usage and provide insights."""
    
    @staticmethod
    def log_memory_stats(prefix: str = ""):
        """Log current memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, "
                       f"Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")
        
        # Log CPU memory if psutil available
        try:
            import psutil
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / (1024**3)  # GB
            logger.info(f"{prefix}CPU Memory - Process: {cpu_memory:.2f}GB")
        except ImportError:
            pass
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        try:
            import psutil
            process = psutil.Process()
            stats['cpu_memory_gb'] = process.memory_info().rss / (1024**3)
            stats['cpu_memory_percent'] = process.memory_percent()
        except ImportError:
            pass
        
        return stats


def cleanup_on_oom(func: Callable) -> Callable:
    """
    Decorator that performs cleanup on OOM errors and retries.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM error in {func.__name__}, attempting cleanup and retry")
                
                # Aggressive cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Log memory stats
                MemoryMonitor.log_memory_stats("After OOM cleanup: ")
                
                # Retry once
                try:
                    return func(*args, **kwargs)
                except RuntimeError as retry_error:
                    if "out of memory" in str(retry_error).lower():
                        logger.error(f"OOM error persists after cleanup in {func.__name__}")
                    raise
            else:
                raise
    
    return wrapper


# Global memory manager instance for convenience
memory_manager = MemoryManager()


def set_memory_efficient_mode(enabled: bool = True):
    """
    Enable or disable memory efficient mode globally.
    
    Args:
        enabled: Whether to enable memory efficient mode
    """
    if enabled:
        # More frequent cleanup
        MemoryManager.set_cleanup_frequency(5)
        logger.info("Memory efficient mode enabled")
    else:
        # Less frequent cleanup
        MemoryManager.set_cleanup_frequency(20)
        logger.info("Memory efficient mode disabled") 