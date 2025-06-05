"""
Example of applying the optimization framework to existing code.
This demonstrates the pattern for refactoring the codebase.
"""

# BEFORE: Old style with problems
def old_process_data():
    import gc  # Import inside function
    
    # Hardcoded values
    chunk_size = 1000
    if memory < 4 * 1024**3:
        chunk_size = 500
    
    for chunk in chunks:
        process_chunk(chunk)
        gc.collect()  # Excessive gc.collect()
    
    try:
        risky_operation()
    except:  # Bare except
        pass


# AFTER: Optimized version using new utilities
from src.config.data_config import DataConfig
from src.utils.memory_management import MemoryManager, strategic_gc
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@strategic_gc(threshold_mb=500)
def new_process_data():
    """Process data with optimized memory management."""
    
    # Use centralized configuration
    chunk_size = DataConfig.get_chunk_size()
    
    # Use memory manager for strategic cleanup
    with MemoryManager():
        for chunk in chunks:
            process_chunk(chunk)
            # No manual gc.collect() needed!
    
    # Proper error handling
    try:
        risky_operation()
    except (ValueError, IOError) as e:
        logger.warning(f"Operation failed: {e}")


# Example of refactoring parallel processing
def optimized_parallel_processing(items):
    """Example of using configuration for parallel processing."""
    
    if not DataConfig.should_use_parallel(len(items)):
        # Process sequentially
        return [process_item(item) for item in items]
    
    # Use parallel processing with configured workers
    from concurrent.futures import ProcessPoolExecutor
    
    n_workers = DataConfig.get_max_workers()
    logger.info(f"Processing {len(items)} items with {n_workers} workers")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_item, items))
    
    return results


# Example of memory-efficient batch processing
class OptimizedDataProcessor:
    """Example of applying all optimizations to a class."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = DataConfig()
    
    def process_dataset(self, dataset):
        """Process dataset with all optimizations applied."""
        
        # Log start
        self.logger.info(f"Processing dataset with {len(dataset)} items")
        
        # Determine processing strategy
        chunk_size = self.config.get_chunk_size()
        use_parallel = self.config.should_use_parallel(len(dataset))
        
        # Process with memory management
        with MemoryManager(force_cleanup=len(dataset) > 10000):
            if use_parallel:
                return self._parallel_process(dataset, chunk_size)
            else:
                return self._sequential_process(dataset, chunk_size)
    
    @strategic_gc(threshold_mb=1000)
    def _parallel_process(self, dataset, chunk_size):
        """Parallel processing with memory management."""
        # Implementation here
        pass
    
    def _sequential_process(self, dataset, chunk_size):
        """Sequential processing with memory management."""
        # Implementation here
        pass 