"""
Main entry point for GL-Fusion training.
This provides a clean, modular training script.
"""

import os
import sys
import logging
import gc
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training_v2.core.config import ConfigManager, parse_args
from src.training_v2.distributed.manager import DistributedManager
from src.training_v2.models.factory import ModelFactory
from src.training_v2.data.loader import DataLoaderFactory
from src.training_v2.core.trainer import Trainer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up environment variables for optimal performance."""
    # DeepSpeed environment variables
    os.environ["DS_DISABLE_CUSTOM_OPS"] = "1"
    os.environ["DS_DISABLE_TRITON"] = "1"
    os.environ["DEEPSPEED_DISABLE_TRITON"] = "1"
    
    # PyTorch optimizations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # CUDA optimizations
    if "CUDA_HOME" not in os.environ:
        cuda_paths = ["/apps/cuda/12.3.2", "/apps/cuda/12.4.1", "/usr/local/cuda"]
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                os.environ["CUDA_HOME"] = cuda_path
                break


def main():
    """Main training function."""
    # Setup environment
    setup_environment()
    
    # Parse arguments
    args = parse_args()
    
    # Setup configuration
    logger.info("Loading configuration...")
    config_manager = ConfigManager(args.config)
    config_manager.merge_args(args)
    config = config_manager.get_config()
    
    # Setup distributed training
    logger.info("Setting up distributed training...")
    dist_manager = DistributedManager()
    rank, local_rank, world_size, device = dist_manager.setup()
    
    if dist_manager.is_main_process():
        logger.info(f"Configuration loaded from {args.config}")
        logger.info(f"Task ID: {config['data']['task_id']}")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = DataLoaderFactory.create_data_loaders(
        config, dist_manager
    )
    
    # Build model
    logger.info("Building model...")
    model = ModelFactory.build_model(config)
    
    # Precompute graph embeddings for efficiency
    if hasattr(model, 'precompute_graph_embeddings'):
        logger.info("Precomputing graph embeddings...")
        model.precompute_graph_embeddings()
    
    # Calculate total training steps
    num_epochs = config['training']['num_epochs']
    steps_per_epoch = len(train_loader) // config['training'].get('gradient_accumulation_steps', 1)
    total_steps = steps_per_epoch * num_epochs
    
    # Build optimizer and scheduler
    logger.info("Building optimizer and scheduler...")
    optimizer = ModelFactory.build_optimizer(model, config)
    scheduler = ModelFactory.build_scheduler(optimizer, config, total_steps)
    
    # Handle DeepSpeed if requested
    use_deepspeed = args.deepspeed and args.deepspeed_config
    if use_deepspeed:
        try:
            import deepspeed
            # DeepSpeed will be handled differently - keeping it simple for now
            logger.warning("DeepSpeed integration not fully implemented in modular version yet")
            use_deepspeed = False
        except ImportError:
            logger.warning("DeepSpeed requested but not available")
            use_deepspeed = False
    
    # Wrap model for distributed training
    if not use_deepspeed:
        model = ModelFactory.wrap_model_for_distributed(
            model, device, dist_manager, use_ddp=dist_manager.is_distributed
        )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        distributed_manager=dist_manager,
        device=device
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_checkpoint:
        start_epoch = trainer.load_checkpoint(args.resume_checkpoint)
    
    # Training
    if dist_manager.is_main_process():
        logger.info(f"Starting training for {num_epochs} epochs...")
        
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        start_epoch=start_epoch
    )
    
    # Cleanup
    dist_manager.cleanup()
    
    if dist_manager.is_main_process():
        logger.info("Training completed!")


if __name__ == "__main__":
    main() 