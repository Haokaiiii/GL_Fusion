"""
Model factory for GL-Fusion training.
Handles model, optimizer, and scheduler creation.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.config.token_config import TokenConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating models, optimizers, and schedulers."""
    
    @staticmethod
    def build_model(config: Dict[str, Any]) -> nn.Module:
        """
        Build the GL-Fusion model.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            GLFusionModel instance
        """
        # Import here to avoid circular dependencies
        try:
            from model.gl_fusion_model import GLFusionModel
        except ImportError:
            from src.model.gl_fusion_model import GLFusionModel
            
        logger.info("Building GL-Fusion model...")
        model = GLFusionModel(config)
        
        # The resizing is now handled inside the LLMWrapper.
        # This call is no longer needed and causes errors with LoRA.
        # tokenizer = model.llm.tokenizer
        # if len(tokenizer) != model.llm.model.get_input_embeddings().weight.shape[0]:
        #     logger.info(f"Resizing LLM token embeddings to {len(tokenizer)}")
        #     model.llm.model.resize_token_embeddings(len(tokenizer))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model built: {total_params:,} total params, {trainable_params:,} trainable params")
        
        return model
    
    @staticmethod
    def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """
        Build the optimizer for the model.
        
        Args:
            model: The model to optimize
            config: Training configuration dictionary
            
        Returns:
            Optimizer instance
        """
        # Get optimizer parameters from config
        learning_rate = float(config['training'].get('learning_rate', 5e-5))
        weight_decay = config['training'].get('weight_decay', 0.01)
        optimizer_type = config['training'].get('optimizer', 'adamw')
        
        # Filter parameters that require gradients
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                params_to_optimize,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                params_to_optimize,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        logger.info(f"Optimizer: {optimizer.__class__.__name__}, lr={learning_rate}, "
                   f"weight_decay={weight_decay}")
        
        return optimizer
    
    @staticmethod
    def build_scheduler(
        optimizer: optim.Optimizer, 
        config: Dict[str, Any], 
        total_steps: int
    ) -> Optional[object]:
        """
        Build the learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            config: Training configuration dictionary
            total_steps: Total number of training steps
            
        Returns:
            Learning rate scheduler instance or None
        """
        scheduler_config = config['training'].get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'linear').lower()
        
        if scheduler_type == 'none':
            return None
            
        elif scheduler_type == 'linear':
            # Linear schedule with warmup
            warmup_steps = scheduler_config.get('num_warmup_steps', 100)
            if 'warmup_ratio' in scheduler_config:
                warmup_ratio = scheduler_config['warmup_ratio']
                warmup_steps = int(total_steps * warmup_ratio)
                
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            logger.info(f"Scheduler: Linear with {warmup_steps} warmup steps")
            
        elif scheduler_type == 'reduce_on_plateau':
            # ReduceLROnPlateau scheduler
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 2),
                verbose=True
            )
            logger.info(f"Scheduler: ReduceLROnPlateau, factor={scheduler_config.get('factor', 0.5)}, "
                       f"patience={scheduler_config.get('patience', 2)}")
            
        elif scheduler_type == 'cosine':
            # Cosine annealing
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=scheduler_config.get('eta_min', 0)
            )
            logger.info(f"Scheduler: CosineAnnealing, T_max={total_steps}")
            
        else:
            # Default to constant learning rate
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            logger.info("Scheduler: Constant learning rate")
            
        return scheduler
    
    @staticmethod
    def _get_tokenizer(config: Dict[str, Any]):
        """Get tokenizer for the model."""
        model_path = config['llm'].get('local_model_path', config['llm']['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer
    
    @staticmethod
    def wrap_model_for_distributed(
        model: nn.Module,
        device: torch.device,
        distributed_manager: Any,
        use_ddp: bool = True
    ) -> nn.Module:
        """
        Wrap model for distributed training.
        
        Args:
            model: The model to wrap
            device: The device to use
            distributed_manager: The distributed manager instance
            use_ddp: Whether to use DDP (vs keeping model as-is)
            
        Returns:
            Wrapped model
        """
        # Move model to device
        model = model.to(device)
        
        # Wrap with DDP if in distributed mode
        if use_ddp and distributed_manager.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            logger.info("Wrapping model with DistributedDataParallel")
            model = DDP(
                model,
                device_ids=[distributed_manager.local_rank] if device.type == 'cuda' else None,
                output_device=distributed_manager.local_rank if device.type == 'cuda' else None,
                find_unused_parameters=True
            )
            # Set the static graph flag to resolve checkpointing conflicts with DDP
            model._set_static_graph()
            
        return model 