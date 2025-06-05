"""
Wrapper for Large Language Model (LLM) - Qwen2.5.
"""

import os
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class QwenModel(nn.Module):
    """Wrapper for Qwen model from Hugging Face."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Qwen model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        llm_config = config.get('llm', {})
        
        # Model path configuration
        model_name = llm_config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')
        local_model_path = llm_config.get('local_model_path')
        model_path = local_model_path if local_model_path else model_name
        
        logger.info(f"Initializing Qwen model from: {model_path}")
        
        # Initialize tokenizer
        self._init_tokenizer(model_path)
        
        # Initialize model with optional quantization
        self._init_model(model_path, llm_config)
        
        # Apply LoRA if configured
        if llm_config.get('use_lora', False):
            self._apply_lora(llm_config)
    
    def _init_tokenizer(self, model_path: str):
        """Initialize tokenizer with special tokens."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            pad_token='<|endoftext|>'
        )
        
        # Add special tokens
        special_tokens = ['<graph_start>', '<graph_end>', '<node>']
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        
        if num_added > 0:
            logger.info(f"Added {num_added} special tokens to tokenizer")
    
    def _init_model(self, model_path: str, llm_config: Dict[str, Any]):
        """Initialize the model with optional quantization."""
        # Check if quantization is disabled
        disable_quantization = (
            os.environ.get('BNB_8BIT_ENABLED', '0') == '0' or
            llm_config.get('quantization', {}).get('disable', False)
        )
        
        # Model configuration
        model_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
            'torch_dtype': torch.bfloat16,
            'device_map': None  # For DeepSpeed compatibility
        }
        
        # Add quantization if not disabled
        if not disable_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4"
            )
            model_kwargs['quantization_config'] = quantization_config
            logger.info("Loading model with 8-bit quantization")
        else:
            logger.info("Loading model without quantization")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Freeze parameters by default
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _apply_lora(self, llm_config: Dict[str, Any]):
        """Apply LoRA configuration to the model."""
        lora_config = llm_config.get('lora_config', {})
        
        # Prepare model for LoRA training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', 'none'),
            target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("Applied LoRA configuration")
    
    def encode_sequence(self, node_ids: list, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Encode a sequence of node IDs for input to the LLM.
        
        Args:
            node_ids: List of node IDs
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with encoded inputs
        """
        # Convert node IDs to text
        text_sequence = " ".join(str(nid) for nid in node_ids)
        
        # Determine max length
        if max_length is None:
            max_length = self.config['llm'].get('sequence_length', 512)
        
        # Tokenize
        inputs = self.tokenizer(
            text_sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        return inputs
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Any:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for computing loss
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
    
    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        return self.model.config.hidden_size 