"""
Patch DeepSpeed to avoid C++ extensions on Gadi.

This module patches DeepSpeed to ensure C++ extensions are not loaded,
which avoids issues with incompatible compilers on NCI Gadi.
"""

import os
import logging
import sys
import importlib
import traceback
import socket
import types
import gc
from typing import Optional, Dict, Any

# Try to import torch - it may not be available when this module is first imported
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# Global variable to track if patches have been applied
_patches_applied = False

def is_mpi_environment():
    """Check if running in an MPI environment."""
    mpi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"]
    return any(var in os.environ for var in mpi_vars)

def get_mpi_info():
    """Get MPI environment information."""
    if not is_mpi_environment():
        return "Not an MPI environment"
    
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", "unknown")
    size = os.environ.get("OMPI_COMM_WORLD_SIZE", "unknown")
    local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "unknown")
    
    return f"MPI environment: rank={rank}, size={size}, local_rank={local_rank}"

def is_v100_gpu():
    """Check if the environment is using V100 GPUs."""
    global _patches_applied
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        logger.info("[is_v100_gpu] CUDA not available or torch not imported. Assuming not V100.")
        return False
    
    try:
        # Get local rank if available, to log which process is checking
        local_rank = os.environ.get("LOCAL_RANK", "N/A")
        device_count = torch.cuda.device_count()
        logger.info(f"[is_v100_gpu, LocalRank {local_rank}] CUDA device count: {device_count}")

        if device_count == 0:
            logger.info(f"[is_v100_gpu, LocalRank {local_rank}] No CUDA devices found. Assuming not V100.")
            return False

        # Check the name of the first GPU (or the GPU assigned to this local_rank if possible)
        # It's safer to check the device corresponding to the current local_rank if set
        # However, this function might be called before torch.cuda.set_device(local_rank)
        # So, checking device 0 is a common practice, assuming homogeneous nodes.
        
        # Let's try to get current device, and if it fails, default to 0.
        current_device_idx = 0
        try:
            current_device_idx = torch.cuda.current_device() # This might be global index or local if set
            logger.info(f"[is_v100_gpu, LocalRank {local_rank}] Current CUDA device index for check: {current_device_idx}")
        except Exception as e:
            logger.warning(f"[is_v100_gpu, LocalRank {local_rank}] Could not get torch.cuda.current_device() (error: {e}). Defaulting to check device 0.")
            current_device_idx = 0 # Fallback to checking the first device

        if current_device_idx >= device_count:
            logger.warning(f"[is_v100_gpu, LocalRank {local_rank}] Current device index {current_device_idx} is out of bounds for count {device_count}. Defaulting to check device 0.")
            current_device_idx = 0
            
        gpu_name = torch.cuda.get_device_name(current_device_idx)
        logger.info(f"[is_v100_gpu, LocalRank {local_rank}] GPU name for device {current_device_idx}: {gpu_name}")
        is_v100 = "V100" in gpu_name
        logger.info(f"[is_v100_gpu, LocalRank {local_rank}] Is V100 based on name '{gpu_name}'? {is_v100}")
        return is_v100
    except Exception as e:
        logger.error(f"[is_v100_gpu, LocalRank {local_rank}] Error checking GPU type: {e}. Assuming not V100.", exc_info=True)
        return False

def apply_v100_optimizations():
    """Apply V100-specific optimizations for DeepSpeed."""
    if not is_v100_gpu():
        logger.info("Not running on V100 GPUs, skipping V100-specific optimizations")
        return
    
    logger.info("Applying V100-specific optimizations for DeepSpeed")
    
    # V100-specific configuration for better memory management
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Ensure consistent device ordering
    os.environ["NCCL_P2P_LEVEL"] = "NVL"  # NVLink optimization for V100
    os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Better for smaller GPU counts
    
    # Only set PYTORCH_CUDA_ALLOC_CONF if not already set (to avoid conflicts)
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        # Note: max_split_size_mb should use : not = for separator
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8"
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF for V100 optimization")
    else:
        logger.info(f"PYTORCH_CUDA_ALLOC_CONF already set to: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    
    # Set ZeRO stage 3 for optimal memory usage on V100
    os.environ["ZERO_STAGE"] = "3"
    
    logger.info("V100-specific optimizations applied")

def apply_multinode_optimizations():
    """Apply optimizations for multi-node training."""
    logger.info("Applying multi-node training optimizations")
    
    # Improve multi-node communication
    os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand
    os.environ["NCCL_SOCKET_IFNAME"] = "ib0"  # Use InfiniBand interface on Gadi
    os.environ["NCCL_DEBUG"] = "INFO"  # More debug information
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Better error handling
    
    # For NCI Gadi's V100 nodes with NVLink
    os.environ["NCCL_P2P_LEVEL"] = "NVL"  # Use NVLink when available
    
    # Better NCCL parameters for Gadi's network topology
    os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Use ring algorithm for small GPU counts
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Improved stability for multi-node
    
    # Reduce buffer sizes to avoid OOM
    os.environ["NCCL_BUFFSIZE"] = "4194304"  # 4 MB, smaller than default
    
    # Further optimize ZeRO-3 for multi-node
    os.environ["ZERO_CONTIGUOUS_GRADIENTS"] = "1"
    os.environ["ZERO_REDUCE_SCATTER"] = "true"
    os.environ["ZERO_OVERLAP_COMM"] = "true"
    
    logger.info("Multi-node optimizations applied")

def apply_patch():
    """
    Apply patches to DeepSpeed to make it work without C++ extensions.
    
    Sets environment variables to disable custom C++ ops and patches specific
    functions in the DeepSpeed codebase.
    
    Returns:
        bool: True if patches were applied, False if already applied
    """
    global _patches_applied
    
    # Avoid reapplying patches
    if _patches_applied:
        logger.debug("DeepSpeed patches already applied, skipping")
        return False
        
    logger.info("Patching DeepSpeed for compatibility with NCI Gadi environment")
    
    # Log MPI environment
    logger.info(get_mpi_info())
    
    # Apply environment variable patches
    os.environ["DS_DISABLE_CUSTOM_OPS"] = "1"
    logger.info("Set DS_DISABLE_CUSTOM_OPS=1 to disable DeepSpeed custom C++ ops")
    
    # Disable Triton to avoid CUDA initialization issues
    os.environ["DS_DISABLE_TRITON"] = "1"
    os.environ["DEEPSPEED_DISABLE_TRITON"] = "1"
    logger.info("Set DS_DISABLE_TRITON=1 and DEEPSPEED_DISABLE_TRITON=1 to disable Triton")
    
    # Apply V100-specific optimizations if detected
    apply_v100_optimizations()
    
    # Check if this is a multi-node job
    is_multinode = False
    if "PBS_NNODES" in os.environ:
        nnodes = int(os.environ["PBS_NNODES"])
        is_multinode = nnodes > 1
        logger.info(f"Detected PBS_NNODES={nnodes}, multi-node job: {is_multinode}")
    
    if is_multinode:
        apply_multinode_optimizations()
    
    # Limit tensor parallelism for NCI Gadi's configuration
    os.environ["DEEPSPEED_TENSOR_PARALLEL_MAX"] = "4"
    logger.info("Set DEEPSPEED_TENSOR_PARALLEL_MAX=4 to avoid memory issues")
    
    # Set up proper environment variables for single-node training
    if is_mpi_environment():
        if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
            os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
            logger.info(f"Set RANK={os.environ['RANK']} from MPI rank")
            
        if "WORLD_SIZE" not in os.environ and "OMPI_COMM_WORLD_SIZE" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
            logger.info(f"Set WORLD_SIZE={os.environ['WORLD_SIZE']} from MPI size")
            
        if "LOCAL_RANK" not in os.environ and "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
            logger.info(f"Set LOCAL_RANK={os.environ['LOCAL_RANK']} from MPI local rank")
            
        # Set MASTER_ADDR and MASTER_PORT if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = socket.gethostname()
            logger.info(f"Set MASTER_ADDR={os.environ['MASTER_ADDR']}")
            
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
            logger.info(f"Set MASTER_PORT=29500")
    
    # Enable checkpoint in CPU for V100 memory constraints
    os.environ["TORCH_GRAD_CHECKPOINT_ENABLED"] = "true"
    logger.info("Set TORCH_GRAD_CHECKPOINT_ENABLED=true to enable gradient checkpointing")
    
    # Disable tensor parallelism as it can cause issues with LoRA
    os.environ["DISABLE_TENSOR_PARALLEL"] = "1"
    logger.info("Set DISABLE_TENSOR_PARALLEL=1 to prevent conflicts with LoRA")
    
    # Attempt to patch specific functions and modules
    try:
        # Patch activation checkpointing if the module is imported
        if 'deepspeed.runtime.activation_checkpointing.checkpointing' in sys.modules:
            sys.modules['deepspeed.runtime.activation_checkpointing.checkpointing'] = None
            logger.info("Patched activation checkpointing module")
    except Exception as e:
        logger.warning(f"Could not patch activation checkpointing: {e}")
    
    # Mark patches as applied
    _patches_applied = True
    
    # Log successful patching
    logger.info("DeepSpeed patching completed successfully")
    
    return True

def check_nci_deepspeed():
    """
    Check if we're running on NCI Gadi with the DeepSpeed module loaded.
    
    Returns:
        tuple: (is_nci, message)
    """
    # Check for indicators of NCI environment
    hostname = os.uname().nodename
    is_nci = "gadi" in hostname.lower()
    
    # Check for DeepSpeed module
    deepspeed_module_loaded = "DEEPSPEED_ROOT" in os.environ
    
    # Check MPI environment
    mpi_env = is_mpi_environment()
    
    # Check for V100 GPUs
    v100_detected = is_v100_gpu()
    
    message = (
        f"Running on NCI Gadi: {is_nci}, "
        f"DeepSpeed module loaded: {deepspeed_module_loaded}, "
        f"MPI environment: {mpi_env}, "
        f"V100 GPUs detected: {v100_detected}"
    )
    
    return is_nci, message

class DeepSpeedPatcher:
    """
    Applies patches to DeepSpeed to make it compatible with Qwen models using LoRA.
    These patches address issues related to DTensor, tensor parallel, and other compatibility issues.
    """
    def __init__(self):
        """Initialize the patcher."""
        self.patched = False
        
    def apply_patches(self):
        """Apply all necessary patches to DeepSpeed."""
        global _patches_applied
        
        # If global patches already applied, just mark this instance as patched
        if _patches_applied:
            self.patched = True
            logger.debug("DeepSpeed patches already applied globally")
            return
            
        if self.patched:
            logger.debug("DeepSpeed patches already applied by this patcher")
            return
        
        logger.info("Applying DeepSpeed patches...")
        
        # First check if DeepSpeed is available
        try:
            import deepspeed
            logger.info(f"Found DeepSpeed version {deepspeed.__version__}")
        except ImportError:
            logger.warning("DeepSpeed not found, skipping patches")
            return
        
        # Apply patches that disable problematic features
        self._patch_dtensor_handling()
        self._patch_tensor_parallel()
        self._patch_memory_cleanup()
        self._patch_v100_optimizations()
        self._patch_zero3_for_stability()
        
        # Check if multi-node and apply specific patches
        is_multinode = False
        if "PBS_NNODES" in os.environ:
            nnodes = int(os.environ["PBS_NNODES"])
            is_multinode = nnodes > 1
        
        if is_multinode:
            self._patch_for_multinode()
        
        # Mark as patched both locally and globally
        self.patched = True
        _patches_applied = True
        logger.info("All DeepSpeed patches applied successfully")
        
    def _patch_dtensor_handling(self):
        """
        Patch DeepSpeed's DTensor handling to avoid compatibility issues.
        This primarily targets the broadcast operations that use DTensor.
        """
        try:
            import deepspeed
            import torch
            
            # Patch deepspeed.runtime.utils to bypass DTensor usage
            if hasattr(deepspeed.runtime, 'utils'):
                utils_module = deepspeed.runtime.utils
                
                # Original broadcast function
                original_broadcast = utils_module.broadcast_scalar
                
                # Define the patched function
                def patched_broadcast(tensor, root_rank=0, group=None):
                    """
                    A patched version of broadcast that ensures we don't use DTensor.
                    Converts any DTensor to a normal tensor before broadcasting.
                    """
                    # Convert DTensor to regular tensor if needed
                    try:
                        if hasattr(tensor, '_c_local_tensor'):
                            # This is likely a DTensor, convert to regular tensor
                            logger.debug("Converting DTensor to regular tensor before broadcast")
                            tensor = tensor._c_local_tensor()
                        
                        # Ensure tensor is a torch.Tensor
                        if not isinstance(tensor, torch.Tensor):
                            tensor = torch.tensor(tensor, device=torch.cuda.current_device())
                            
                        # Force a clean tensor without history for V100
                        if is_v100_gpu():
                            tensor = tensor.detach().clone()
                    except Exception as e:
                        logger.warning(f"Error in patched broadcast during tensor conversion: {e}")
                        # Fall back to a safer approach
                        if not isinstance(tensor, torch.Tensor):
                            tensor = torch.tensor(tensor, device='cpu')
                            if torch.cuda.is_available():
                                try:
                                    tensor = tensor.cuda()
                                except:
                                    pass
                    
                    # Safety call to garbage collection to prevent memory buildup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    try:
                        # Call the original broadcast function
                        return original_broadcast(tensor, root_rank, group)
                    except Exception as e:
                        logger.warning(f"Error in patched broadcast during original call: {e}")
                        # Fall back to a simple return of the tensor
                        return tensor
                
                # Apply the patch
                utils_module.broadcast_scalar = patched_broadcast
                logger.info("Patched deepspeed.runtime.utils.broadcast_scalar to handle DTensor properly")
                
            logger.info("Applied DeepSpeed DTensor handling patches")
        except Exception as e:
            logger.warning(f"Failed to patch DeepSpeed DTensor handling: {e}")
    
    def _patch_tensor_parallel(self):
        """
        Patch DeepSpeed's tensor parallelism to avoid issues with LoRA.
        This disables certain tensor parallel features that conflict with LoRA.
        """
        try:
            import deepspeed
            
            # Check if tensor parallel module exists
            if hasattr(deepspeed, 'module') and hasattr(deepspeed.module, 'tensor_parallel'):
                # Set a flag to disable tensor parallel operations
                if not hasattr(deepspeed.module.tensor_parallel, 'TENSOR_PARALLEL_DISABLED'):
                    deepspeed.module.tensor_parallel.TENSOR_PARALLEL_DISABLED = True
                    logger.info("Set TENSOR_PARALLEL_DISABLED flag to True")
                
            logger.info("Applied DeepSpeed tensor parallel patches")
        except Exception as e:
            logger.warning(f"Failed to patch DeepSpeed tensor parallel: {e}")

    def _patch_memory_cleanup(self):
        """
        Patch DeepSpeed's memory management functions to add more aggressive cleanup
        to prevent memory leaks and fragmentation.
        """
        try:
            import deepspeed
            import torch
            
            # Patch stage manager to add memory cleanup
            if hasattr(deepspeed, 'runtime') and hasattr(deepspeed.runtime, 'zero'):
                if hasattr(deepspeed.runtime.zero, 'stage_1_and_2'):
                    stage_manager = deepspeed.runtime.zero.stage_1_and_2
                    
                    # Original function
                    original_forward = stage_manager.DeepSpeedZeroOptimizer.forward
                    
                    # Define the patched function
                    def patched_forward(self, *inputs, **kwargs):
                        """
                        A patched version of forward that adds memory cleanup.
                        """
                        try:
                            # Clean memory before forward
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Call the original forward function
                            outputs = original_forward(self, *inputs, **kwargs)
                            
                            # Add memory cleanup after forward
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                            return outputs
                        except Exception as e:
                            logger.error(f"Error in patched forward: {e}")
                            # Try to clean up memory even in case of error
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Re-raise the exception
                            raise
                    
                    # Apply the patch
                    stage_manager.DeepSpeedZeroOptimizer.forward = patched_forward
                    logger.info("Patched DeepSpeedZeroOptimizer.forward to add memory cleanup")
            
            logger.info("Applied DeepSpeed memory cleanup patches")
        except Exception as e:
            logger.warning(f"Failed to patch DeepSpeed memory cleanup: {e}")
    
    def _patch_v100_optimizations(self):
        """
        Apply V100-specific patches to DeepSpeed for optimal performance.
        """
        if not is_v100_gpu():
            logger.info("Not running on V100 GPUs, skipping V100-specific patches")
            return
        
        try:
            import deepspeed
            
            # Only proceed if torch is available
            if not TORCH_AVAILABLE:
                logger.warning("Torch not available, skipping V100-specific DeepSpeed patches")
                return
            
            # Patch DeepSpeed's memory allocation strategies
            if hasattr(deepspeed, 'utils') and hasattr(deepspeed.utils, 'memory'):
                memory_utils = deepspeed.utils.memory
                
                # Reduce memory allocation granularity
                if hasattr(memory_utils, 'MEMORY_ALLOCATION_GRANULARITY'):
                    original_granularity = memory_utils.MEMORY_ALLOCATION_GRANULARITY
                    memory_utils.MEMORY_ALLOCATION_GRANULARITY = 64  # 64MB is better for V100
                    logger.info(f"Reduced DeepSpeed memory allocation granularity from {original_granularity}MB to 64MB")
            
            # Patch ZeRO optimizer if using ZeRO-3
            if os.environ.get('ZERO_STAGE', '0') == '3':
                if hasattr(deepspeed.runtime.zero, 'stage3'):
                    # Apply V100-specific optimizations to ZeRO-3
                    stage3 = deepspeed.runtime.zero.stage3
                    
                    # Set more conservative threshold for V100
                    if hasattr(stage3, 'ZERO_PARAM_PERSISTENCE_THRESHOLD'):
                        stage3.ZERO_PARAM_PERSISTENCE_THRESHOLD = 1e5  # Lower value for V100
                        logger.info("Set more conservative parameter persistence threshold for V100")
            
            logger.info("Applied V100-specific DeepSpeed optimizations")
        except Exception as e:
            logger.warning(f"Failed to apply V100-specific DeepSpeed optimizations: {e}")
            
    def _patch_zero3_for_stability(self):
        """
        Patch ZeRO-3 for better stability, especially for multi-node training.
        Focus on reducing communication overhead and improving reliability.
        """
        try:
            import deepspeed
            import torch
            
            # Only proceed if we're using ZeRO-3
            if os.environ.get('ZERO_STAGE', '0') != '3':
                logger.info("Not using ZeRO-3, skipping ZeRO-3 stability patches")
                return
                
            logger.info("Applying ZeRO-3 stability patches")
            
            # Patch stage3 module if available
            if hasattr(deepspeed.runtime.zero, 'stage3'):
                stage3 = deepspeed.runtime.zero.stage3
                
                # Get the DeepSpeedZeroOptimizer_Stage3 class
                if hasattr(stage3, 'DeepSpeedZeroOptimizer_Stage3'):
                    zero3_class = stage3.DeepSpeedZeroOptimizer_Stage3
                    
                    # Patch the gather_partitioned_parameters function to add error handling
                    if hasattr(zero3_class, 'gather_partitioned_parameters'):
                        original_gather = zero3_class.gather_partitioned_parameters
                        
                        def patched_gather_partitioned_parameters(self, *args, **kwargs):
                            """
                            Add better error handling and memory management to ZeRO-3's parameter gathering.
                            """
                            try:
                                # Perform memory cleanup before gathering
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                                # Call original function
                                return original_gather(self, *args, **kwargs)
                            except Exception as e:
                                logger.error(f"Error in gather_partitioned_parameters: {e}")
                                # Try to recover gracefully
                                try:
                                    # Clear memory
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    
                                    # Retry with more conservative settings
                                    logger.info("Retrying gather operation with more conservative settings")
                                    old_bucket_size = getattr(self, 'reduce_bucket_size', 5e8)
                                    setattr(self, 'reduce_bucket_size', old_bucket_size / 2)
                                    setattr(self, 'allgather_bucket_size', old_bucket_size / 2)
                                    return original_gather(self, *args, **kwargs)
                                except Exception as retry_error:
                                    logger.error(f"Retry also failed: {retry_error}")
                                    raise  # Re-raise the exception
                        
                        # Apply the patch
                        zero3_class.gather_partitioned_parameters = patched_gather_partitioned_parameters
                        logger.info("Patched ZeRO-3's gather_partitioned_parameters for better stability")
            
            logger.info("Applied ZeRO-3 stability patches")
        except Exception as e:
            logger.warning(f"Failed to apply ZeRO-3 stability patches: {e}")
            
    def _patch_for_multinode(self):
        """
        Apply specific patches for multi-node training to improve stability and performance.
        """
        try:
            import deepspeed
            import torch
            
            logger.info("Applying multi-node specific patches")
            
            # Optimize NCCL parameters for multi-node
            os.environ["NCCL_IB_TIMEOUT"] = "22"  # Increase timeout for large transfers
            os.environ["NCCL_MAX_NRINGS"] = "4"   # Limit rings to avoid memory issues
            
            # Patch distributed initialization if needed
            if hasattr(deepspeed, 'init_distributed'):
                original_init_distributed = deepspeed.init_distributed
                
                def patched_init_distributed(dist_backend=None, auto_mpi_discovery=True, distributed_port=29500, verbose=True):
                    """
                    Patched distributed initialization with better error handling for multi-node.
                    """
                    try:
                        # Set timeout for initialization
                        os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"  # 30 minutes
                        
                        # Call original function
                        result = original_init_distributed(dist_backend, auto_mpi_discovery, distributed_port, verbose)
                        
                        # Log success message
                        logger.info("Successfully initialized distributed training")
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error in init_distributed: {e}")
                        
                        # Try to provide more diagnostic information
                        try:
                            import subprocess
                            hostname = subprocess.check_output("hostname", shell=True).decode().strip()
                            ip_output = subprocess.check_output("hostname -I", shell=True).decode().strip()
                            env_vars = {k: v for k, v in os.environ.items() if k.startswith(("MASTER_", "WORLD_", "RANK", "LOCAL_", "NCCL"))}
                            
                            logger.error(f"Diagnostic information from {hostname}:")
                            logger.error(f"IP addresses: {ip_output}")
                            logger.error(f"Environment variables: {env_vars}")
                        except:
                            pass
                        
                        # Re-raise the exception after diagnostics
                        raise
                
                # Apply the patch
                deepspeed.init_distributed = patched_init_distributed
                logger.info("Patched deepspeed.init_distributed for multi-node stability")
            
            logger.info("Applied multi-node specific patches")
        except Exception as e:
            logger.warning(f"Failed to apply multi-node specific patches: {e}")

# Apply patch when the module is imported
apply_patch()

# Avoid applying patches twice automatically
# Let users explicitly create DeepSpeedPatcher() when needed 