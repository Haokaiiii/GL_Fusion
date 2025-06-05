"""
Distributed training manager for GL-Fusion.
Handles setup for single GPU, DDP, and DeepSpeed training.
"""

import os
import socket
import logging
import datetime
from typing import Tuple, Optional, Dict, Any
import torch
import torch.distributed as dist
from pathlib import Path

logger = logging.getLogger(__name__)


class DistributedManager:
    """Manages distributed training setup and environment."""
    
    def __init__(self):
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.is_distributed = False
        self.device = None
        self.backend = None
        
    def setup(self) -> Tuple[int, int, int, torch.device]:
        """
        Setup distributed environment.
        Returns: (rank, local_rank, world_size, device)
        """
        # Check if we're in a distributed environment
        if self._check_distributed_env():
            self.is_distributed = True
            self._setup_distributed()
        else:
            self._setup_single_gpu()
            
        return self.rank, self.local_rank, self.world_size, self.device
    
    def _check_distributed_env(self) -> bool:
        """Check if we're in a distributed environment."""
        # Check for PyTorch distributed env vars
        if all(var in os.environ for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']):
            return True
            
        # Check for OpenMPI env vars
        if all(var in os.environ for var in ['OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE']):
            return True
            
        return False
    
    def _setup_distributed(self) -> None:
        """Setup distributed training environment."""
        # Get distributed parameters
        if 'RANK' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            # OpenMPI environment
            self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            self.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            self.local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
            
            # Set PyTorch env vars
            os.environ['RANK'] = str(self.rank)
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['LOCAL_RANK'] = str(self.local_rank)
        
        # Setup master address and port
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = self._get_master_addr()
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
            
        logger.info(f"Distributed setup: rank={self.rank}, world_size={self.world_size}, "
                    f"local_rank={self.local_rank}")
        
        # Initialize process group
        if not dist.is_initialized():
            timeout = datetime.timedelta(seconds=3600)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=self.rank,
                world_size=self.world_size,
                timeout=timeout
            )
            logger.info("Process group initialized")
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            logger.info(f"Set device to {self.device}")
        else:
            self.device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
            
        self.backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        
    def _setup_single_gpu(self) -> None:
        """Setup for single GPU or CPU training."""
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            logger.info("Single GPU training on cuda:0")
        else:
            self.device = torch.device('cpu')
            logger.info("CPU training")
            
    def _get_master_addr(self) -> str:
        """Get master address for distributed training."""
        # Try to get from PBS nodefile
        if 'PBS_NODEFILE' in os.environ:
            try:
                with open(os.environ['PBS_NODEFILE'], 'r') as f:
                    nodes = list(set(f.read().splitlines()))
                    if nodes:
                        return nodes[0]
            except Exception as e:
                logger.warning(f"Failed to read PBS_NODEFILE: {e}")
                
        # Fallback to hostname
        return socket.gethostname()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    def barrier(self) -> None:
        """Synchronization barrier."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            
    def cleanup(self) -> None:
        """Clean up distributed environment."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")
            
    def get_world_info(self) -> Dict[str, Any]:
        """Get information about the distributed world."""
        return {
            'rank': self.rank,
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'is_distributed': self.is_distributed,
            'device': str(self.device),
            'backend': self.backend
        } 