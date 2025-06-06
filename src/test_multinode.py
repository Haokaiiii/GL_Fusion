#!/usr/bin/env python3
"""
Simple multi-node PyTorch test using system modules
Tests basic MPI, PyTorch, and CUDA functionality
"""

import os
import sys
import socket
import subprocess

def main():
    print(f"=== Node Test on {socket.gethostname()} ===")
    
    # Test MPI environment
    try:
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
        size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
        print(f"MPI Rank {rank}/{size}, Local rank {local_rank}")
    except:
        print("MPI environment variables not found")
        rank = 0
        size = 1
        local_rank = 0
    
    # Test Python environment
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"GPU count: {gpu_count}")
            
            # Test GPU access
            if gpu_count > 0:
                for i in range(gpu_count):
                    try:
                        name = torch.cuda.get_device_name(i)
                        print(f"GPU {i}: {name}")
                    except:
                        print(f"GPU {i}: Error accessing device")
                
                # Simple tensor test
                try:
                    device = torch.device(f'cuda:{local_rank % gpu_count}')
                    x = torch.randn(10, 10, device=device)
                    y = torch.randn(10, 10, device=device)
                    z = torch.mm(x, y)
                    print(f"✅ GPU {device} tensor operations work")
                except Exception as e:
                    print(f"❌ GPU tensor test failed: {e}")
        else:
            print("No CUDA GPUs available")
            
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return 1
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return 1
    
    # Test basic MPI communication if available
    try:
        # Try a simple all-reduce operation
        if size > 1:
            import torch.distributed as dist
            
            # Initialize process group with NCCL backend
            dist.init_process_group(
                backend='nccl' if cuda_available else 'gloo',
                init_method='env://',
                rank=rank,
                world_size=size
            )
            
            # Simple tensor for communication test
            if cuda_available and gpu_count > 0:
                device = torch.device(f'cuda:{local_rank % gpu_count}')
                tensor = torch.ones(1, device=device) * rank
            else:
                device = torch.device('cpu')
                tensor = torch.ones(1) * rank
            
            print(f"Rank {rank}: Initial tensor = {tensor.item()}")
            
            # All-reduce test
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected_sum = sum(range(size))
            
            print(f"Rank {rank}: After all-reduce = {tensor.item()}, expected = {expected_sum}")
            
            if abs(tensor.item() - expected_sum) < 1e-6:
                print(f"✅ Rank {rank}: MPI communication test passed")
            else:
                print(f"❌ Rank {rank}: MPI communication test failed")
            
            dist.destroy_process_group()
        else:
            print("Single process - skipping MPI tests")
            
    except Exception as e:
        print(f"❌ MPI test failed: {e}")
        # Don't return error for MPI failure in this basic test
    
    print(f"=== Test completed on {socket.gethostname()} rank {rank} ===")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 