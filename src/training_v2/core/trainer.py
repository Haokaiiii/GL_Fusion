"""
Trainer class for GL-Fusion model.
Encapsulates training loop, validation, and checkpoint management.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import gc
import numpy as np
import sys

# Add geobleu to path if not already installed in environment
# This assumes geobleu-2023 directory is at the project root
geobleu_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'geobleu-2023')
if geobleu_base_dir not in sys.path:
    sys.path.insert(0, geobleu_base_dir)

# Try to import geobleu, handle if not found
try:
    import geobleu
    GEOBLEU_AVAILABLE = True
except ImportError:
    GEOBLEU_AVAILABLE = False
    logger.warning("geobleu package not found. GeoBLEU and DTW metrics will not be calculated.")

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for GL-Fusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Dict[str, Any],
        distributed_manager: Any,
        device: torch.device,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            config: Configuration dictionary
            distributed_manager: Distributed training manager
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.distributed_manager = distributed_manager
        self.device = device
        
        # Training settings
        self.use_amp = config['training'].get('use_amp', True)
        self.gradient_clip_val = config['training'].get('gradient_clip_val', 1.0)
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        
        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir or config['logging'].get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        start_epoch: int = 0
    ) -> None:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
        """
        torch.autograd.set_detect_anomaly(True)
        self.current_epoch = start_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Log training metrics
            if self.distributed_manager.is_main_process():
                self._log_metrics(train_metrics, epoch, 'train')
                
            # Validation
            if epoch % self.config['training'].get('validation_interval', 1) == 0:
                val_metrics = self.validate(val_loader, epoch)
                
                # Save checkpoint
                if self.distributed_manager.is_main_process():
                    self._log_metrics(val_metrics, epoch, 'val')
                    
                    # Check if best model
                    val_loss = val_metrics.get('loss', float('inf'))
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        
                    # Save checkpoint
                    self.save_checkpoint(
                        epoch=epoch,
                        metrics={'train': train_metrics, 'val': val_metrics},
                        is_best=is_best
                    )
                    
            # Synchronize after each epoch
            self.distributed_manager.barrier()
            
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch}',
            disable=not self.distributed_manager.is_main_process()
        )
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                output = self.model(**batch)
                
                # Extract loss from output dictionary
                if isinstance(output, dict):
                    loss = output.get('loss')
                    if loss is None:
                        raise ValueError("Model output dictionary missing 'loss' key during training")
                else:
                    # Backward compatibility: if model returns tensor directly
                    loss = output
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps
                
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                    
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step (if per-batch)
                if self.scheduler and self.config['training'].get('scheduler_step_per_batch', False):
                    self.scheduler.step()
                    
                self.global_step += 1
                
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if self.distributed_manager.is_main_process():
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
                
            # Memory cleanup
            if step % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        # Scheduler step (if per-epoch)
        if self.scheduler and not self.config['training'].get('scheduler_step_per_batch', False):
            self.scheduler.step()
            
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Set epoch for distributed sampler
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)
            
        all_losses = []
        all_preds = [] # To store all predictions from all batches
        all_targets = [] # To store all ground truths from all batches
        # We need a way to group predictions/targets by original sample/user for GeoBLEU/DTW
        # Assuming batch contains enough info, e.g., 'uid' or a unique identifier per sample
        # and also the d,t for each point if needed. For now, let's assume we can get (x,y) per sample.
        # For GeoBLEU/DTW, we need per-trajectory lists.
        # The structure of data for geobleu is a list of (d,t,x,y) tuples for a trajectory.
        # The TrajectoryDataset.__getitem__ returns a dict that includes 'target_coords' (x,y tensor)
        # and other info. The collate_fn then batches this up.
        # We need to ensure the collate_fn passes through identifiers (e.g. uid, original_sample_index)
        # and the d,t components for each point if we are to reconstruct (d,t,x,y).
        # For now, let's focus on getting (x,y) for each sample. The trainer currently gets 'target_positions'
        # Let's assume 'target_positions' is a batch of (x,y) target coordinates.
        # And model output is also a batch of (x,y) predicted coordinates.

        # Store raw sequence data for GeoBLEU/DTW if needed
        # This requires the batch to contain more info like original day/time for each coord
        # This is a placeholder for how we might collect data for full trajectories
        user_trajectories = {} # uid -> {'pred': [], 'target': []}

        # Progress bar
        pbar = tqdm(
            val_loader,
            desc=f'Validation {epoch}',
            disable=not self.distributed_manager.is_main_process()
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                prepared_batch = self._prepare_batch(batch)
                
                # Extract targets for metric calculation (must be done before filtering for model)
                # Assuming 'target_positions' is the key for ground truth coordinates from collate_fn
                targets = batch.get('target_positions')
                if targets is not None:
                    all_targets.append(targets.cpu()) #.numpy() ? Check format for calculate_metrics later

                # Forward pass
                with autocast(enabled=self.use_amp):
                    # The model should return predicted coordinates and loss
                    # Let's assume model returns a dict: {'loss': ..., 'predictions': ...}
                    # Or, if it returns a tuple (loss, preds) like before, we adapt.
                    output_dict = self.model(**prepared_batch) 
                    
                    loss = output_dict.get('loss')
                    preds = output_dict.get('predictions') # Predicted coordinates
                                            
                if loss is not None:
                    all_losses.append(loss.item())
                
                if preds is not None and targets is not None:
                    all_preds.append(preds.cpu()) #.numpy()?
                    
                    # For GeoBLEU/DTW, reconstruct trajectories.
                    # Note: This requires additional information (uid, day, time) that may not be available
                    # in the current batch structure. Skip for now to avoid errors.
                    if GEOBLEU_AVAILABLE and (self.config['evaluation']['metrics'] and 
                                              ('geo_bleu' in self.config['evaluation']['metrics'] or \
                                               'dtw' in self.config['evaluation']['metrics'])):
                        # Check if required keys exist in batch
                        if all(key in batch for key in ['uid', 'day', 'time']):
                            batch_size = preds.shape[0]
                            for i in range(batch_size):
                                try:
                                    uid = batch['uid'][i] # Assuming 'uid' is in batch
                                    d = batch['day'][i]   # Assuming 'day' is in batch
                                    t = batch['time'][i]  # Assuming 'time' is in batch
                                    
                                    pred_point = (d, t, preds[i, 0].item(), preds[i, 1].item())
                                    target_point = (d, t, targets[i, 0].item(), targets[i, 1].item())
                                    
                                    if uid not in user_trajectories:
                                        user_trajectories[uid] = {'pred': [], 'target': []}
                                    user_trajectories[uid]['pred'].append(pred_point)
                                    user_trajectories[uid]['target'].append(target_point)
                                except (KeyError, IndexError) as e:
                                    logger.warning(f"Error processing batch for GeoBLEU/DTW: {e}")
                                    break
                        else:
                            logger.debug("GeoBLEU/DTW metrics requested but required keys (uid, day, time) not found in batch")

                # Update progress bar
                if self.distributed_manager.is_main_process() and loss is not None:
                    current_avg_loss = np.mean(all_losses) if all_losses else 0
                    pbar.set_postfix({'loss': f'{current_avg_loss:.4f}'})
                    
        # Calculate average loss
        avg_loss = np.mean(all_losses) if all_losses else float('nan')
        metrics = {'loss': avg_loss}
        
        # Calculate MSE and Euclidean Distance if predictions and targets were collected
        if all_preds and all_targets:
            # Concatenate all predictions and targets from batches
            # Ensure they are in the correct format (likely torch tensors)
            # The `calculate_metrics` function expects torch tensors.
            # If they are lists of numpy arrays, convert them.
            
            # Assuming all_preds and all_targets are lists of tensors from each batch
            # We need to cat them into single tensors
            try:
                all_preds_tensor = torch.cat(all_preds, dim=0)
                all_targets_tensor = torch.cat(all_targets, dim=0)
                
                # Standard metrics (MSE, Euclidean)
                # Note: calculate_metrics is from training.utils
                # It needs to be compatible or we call a different one for trainer.
                # For now, assuming it's the same one used by evaluate.py (imported)
                # from training.utils import calculate_metrics - this is not in this file.
                # Let's define it or import it here. For now, let's assume it can be called.
                # We will use the same logic as in evaluate.py for these.

                mse_val = F.mse_loss(all_preds_tensor, all_targets_tensor).item()
                euc_dist_val = torch.mean(torch.sqrt(torch.sum((all_preds_tensor - all_targets_tensor) ** 2, dim=1))).item()
                metrics['mse'] = mse_val
                metrics['euclidean_dist'] = euc_dist_val

            except Exception as e:
                logger.error(f"Error calculating standard metrics (MSE, Euclidean): {e}")
                metrics['mse'] = float('nan')
                metrics['euclidean_dist'] = float('nan')

        # Calculate GeoBLEU and DTW
        if GEOBLEU_AVAILABLE and user_trajectories:
            user_geobleu_scores = []
            user_dtw_scores = []
            for uid, trajectories in user_trajectories.items():
                pred_traj = trajectories['pred']
                target_traj = trajectories['target']
                if pred_traj and target_traj:
                    try:
                        if 'geo_bleu' in self.config['evaluation']['metrics']:
                             if len(pred_traj) >= 1 and len(target_traj) >=1: # Min length for geobleu calc
                                gb_score = geobleu.calc_geobleu_single(pred_traj, target_traj)
                                user_geobleu_scores.append(gb_score)
                        
                        if 'dtw' in self.config['evaluation']['metrics']:
                            dtw_score = geobleu.calc_dtw_single(pred_traj, target_traj)
                            user_dtw_scores.append(dtw_score)
                    except Exception as e:
                        logger.warning(f"Could not calculate GeoBLEU/DTW for UID {uid} in trainer. Error: {e}")
            
            if 'geo_bleu' in self.config['evaluation']['metrics']:
                metrics['geo_bleu'] = np.mean(user_geobleu_scores) if user_geobleu_scores else float('nan')
            if 'dtw' in self.config['evaluation']['metrics']:
                metrics['dtw'] = np.mean(user_dtw_scores) if user_dtw_scores else float('nan')
        else:
            if 'geo_bleu' in self.config['evaluation']['metrics']:
                 metrics['geo_bleu'] = float('nan')
            if 'dtw' in self.config['evaluation']['metrics']:
                 metrics['dtw'] = float('nan')

        return metrics
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device and filter for model."""
        prepared_batch = {}
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared_batch[k] = v.to(self.device)
            else:
                prepared_batch[k] = v
                
        # Filter batch for model's expected inputs
        return self._filter_batch_for_model(prepared_batch)
    
    def _filter_batch_for_model(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Filter batch to only include keys expected by model."""
        import inspect
        
        # Get the forward method
        if hasattr(self.model, 'module'):
            forward_method = self.model.module.forward
        else:
            forward_method = self.model.forward
            
        # Get expected parameters
        try:
            sig = inspect.signature(forward_method)
            expected_keys = list(sig.parameters.keys())
            
            # Filter batch
            filtered_batch = {k: v for k, v in batch.items() if k in expected_keys}
            
            return filtered_batch
        except:
            # If inspection fails, return original batch
            return batch
            
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False
    ) -> None:
        """Save a checkpoint."""
        # Only save on main process
        if not self.distributed_manager.is_main_process():
            return
            
        # Get model state dict
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Starting epoch
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load scaler state
        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        # Load other states
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint['epoch'] + 1
        
        logger.info(f"Resumed from epoch {start_epoch}")
        return start_epoch
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str) -> None:
        """Log metrics to console."""
        message = f"Epoch {epoch} {phase.capitalize()} - "
        message += " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(message) 