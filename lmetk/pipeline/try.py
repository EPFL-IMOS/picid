"""
Trial1Pipeline implementation following BasePipelineStrategy interface.

This pipeline implements sub-channel training for transformer models with
improved code organization, better separation of concerns, and enhanced
maintainability compared to the original Exp_Sub_Channel_Training.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from omegaconf import DictConfig

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Trial1Pipeline:
    """
    Clean implementation of sub-channel training pipeline strategy.
    
    This class reorganizes the original Exp_Sub_Channel_Training functionality
    into a more modular, maintainable structure with clear separation of concerns.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        datamodule: Any = None,
        model: Any = None,
        evaluator: Any = None,
        logger: Any = None,
        optimizer: Any = None,
        loss: Any = None
    ) -> None:
        """Initialize the Trial1Pipeline with configuration and components."""
        self.cfg = cfg
        self.args = cfg  # For compatibility with original code
        self.datamodule = datamodule
        self.model = model
        self.evaluator = evaluator
        self.logger = logger
        self.optimizer = optimizer
        self.loss = loss
        
        # Pipeline state
        self._is_setup = False
        self.device = self._get_device()
        
        # Training components (initialized in setup)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model_optim = None
        self.criterion = None
        self.scaler = None
        
        # Results tracking
        self.record_test_shape = None
    
    def _get_device(self) -> torch.device:
        """Determine the appropriate device for training."""
        if hasattr(self.args, 'use_gpu') and self.args.use_gpu:
            if torch.cuda.is_available():
                device_id = self.args.gpu if hasattr(self.args, 'gpu') else 0
                return torch.device(f'cuda:{device_id}')
        return torch.device('cpu')
    
    def setup(self) -> None:
        """Setup all pipeline components for training and evaluation."""
        if self._is_setup:
            return
            
        print("Setting up Trial1Pipeline...")
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Setup model
        self._setup_model()
        
        # Setup training components
        self._setup_training_components()
        
        self._is_setup = True
        print("Pipeline setup completed successfully!")
    
    def _setup_data_loaders(self) -> None:
        """Initialize data loaders for train, validation, and test."""
        print("Setting up data loaders...")
        
        train_data, self.train_loader = data_provider(self.args, 'train')
        val_data, self.val_loader = data_provider(self.args, 'val')
        test_data, self.test_loader = data_provider(self.args, 'test')
        
        # Store data objects for potential inverse transforms
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
    
    def _setup_model(self) -> None:
        """Initialize and configure the model."""
        if self.model is None:
            # Build model from configuration if not provided
            from experiments.exp_basic import Exp_Basic
            exp_basic = Exp_Basic(self.args)
            self.model = exp_basic._build_model()
        
        self.model = self.model.float().to(self.device)
        
        # Setup multi-GPU if configured
        if (hasattr(self.args, 'use_multi_gpu') and self.args.use_multi_gpu and 
            hasattr(self.args, 'use_gpu') and self.args.use_gpu):
            if hasattr(self.args, 'device_ids'):
                self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
    
    def _setup_training_components(self) -> None:
        """Initialize optimizer, loss function, and other training components."""
        # Setup optimizer
        self.model_optim = self._create_optimizer()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup mixed precision scaler if enabled
        if hasattr(self.args, 'use_amp') and self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = getattr(self.args, 'optimizer', 'adam').lower()
        lr = getattr(self.args, 'learning_rate', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f'Invalid optimizer: {optimizer_name}')
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        return nn.MSELoss()
    
    def _process_batch_with_subchannels(
        self, 
        batch_x: torch.Tensor, 
        batch_y: torch.Tensor,
        batch_x_mark: Optional[torch.Tensor] = None,
        batch_y_mark: Optional[torch.Tensor] = None,
        is_training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch using sub-channel approach.
        
        This is the core logic that handles splitting data into sub-channels
        and processing them iteratively.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        
        if not is_training:
            batch_y = batch_y.to(self.device)
        
        channel_size = batch_x.size(2)
        max_channel_size = getattr(self.args, 'max_channel', channel_size)
        number_of_iter = channel_size // max_channel_size + 1
        
        pred = None
        true = None
        total_loss = 0.0
        
        for j in range(number_of_iter):
            # Extract sub-channel data
            start_idx = j * max_channel_size
            end_idx = (j + 1) * max_channel_size
            
            batch_x_subset = batch_x[:, :, start_idx:end_idx]
            batch_y_subset = batch_y[:, :, start_idx:end_idx]
            
            # Handle time marks for specific datasets
            if self._should_skip_time_marks():
                batch_x_mark_subset = None
                batch_y_mark_subset = None
            else:
                batch_x_mark_subset = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                batch_y_mark_subset = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None
            
            # Process subset
            if is_training:
                subset_loss = self._train_subset(
                    batch_x_subset, batch_y_subset, 
                    batch_x_mark_subset, batch_y_mark_subset
                )
                total_loss += subset_loss
            else:
                pred_subset, true_subset = self._predict_subset(
                    batch_x_subset, batch_y_subset,
                    batch_x_mark_subset, batch_y_mark_subset
                )
                
                # Concatenate results
                if pred is None:
                    pred = pred_subset
                    true = true_subset
                else:
                    if isinstance(pred_subset, torch.Tensor):
                        pred = torch.cat((pred, pred_subset), dim=2)
                        true = torch.cat((true, true_subset), dim=2)
                    else:
                        pred = np.concatenate((pred, pred_subset), axis=2)
                        true = np.concatenate((true, true_subset), axis=2)
        
        if is_training:
            return total_loss, None
        else:
            return pred, true
    
    def _should_skip_time_marks(self) -> bool:
        """Check if time marks should be skipped for specific datasets."""
        data_name = getattr(self.args, 'data', '')
        return 'PEMS' in data_name or 'Solar' in data_name
    
    def _train_subset(
        self,
        batch_x_subset: torch.Tensor,
        batch_y_subset: torch.Tensor,
        batch_x_mark_subset: Optional[torch.Tensor],
        batch_y_mark_subset: Optional[torch.Tensor]
    ) -> float:
        """Train on a single subset and return loss."""
        # Prepare decoder input
        dec_inp = self._prepare_decoder_input(batch_y_subset)
        
        # Forward pass
        if hasattr(self.args, 'use_amp') and self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self._model_forward(
                    batch_x_subset, batch_x_mark_subset, 
                    dec_inp, batch_y_mark_subset
                )
                outputs, batch_y_subset = self._process_outputs(outputs, batch_y_subset)
                loss = self.criterion(outputs, batch_y_subset)
        else:
            outputs = self._model_forward(
                batch_x_subset, batch_x_mark_subset,
                dec_inp, batch_y_mark_subset
            )
            outputs, batch_y_subset = self._process_outputs(outputs, batch_y_subset)
            loss = self.criterion(outputs, batch_y_subset)
        
        return loss.item()
    
    def _predict_subset(
        self,
        batch_x_subset: torch.Tensor,
        batch_y_subset: torch.Tensor,
        batch_x_mark_subset: Optional[torch.Tensor],
        batch_y_mark_subset: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions for a single subset."""
        # Prepare decoder input
        dec_inp = self._prepare_decoder_input(batch_y_subset)
        
        # Forward pass
        if hasattr(self.args, 'use_amp') and self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self._model_forward(
                    batch_x_subset, batch_x_mark_subset,
                    dec_inp, batch_y_mark_subset
                )
        else:
            outputs = self._model_forward(
                batch_x_subset, batch_x_mark_subset,
                dec_inp, batch_y_mark_subset
            )
        
        # Process outputs
        f_dim = -1 if getattr(self.args, 'features', '') == 'MS' else 0
        pred_len = getattr(self.args, 'pred_len', 96)
        
        outputs = outputs[:, -pred_len:, f_dim:]
        batch_y_subset = batch_y_subset[:, -pred_len:, f_dim:].to(self.device)
        
        return outputs.detach().cpu(), batch_y_subset.detach().cpu()
    
    def _prepare_decoder_input(self, batch_y: torch.Tensor) -> torch.Tensor:
        """Prepare decoder input tensor."""
        pred_len = getattr(self.args, 'pred_len', 96)
        label_len = getattr(self.args, 'label_len', 48)
        
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(self.device)
        return dec_inp
    
    def _model_forward(
        self,
        batch_x: torch.Tensor,
        batch_x_mark: Optional[torch.Tensor],
        dec_inp: torch.Tensor,
        batch_y_mark: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Execute model forward pass."""
        if hasattr(self.args, 'output_attention') and self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    def _process_outputs(
        self, 
        outputs: torch.Tensor, 
        batch_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process model outputs based on configuration."""
        f_dim = -1 if getattr(self.args, 'features', '') == 'MS' else 0
        pred_len = getattr(self.args, 'pred_len', 96)
        
        # Handle subset prediction if configured
        if hasattr(self.args, 'subset_prediction') and self.args.subset_prediction != -1:
            subset_pred = self.args.subset_prediction
            outputs = outputs[:, -pred_len:-pred_len + subset_pred, f_dim:]
            batch_y = batch_y[:, -pred_len:-pred_len + subset_pred, f_dim:].to(self.device)
        else:
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
        
        return outputs, batch_y
    
    def train(self) -> Dict[str, Any]:
        """Execute the training phase."""
        self._ensure_setup()
        
        print("Starting training...")
        
        # Setup checkpoint directory
        setting = getattr(self.args, 'setting', 'default_setting')
        checkpoint_path = Path(getattr(self.args, 'checkpoints', './checkpoints')) / setting
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Load pre-trained model if specified
        if hasattr(self.args, 'load') and self.args.load:
            self._load_pretrained_model()
            return self._evaluate_loaded_model()
        
        # Training setup
        train_epochs = getattr(self.args, 'train_epochs', 10)
        patience = getattr(self.args, 'patience', 3)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        training_results = []
        
        for epoch in range(train_epochs):
            epoch_start = time.time()
            
            # Training epoch
            train_loss = self._train_epoch(epoch)
            
            # Validation
            val_metrics = self._validate()
            test_metrics = self._test_epoch()
            
            # Logging
            self._log_epoch_results(epoch, train_loss, val_metrics, test_metrics)
            
            # Early stopping check
            early_stopping(val_metrics['loss'], self.model, str(checkpoint_path))
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            # Learning rate adjustment
            adjust_learning_rate(self.model_optim, None, epoch + 1, self.args)
            
            # Store results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'epoch_time': time.time() - epoch_start,
                **val_metrics,
                **{f'test_{k}': v for k, v in test_metrics.items()}
            }
            training_results.append(epoch_results)
        
        # Load best model
        best_model_path = checkpoint_path / 'checkpoint.pth'
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
        
        return {
            'training_results': training_results,
            'final_model': self.model,
            'checkpoint_path': str(checkpoint_path)
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """Execute one training epoch."""
        self.model.train()
        train_losses = []
        iter_count = 0
        time_now = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            iter_count += 1
            self.model_optim.zero_grad()
            
            # Process batch with sub-channels
            total_loss, _ = self._process_batch_with_subchannels(
                batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=True
            )
            
            # Create tensor for backpropagation
            loss_tensor = torch.tensor(total_loss, requires_grad=True)
            train_losses.append(total_loss)
            
            # Backward pass
            if hasattr(self.args, 'use_amp') and self.args.use_amp and self.scaler:
                self.scaler.scale(loss_tensor).backward()
                self.scaler.step(self.model_optim)
                self.scaler.update()
            else:
                loss_tensor.backward()
                self.model_optim.step()
            
            # Progress logging
            if (i + 1) % 100 == 0:
                self._log_training_progress(i, epoch, total_loss, time_now, iter_count)
                iter_count = 0
                time_now = time.time()
        
        return np.mean(train_losses)
    
    def _validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        return self._evaluate_on_loader(self.val_loader, self.val_data, 'validation')
    
    def _test_epoch(self) -> Dict[str, float]:
        """Run test evaluation during training."""
        return self._evaluate_on_loader(self.test_loader, self.test_data, 'test')
    
    def _evaluate_on_loader(self, data_loader, data_obj, phase: str) -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_losses = []
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                pred, true = self._process_batch_with_subchannels(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=False
                )
                
                loss = self.criterion(pred, true)
                total_losses.append(loss.item())
                all_preds.append(pred.numpy())
                all_trues.append(true.numpy())
        
        # Calculate metrics
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        return {
            'loss': np.mean(total_losses),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }
    
    def test(self) -> Dict[str, Any]:
        """Execute comprehensive testing phase."""
        self._ensure_setup()
        
        setting = getattr(self.args, 'setting', 'default_setting')
        
        # Load model if specified
        if hasattr(self.args, 'load') and self.args.load:
            load_path = getattr(self.args, 'load_path', '')
            self.model.load_state_dict(torch.load(load_path))
            print(f'Model loaded from {load_path}')
        else:
            checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
                print(f'Model loaded from {checkpoint_path}')
        
        # Run comprehensive evaluation
        test_results = self._comprehensive_test()
        
        # Save results
        self._save_test_results(test_results, setting)
        
        return test_results
    
    def _comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive testing with visualization."""
        self.model.eval()
        all_preds = []
        all_trues = []
        
        setting = getattr(self.args, 'setting', 'default_setting')
        folder_path = Path(f'./test_results/{setting}')
        folder_path.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                pred, true = self._process_batch_with_subchannels(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=False
                )
                
                # Convert to numpy for processing
                pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred
                true_np = true.detach().cpu().numpy() if isinstance(true, torch.Tensor) else true
                
                # Apply inverse transform if needed
                if (hasattr(self.test_data, 'scale') and self.test_data.scale and 
                    hasattr(self.args, 'inverse') and self.args.inverse):
                    pred_np = self._apply_inverse_transform(pred_np, self.test_data)
                    true_np = self._apply_inverse_transform(true_np, self.test_data)
                
                all_preds.append(pred_np)
                all_trues.append(true_np)
                
                # Generate visualizations periodically
                if i % 20 == 0:
                    self._generate_visualization(
                        batch_x, pred_np, true_np, folder_path / f'{i}.pdf'
                    )
        
        # Process final results
        preds = np.array(all_preds)
        trues = np.array(all_trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        self.record_test_shape = preds.shape
        
        # Calculate final metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        return {
            'predictions': preds,
            'ground_truth': trues,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'mspe': mspe
            },
            'test_shape': preds.shape
        }
    
    def predict(self, data_loader: Optional[Any] = None) -> Any:
        """Generate predictions using the trained model."""
        self._ensure_setup()
        
        if data_loader is None:
            pred_data, pred_loader = data_provider(self.args, 'pred')
        else:
            pred_loader = data_loader
            pred_data = None
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                if not self._should_skip_time_marks():
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    batch_x_mark = batch_y_mark = None
                
                # Prepare decoder input
                dec_inp = self._prepare_decoder_input(batch_y)
                
                # Model forward pass
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                
                # Apply inverse transform if needed
                if (pred_data and hasattr(pred_data, 'scale') and pred_data.scale and 
                    hasattr(self.args, 'inverse') and self.args.inverse):
                    outputs = self._apply_inverse_transform(outputs, pred_data)
                
                predictions.append(outputs)
        
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, predictions.shape[-2], predictions.shape[-1])
        
        return predictions
    
    def evaluate(self, data_loader: Optional[Any] = None) -> Dict[str, float]:
        """Evaluate model performance on given data loader."""
        self._ensure_setup()
        
        if data_loader is None:
            data_loader = self.test_loader
            data_obj = self.test_data
        else:
            data_obj = None
        
        return self._evaluate_on_loader(data_loader, data_obj, 'evaluation')
    
    def save(self, checkpoint_path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model_optim.state_dict() if self.model_optim else None,
            'config': self.cfg,
            'test_shape': self.record_test_shape
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.model_optim and 'optimizer_state_dict' in checkpoint:
            self.model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'test_shape' in checkpoint:
            self.record_test_shape = checkpoint['test_shape']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    # Helper methods
    def _ensure_setup(self) -> None:
        """Ensure pipeline is set up before execution."""
        if not self._is_setup:
            raise RuntimeError("Pipeline must be set up before execution. Call setup() first.")
    
    def _load_pretrained_model(self) -> None:
        """Load pre-trained model."""
        load_path = getattr(self.args, 'load_path', '')
        if load_path and os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Pre-trained model loaded from {load_path}")
    
    def _evaluate_loaded_model(self) -> Dict[str, Any]:
        """Evaluate loaded model without training."""
        val_metrics = self._validate()
        test_metrics = self._test_epoch()
        
        print(f"Loaded model - Validation Loss: {val_metrics['loss']:.7f}, Test Loss: {test_metrics['loss']:.7f}")
        
        # Save zero-shot results
        setting = getattr(self.args, 'setting', 'default_setting')
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(f"zero-shot-{setting}\n")
            f.write(f"mse:{test_metrics['mse']}, mae:{test_metrics['mae']}\n\n")
        
        return {'validation': val_metrics, 'test': test_metrics}
    
    def _log_epoch_results(self, epoch: int, train_loss: float, val_metrics: Dict, test_metrics: Dict) -> None:
        """Log results for an epoch."""
        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | "
              f"Val Loss: {val_metrics['loss']:.7f} | Test Loss: {test_metrics['loss']:.7f}")
        
        # Log to external logger if available (e.g., wandb)
        if hasattr(self, 'external_logger'):
            log_dict = {
                'train_loss': train_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                **{f'test_{k}': v for k, v in test_metrics.items()}
            }
            self.external_logger.log(log_dict)
    
    def _log_training_progress(self, iteration: int, epoch: int, loss: float, start_time: float, iter_count: int) -> None:
        """Log training progress."""
        print(f"\titers: {iteration + 1}, epoch: {epoch + 1} | loss: {loss:.7f}")
        speed = (time.time() - start_time) / iter_count
        remaining_iters = (getattr(self.args, 'train_epochs', 10) - epoch) * len(self.train_loader) - iteration
        left_time = speed * remaining_iters
        print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
    
    def _apply_inverse_transform(self, data: np.ndarray, data_obj: Any) -> np.ndarray:
        """Apply inverse transformation to denormalize data."""
        if hasattr(data_obj, 'inverse_transform'):
            shape = data.shape
            return data_obj.inverse_transform(data.squeeze(0)).reshape(shape)
        return data
    
    def _generate_visualization(
        self, 
        batch_x: torch.Tensor, 
        pred: np.ndarray, 
        true: np.ndarray, 
        save_path: Path
    ) -> None:
        """Generate visualization comparing predictions with ground truth."""
        try:
            input_data = batch_x.detach().cpu().numpy()
            
            # Apply inverse transform to input if needed
            if (hasattr(self.test_data, 'scale') and self.test_data.scale and 
                hasattr(self.args, 'inverse') and self.args.inverse):
                input_shape = input_data.shape
                input_data = self._apply_inverse_transform(input_data, self.test_data)
            
            # Create visualization data
            gt = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
            pd = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)
            
            # Generate plot
            visual(gt, pd, str(save_path))
        except Exception as e:
            print(f"Visualization generation failed: {e}")
    
    def _save_test_results(self, test_results: Dict[str, Any], setting: str) -> None:
        """Save test results to files."""
        # Save metrics to text file
        metrics = test_results['metrics']
        print(f'Test Results - MSE: {metrics["mse"]}, MAE: {metrics["mae"]}')
        
        with open("result_long_term_forecast.txt", 'a') as f:
            prefix = "fine-tuned-" if (hasattr(self.args, 'load') and self.args.load) else ""
            f.write(f"{prefix}{setting}\n")
            f.write(f'mse:{metrics["mse"]}, mae:{metrics["mae"]}\n\n')
        
        # Optionally save detailed results
        if hasattr(self.args, 'save_detailed_results') and self.args.save_detailed_results:
            results_folder = Path(f'./results/{setting}')
            results_folder.mkdir(parents=True, exist_ok=True)
            
            np.save(results_folder / 'metrics.npy', 
                   np.array([metrics['mae'], metrics['mse'], metrics['rmse'], 
                           metrics['mape'], metrics['mspe']]))
            np.save(results_folder / 'pred.npy', test_results['predictions'])
            np.save(results_folder / 'true.npy', test_results['ground_truth'])
    
    # Properties for compatibility and introspection
    @property
    def experiment_name(self) -> str:
        """Get experiment name from configuration."""
        return getattr(self.cfg, 'exp_name', 'trial1_experiment')
    
    @property
    def is_setup(self) -> bool:
        """Check if pipeline is set up."""
        return self._is_setup
    
    @property
    def lags(self) -> int:
        """Get sequence length (for compatibility)."""
        return getattr(self.args, 'seq_len', 96)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model and training state."""
        if not self.model:
            return {'status': 'No model loaded'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_setup': self.is_setup,
            'test_shape': self.record_test_shape
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get current training configuration."""
        return {
            'optimizer': getattr(self.args, 'optimizer', 'adam'),
            'learning_rate': getattr(self.args, 'learning_rate', 1e-4),
            'batch_size': getattr(self.args, 'batch_size', 32),
            'train_epochs': getattr(self.args, 'train_epochs', 10),
            'patience': getattr(self.args, 'patience', 3),
            'max_channel': getattr(self.args, 'max_channel', None),
            'pred_len': getattr(self.args, 'pred_len', 96),
            'seq_len': getattr(self.args, 'seq_len', 96),
            'use_amp': getattr(self.args, 'use_amp', False),
            'use_multi_gpu': getattr(self.args, 'use_multi_gpu', False)
        }
    
    def reset_training_state(self) -> None:
        """Reset training state while keeping model and data."""
        if self.model_optim:
            # Reset optimizer state
            self.model_optim = self._create_optimizer()
        
        if hasattr(self.args, 'use_amp') and self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print("Training state reset successfully")
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        status = "configured" if self._is_setup else "not configured"
        return f"Trial1Pipeline(experiment='{self.experiment_name}', status='{status}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Trial1Pipeline("
            f"experiment='{self.experiment_name}', "
            f"setup={self.is_setup}, "
            f"lags={self.lags}, "
            f"device='{self.device}')"
        )