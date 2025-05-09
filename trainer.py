import torch
import logging
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler', {}).get('factor', 0.5),
            patience=config.get('scheduler', {}).get('patience', 5),
            min_lr=config.get('scheduler', {}).get('min_lr', 1e-6)
        )
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 100)
        self.gradient_clip_val = config.get('gradient_clip_val', 1.0)
        
        # Initialize best validation loss for model saving
        self.best_val_loss = float('inf')
        
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # Move batch to device and ensure gradients
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    processed_batch[k] = v.to(self.device).float().requires_grad_(True)
                else:
                    processed_batch[k] = v
            
            # Forward pass
            outputs = self.model(processed_batch)
            
            # Ensure output requires gradients
            output_audio = outputs['output_audio'].float().requires_grad_(True)
            target_audio = processed_batch['target_audio'].float().requires_grad_(True)
            
            # Match dimensions
            if output_audio.dim() == 2:
                output_audio = output_audio.unsqueeze(1)
            if target_audio.dim() == 2:
                target_audio = target_audio.unsqueeze(1)
            
            # Calculate loss
            loss = F.mse_loss(output_audio, target_audio)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('gradient_clip_val', 1.0)
            )
            
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise
    def validation_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate validation loss
            target_audio = batch['target_audio'].unsqueeze(1)
            output_audio = outputs['output_audio'].unsqueeze(1)
            val_loss = F.mse_loss(output_audio, target_audio)
            
            return val_loss.item()
    
    def train_epoch(self, epoch):
        epoch_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            epoch_loss += loss
            
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        val_loss = 0
        for batch in self.val_loader:
            loss = self.validation_step(batch)
            val_loss += loss
            
        return val_loss / len(self.val_loader)
    
    def train(self):
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f'best_model.pth')
            
            # Log progress
            logger.info(f'Epoch {epoch+1}/{self.num_epochs}')
            logger.info(f'Train Loss: {train_loss:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}')
            
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], filename))
        
    def cleanup(self):
        """Cleanup any resources"""
        pass
