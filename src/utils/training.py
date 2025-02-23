import torch
import os
from .visualization import plot_training_progress, save_model_samples, visualize_progress, plot_losses
import time
from datetime import datetime

def train_epoch(model, train_loader, epoch, save_freq=100):
    """Train model for one epoch"""
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        real_t1 = batch['T1']
        real_t2 = batch['T2']
        
        # Set input
        model.set_input(real_t1, real_t2)
        
        # Optimize parameters
        model.optimize()
        
        # Save progress
        if batch_idx % save_freq == 0:
            plot_training_progress(model, epoch, batch_idx)
            save_model_samples(model, epoch)

def validate(model, val_loader):
    """Validate model performance"""
    model.eval()
    val_losses = {
        'G_loss': 0, 'D_X_loss': 0, 'D_Y_loss': 0,
        'cycle_loss': 0, 'identity_loss': 0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            real_t1 = batch['T1']
            real_t2 = batch['T2']
            
            model.set_input(real_t1, real_t2)
            model.forward()
            
            # Calculate losses without backprop
            val_losses['G_loss'] += model.loss_G.item()
            val_losses['D_X_loss'] += model.loss_D_X.item()
            val_losses['D_Y_loss'] += model.loss_D_Y.item()
            val_losses['cycle_loss'] += (model.loss_cycle_X.item() + model.loss_cycle_Y.item())
            val_losses['identity_loss'] += (model.loss_idt_X.item() + model.loss_idt_Y.item())
    
    # Average losses
    for k in val_losses:
        val_losses[k] /= len(val_loader)
    
    return val_losses

def save_checkpoint(model, epoch, save_dir='checkpoints'):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'G_XtoY_state_dict': model.G_XtoY.state_dict(),
        'G_YtoX_state_dict': model.G_YtoX.state_dict(),
        'D_X_state_dict': model.D_X.state_dict(),
        'D_Y_state_dict': model.D_Y.state_dict(),
        'opt_G_state_dict': model.opt_G.state_dict(),
        'opt_D_X_state_dict': model.opt_D_X.state_dict(),
        'opt_D_Y_state_dict': model.opt_D_Y.state_dict(),
        'history': model.history
    }
    
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'))

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.G_XtoY.load_state_dict(checkpoint['G_XtoY_state_dict'])
    model.G_YtoX.load_state_dict(checkpoint['G_YtoX_state_dict'])
    model.D_X.load_state_dict(checkpoint['D_X_state_dict'])
    model.D_Y.load_state_dict(checkpoint['D_Y_state_dict'])
    model.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
    model.opt_D_X.load_state_dict(checkpoint['opt_D_X_state_dict'])
    model.opt_D_Y.load_state_dict(checkpoint['opt_D_Y_state_dict'])
    model.history = checkpoint['history']
    
    return checkpoint['epoch']

def train_cyclegan(model, dataloader, num_epochs=5, device="cuda", resume_training=True):
    """
    Train the CycleGAN model with checkpoint saving and visualization
    
    Args:
        model: CycleGAN model instance
        dataloader: DataLoader with training data
        num_epochs: Number of epochs to train
        device: Device to train on
        resume_training: Whether to look for and resume from latest checkpoint
    """
    
    # Create checkpoint directory structure
    base_checkpoint_dir = 'checkpoints'
    current_date = datetime.now().strftime('%Y_%m_%d')
    checkpoint_dir = os.path.join(base_checkpoint_dir, current_date)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create visualization directory
    vis_dir = os.path.join('visualizations', current_date)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize starting epoch and history
    start_epoch = 0
    
    # Look for latest checkpoint if resume_training is True
    if resume_training:
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint)
            model.G_XtoY.load_state_dict(checkpoint['G_XtoY_state'])
            model.G_YtoX.load_state_dict(checkpoint['G_YtoX_state'])
            model.D_X.load_state_dict(checkpoint['D_X_state'])
            model.D_Y.load_state_dict(checkpoint['D_Y_state'])
            model.opt_G.load_state_dict(checkpoint['opt_G_state'])
            model.opt_D_X.load_state_dict(checkpoint['opt_D_X_state'])
            model.opt_D_Y.load_state_dict(checkpoint['opt_D_Y_state'])
            model.scheduler_G.load_state_dict(checkpoint['scheduler_G_state'])
            model.scheduler_D_X.load_state_dict(checkpoint['scheduler_D_X_state'])
            model.scheduler_D_Y.load_state_dict(checkpoint['scheduler_D_Y_state'])
            model.history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    print(f"Starting training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            # Set model input
            model.set_input(batch['T1'], batch['T2'])
            
            # Generate images
            model.forward()
            
            # Update networks
            model.optimize()
            
            # Store losses
            model.history['G_losses'].append(model.loss_G.item())
            model.history['D_X_losses'].append(model.loss_D_X.item())
            model.history['D_Y_losses'].append(model.loss_D_Y.item())
            model.history['cycle_losses'].append((model.loss_cycle_X + model.loss_cycle_Y).item())
            model.history['identity_losses'].append((model.loss_idt_X + model.loss_idt_Y).item())
            
            # Print progress and visualize (every 100 steps)
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{start_epoch + num_epochs-1}] "
                      f"Step [{i}/{len(dataloader)}] "
                      f"G: {model.loss_G.item():.4f} "
                      f"D_X: {model.loss_D_X.item():.4f} "
                      f"D_Y: {model.loss_D_Y.item():.4f} "
                      f"Cyc: {(model.loss_cycle_X + model.loss_cycle_Y).item():.4f} "
                      f"Idt: {(model.loss_idt_X + model.loss_idt_Y).item():.4f}")
                
                # Visualize progress
                generated_images = {
                    'fake_Y': model.fake_Y,
                    'fake_X': model.fake_X,
                    'rec_X': model.rec_X,
                    'rec_Y': model.rec_Y
                }
                visualize_progress(epoch, i, batch, generated_images, vis_dir)
        
        # Update learning rates
        model.scheduler_G.step()
        model.scheduler_D_X.step()
        model.scheduler_D_Y.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'G_XtoY_state': model.G_XtoY.state_dict(),
            'G_YtoX_state': model.G_YtoX.state_dict(),
            'D_X_state': model.D_X.state_dict(),
            'D_Y_state': model.D_Y.state_dict(),
            'opt_G_state': model.opt_G.state_dict(),
            'opt_D_X_state': model.opt_D_X.state_dict(),
            'opt_D_Y_state': model.opt_D_Y.state_dict(),
            'scheduler_G_state': model.scheduler_G.state_dict(),
            'scheduler_D_X_state': model.scheduler_D_X.state_dict(),
            'scheduler_D_Y_state': model.scheduler_D_Y.state_dict(),
            'history': model.history
        }
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        epoch_time = time.time() - epoch_start_time
        print(f'End of epoch {epoch} / {start_epoch + num_epochs-1} \t Time Taken: {epoch_time:.2f} sec')
        
        # Plot and save loss curves
        plot_losses(model.history, vis_dir) 