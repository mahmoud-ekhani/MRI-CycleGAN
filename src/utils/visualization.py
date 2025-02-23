import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def visualize_progress(epoch, step, batch, generated_images, save_dir):
    """Save and display progress images"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row: T1 → T2
    axes[0,0].imshow(batch['T1'][0,0].cpu().numpy().T, cmap='gray', origin='lower')
    axes[0,0].set_title('Original T1')
    axes[0,1].imshow(generated_images['fake_Y'][0,0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[0,1].set_title('Generated T2')
    axes[0,2].imshow(generated_images['rec_X'][0,0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[0,2].set_title('Reconstructed T1')
    
    # Second row: T2 → T1
    axes[1,0].imshow(batch['T2'][0,0].cpu().numpy().T, cmap='gray', origin='lower')
    axes[1,0].set_title('Original T2')
    axes[1,1].imshow(generated_images['fake_X'][0,0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[1,1].set_title('Generated T1')
    axes[1,2].imshow(generated_images['rec_Y'][0,0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[1,2].set_title('Reconstructed T2')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle(f'Epoch {epoch}, Step {step}')
    
    # Save figure
    save_path = os.path.join(save_dir, f'progress_epoch_{epoch}_step_{step}.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_losses(history, save_dir):
    """Plot and save loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['G_losses'], label='Generator')
    plt.plot(history['D_X_losses'], label='Discriminator X')
    plt.plot(history['D_Y_losses'], label='Discriminator Y')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

def save_model_samples(model, epoch, save_dir='results'):
    """Save model generated samples"""
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Generate samples
        fake_Y = model.G_XtoY(model.real_X)
        fake_X = model.G_YtoX(model.real_Y)
        
        # Convert to numpy and rescale
        samples = {
            'real_T1': model.real_X.cpu().numpy()[0, 0],
            'real_T2': model.real_Y.cpu().numpy()[0, 0],
            'fake_T2': fake_Y.cpu().numpy()[0, 0],
            'fake_T1': fake_X.cpu().numpy()[0, 0]
        }
        
        # Save individual images
        for name, img in samples.items():
            plt.imsave(
                os.path.join(save_dir, f'{name}_epoch{epoch}.png'),
                img.T,
                cmap='gray'
            ) 