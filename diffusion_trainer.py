import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

class DDPMTrainer:
    def __init__(
        self, model, n_timesteps=1000, beta_start=1e-4, beta_end=0.02,
        lr=1e-4, device="cuda"
    ):
        """
        Args:
            model: Your UNet model (with in_channels=..., time conditioning, etc.)
            n_timesteps: Number of diffusion timesteps
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value
            lr: Learning rate for Adam
            device: Device to run on ("cuda" or "cpu")
        """
        self.model = model.to(device)
        self.device = device
        self.n_timesteps = n_timesteps
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Define optimizer inside the trainer
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        # For mixed precision
        self.scaler = GradScaler()

    def diffuse_step(self, x_0, t):
        """Forward diffusion step: x_0 -> x_t with noise."""
        noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise
    
    def train_one_batch(self, x_0, condition=None, context=None):
        """
        Runs one forward + backward pass (single batch).
        
        Args:
            x_0: [B, C, H, W] input images
            condition: e.g. 0 or 1 if your model uses a condition channel
            context: optional cross-attention input
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device)
        
        # Forward diffusion
        x_t, noise = self.diffuse_step(x_0, t)

        # Amp autocast
        with autocast():
            noise_pred = self.model(x_t, t, condition=condition, context=context)
            loss = F.mse_loss(noise_pred, noise)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_one_epoch(self, dataloader, condition=None, context=None):
        """Train for one full epoch over the dataset."""
        self.model.train()
        epoch_loss = 0.0
        for x_0 in dataloader:
            x_0 = x_0.to(self.device)
            # If your condition / context are also from dataloader, pass them similarly
            batch_loss = self.train_one_batch(x_0, condition, context)
            epoch_loss += batch_loss
        return epoch_loss / len(dataloader)
    
    @torch.no_grad()
    def sample(self, condition=None, context=None, shape=None, n_steps=None):
        """
        Reverse diffusion sampling.
        
        Args:
            condition: scalar or small tensor for condition channel
            context: cross-attention input
            shape: shape of the sample, e.g. [batch_size, channels, H, W]
            n_steps: number of sampling steps (default: self.n_timesteps)
        """
        if n_steps is None:
            n_steps = self.n_timesteps
        
        x_t = torch.randn(shape, device=self.device)
        
        for t in reversed(range(n_steps)):
            t_batch = torch.ones(shape[0], device=self.device, dtype=torch.long) * t
            noise_pred = self.model(x_t, t_batch, condition=condition, context=context)
            
            alpha_t = self.alphas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0.
            
            x_t = (1 / torch.sqrt(alpha_t)) * (
                x_t - beta_t / torch.sqrt(1 - alpha_t_cumprod) * noise_pred
            ) + torch.sqrt(beta_t) * noise
        
        return x_t
