import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.time_proj = nn.Sequential(
            nn.Linear(1, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels)
        )

    def forward(self, t):
        # t: (batch_size,)
        t = t.unsqueeze(-1).float()
        return self.time_proj(t)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_channels, out_channels)

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h += self.time_mlp(t)[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=3, time_channels=256, n_channels=64):
        """
        Args:
            in_channels: 3 for conditional generation (T1/T2 + noise + condition_flag)
            time_channels: Dimension of time embedding
            n_channels: Base number of channels
        """
        super().__init__()
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_channels)
        
        # Encoder
        self.inc = ConvBlock(in_channels, n_channels, time_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(n_channels, n_channels*2, time_channels)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(n_channels*2, n_channels*4, time_channels)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(n_channels*4, n_channels*8, time_channels)
        )

        # Bottleneck
        self.bot1 = ConvBlock(n_channels*8, n_channels*8, time_channels)
        self.bot2 = ConvBlock(n_channels*8, n_channels*8, time_channels)
        self.bot3 = ConvBlock(n_channels*8, n_channels*8, time_channels)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(n_channels*16, n_channels*4, time_channels)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(n_channels*8, n_channels*2, time_channels)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(n_channels*4, n_channels, time_channels)
        )
        
        # Output
        self.outc = nn.Conv2d(n_channels, 1, 1)

    def forward(self, x, t, condition=None):
        """
        Args:
            x: Input tensor [B, C, H, W] containing source image + noise
            t: Timesteps [B]
            condition: Optional conditioning flag (e.g., 0 for T1→T2, 1 for T2→T1)
        """
        # Add conditioning information
        if condition is not None:
            condition = condition.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
            x = torch.cat([x, condition], dim=1)
            
        t = self.time_embed(t)
        
        # Encoder
        x1 = self.inc(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        
        # Bottleneck
        x4 = self.bot1(x4, t)
        x4 = self.bot2(x4, t)
        x4 = self.bot3(x4, t)
        
        # Decoder with skip connections
        x = self.up1(torch.cat([x4, x3], dim=1), t)
        x = self.up2(torch.cat([x, x2], dim=1), t)
        x = self.up3(torch.cat([x, x1], dim=1), t)
        
        return self.outc(x) 