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
        
        # Add residual connection if input and output channels match
        self.use_residual = in_channels == out_channels
        if not self.use_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        residual = x if self.use_residual else self.residual_conv(x)
        
        h = self.conv1(x)
        h = self.norm1(h)
        h += self.time_mlp(t)[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + residual

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.ln(x)
        attention_value, _ = self.mha(x, x, x)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(1, 2).view(-1, self.channels, *size)

class CrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, context):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        context = context.flatten(2).transpose(1, 2)
        x = self.ln(x)
        attention_value, _ = self.mha(x, context, context)
        attention_value = attention_value + x
        attention_value = self.ff_cross(attention_value) + attention_value
        return attention_value.transpose(1, 2).view(-1, self.channels, *size)

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
            ConvBlock(n_channels, n_channels*2, time_channels),
            SelfAttention(n_channels*2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(n_channels*2, n_channels*4, time_channels),
            SelfAttention(n_channels*4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(n_channels*4, n_channels*8, time_channels),
            SelfAttention(n_channels*8)
        )

        # Bottleneck with attention
        self.bot1 = ConvBlock(n_channels*8, n_channels*8, time_channels)
        self.bot_attn = SelfAttention(n_channels*8)
        self.cross_attn = CrossAttention(n_channels*8)
        self.bot2 = ConvBlock(n_channels*8, n_channels*8, time_channels)
        self.bot3 = ConvBlock(n_channels*8, n_channels*8, time_channels)

        # Decoder with attention
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(n_channels*16, n_channels*4, time_channels),
            SelfAttention(n_channels*4)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(n_channels*8, n_channels*2, time_channels),
            SelfAttention(n_channels*2)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(n_channels*4, n_channels, time_channels)
        )
        
        # Output
        self.outc = nn.Conv2d(n_channels, 1, 1)

    def forward(self, x, t, condition=None, context=None):
        """
        Args:
            x: Input tensor [B, C, H, W] containing source image + noise
            t: Timesteps [B]
            condition: Optional conditioning flag (e.g., 0 for T1→T2, 1 for T2→T1)
            context: Optional context image for cross-attention (paired T1/T2 image)
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
        
        # Bottleneck with attention
        x4 = self.bot1(x4, t)
        x4 = self.bot_attn(x4)
        if context is not None:
            x4 = self.cross_attn(x4, context)
        x4 = self.bot2(x4, t)
        x4 = self.bot3(x4, t)
        
        # Decoder with skip connections
        x = self.up1(torch.cat([x4, x3], dim=1), t)
        x = self.up2(torch.cat([x, x2], dim=1), t)
        x = self.up3(torch.cat([x, x1], dim=1), t)
        
        return self.outc(x) 