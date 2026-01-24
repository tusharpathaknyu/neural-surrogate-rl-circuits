"""
Forward Surrogate Model: Circuit Parameters → Waveform
Uses 1D-CNN to generate full output waveform from component values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WaveformEncoder(nn.Module):
    """Encode a waveform into a latent representation (for inverse problem)."""
    
    def __init__(self, waveform_length=512, latent_dim=128):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, waveform_length)
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(8),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, waveform):
        # waveform: (batch, waveform_length)
        x = waveform.unsqueeze(1)  # (batch, 1, waveform_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class WaveformDecoder(nn.Module):
    """Decode latent representation to waveform."""
    
    def __init__(self, latent_dim=128, waveform_length=512):
        super().__init__()
        self.waveform_length = waveform_length
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 8)
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 8)
        x = self.deconv_layers(x)
        # Interpolate to exact waveform length
        x = F.interpolate(x, size=self.waveform_length, mode='linear', align_corners=False)
        return x.squeeze(1)


class ForwardSurrogate(nn.Module):
    """
    Forward model: Circuit parameters → Waveform
    
    Input: [L, C, R_load, V_in, f_sw, duty] (6 parameters)
    Output: Voltage waveform (512 points)
    """
    
    def __init__(self, num_params=6, waveform_length=512, hidden_dim=256):
        super().__init__()
        self.num_params = num_params
        self.waveform_length = waveform_length
        
        # Parameter encoder - maps circuit params to latent space
        self.param_encoder = nn.Sequential(
            nn.Linear(num_params, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Initial projection for waveform generation
        self.init_proj = nn.Linear(hidden_dim, 256 * 16)
        
        # 1D Transposed convolutions to generate waveform
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )
        
        # Scaling layers to help with voltage magnitudes
        self.output_scale = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )
    
    def normalize_params(self, params):
        """Normalize parameters to roughly [-1, 1] range."""
        # Expected ranges (for reference):
        # L: 10e-6 to 100e-6 → log scale
        # C: 47e-6 to 470e-6 → log scale
        # R_load: 2 to 50
        # V_in: 10 to 24
        # f_sw: 50e3 to 500e3 → log scale
        # duty: 0.2 to 0.8
        
        normalized = torch.zeros_like(params)
        
        # Log-scale normalization for L, C, f_sw
        normalized[:, 0] = (torch.log10(params[:, 0]) + 5) / 2  # L
        normalized[:, 1] = (torch.log10(params[:, 1]) + 4) / 2  # C
        normalized[:, 4] = (torch.log10(params[:, 4]) - 4) / 2  # f_sw
        
        # Linear normalization for others
        normalized[:, 2] = (params[:, 2] - 25) / 25  # R_load
        normalized[:, 3] = (params[:, 3] - 17) / 7   # V_in
        normalized[:, 5] = (params[:, 5] - 0.5) / 0.3  # duty
        
        return normalized
    
    def forward(self, params, normalize=True):
        """
        Forward pass.
        
        Args:
            params: (batch, 6) circuit parameters [L, C, R_load, V_in, f_sw, duty]
            normalize: whether to normalize input parameters
            
        Returns:
            waveform: (batch, waveform_length) predicted voltage waveform
        """
        if normalize:
            params = self.normalize_params(params)
        
        # Encode parameters
        h = self.param_encoder(params)
        
        # Project to initial feature map
        x = self.init_proj(h)
        x = x.view(x.size(0), 256, 16)
        
        # Decode to waveform
        x = self.decoder(x)
        
        # Interpolate to exact length
        x = F.interpolate(x, size=self.waveform_length, mode='linear', align_corners=False)
        
        # Final scaling
        x = self.output_scale(x)
        
        return x.squeeze(1)


class ForwardSurrogateWithMetrics(ForwardSurrogate):
    """Extended model that also predicts scalar metrics."""
    
    def __init__(self, num_params=6, waveform_length=512, hidden_dim=256, num_metrics=5):
        super().__init__(num_params, waveform_length, hidden_dim)
        
        self.metrics_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_metrics)
        )
        # Metrics: [V_avg, V_ripple, I_avg, overshoot, P_out]
    
    def forward(self, params, normalize=True):
        if normalize:
            params = self.normalize_params(params)
        
        # Encode parameters
        h = self.param_encoder(params)
        
        # Generate waveform
        x = self.init_proj(h)
        x = x.view(x.size(0), 256, 16)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.waveform_length, mode='linear', align_corners=False)
        x = self.output_scale(x)
        waveform = x.squeeze(1)
        
        # Predict metrics
        metrics = self.metrics_head(h)
        
        return waveform, metrics


# Test the model
if __name__ == '__main__':
    # Test forward surrogate
    model = ForwardSurrogate()
    
    # Random batch of parameters
    batch_size = 4
    params = torch.tensor([
        [47e-6, 100e-6, 10, 12, 100e3, 0.5],
        [22e-6, 220e-6, 5, 24, 200e3, 0.3],
        [100e-6, 470e-6, 20, 15, 50e3, 0.7],
        [33e-6, 150e-6, 8, 18, 150e3, 0.6],
    ], dtype=torch.float32)
    
    # Forward pass
    waveform = model(params)
    print(f"Input shape: {params.shape}")
    print(f"Output shape: {waveform.shape}")
    print(f"Output range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    
    # Test with metrics
    model_with_metrics = ForwardSurrogateWithMetrics()
    waveform, metrics = model_with_metrics(params)
    print(f"Metrics shape: {metrics.shape}")
