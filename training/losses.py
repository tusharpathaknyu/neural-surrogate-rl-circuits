"""
Waveform loss functions and distance metrics for power electronics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class MSELoss(nn.Module):
    """Standard MSE loss."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class SpectralLoss(nn.Module):
    """
    Loss on frequency content.
    Important for matching ripple frequency, switching harmonics.
    """
    def __init__(self, num_bins: int = None):
        super().__init__()
        self.num_bins = num_bins
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        if self.num_bins is not None:
            pred_mag = pred_mag[:, :self.num_bins]
            target_mag = target_mag[:, :self.num_bins]
        
        return F.mse_loss(pred_mag, target_mag)


class DerivativeLoss(nn.Module):
    """
    Loss on waveform derivative.
    Captures slew rate, rise/fall times, transitions.
    """
    def __init__(self, order: int = 1):
        super().__init__()
        self.order = order
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        for _ in range(self.order):
            pred = pred[:, 1:] - pred[:, :-1]
            target = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred, target)


class PeakLoss(nn.Module):
    """
    Loss on peak characteristics.
    Critical for overshoot/undershoot constraints.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_max = pred.max(dim=-1)[0]
        pred_min = pred.min(dim=-1)[0]
        target_max = target.max(dim=-1)[0]
        target_min = target.min(dim=-1)[0]
        
        loss = (
            F.mse_loss(pred_max, target_max) +
            F.mse_loss(pred_min, target_min) +
            F.mse_loss(pred_max - pred_min, target_max - target_min)
        )
        return loss


class SettlingTimeLoss(nn.Module):
    """
    Penalize slow settling to steady state.
    Useful for transient response matching.
    """
    def __init__(self, tolerance: float = 0.02, steady_state_window: int = 50):
        super().__init__()
        self.tolerance = tolerance
        self.steady_state_window = steady_state_window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Estimate steady-state from end of waveform
        target_ss = target[:, -self.steady_state_window:].mean(dim=-1, keepdim=True)
        pred_ss = pred[:, -self.steady_state_window:].mean(dim=-1, keepdim=True)
        
        # Normalized error from steady state
        target_error = torch.abs(target - target_ss) / (torch.abs(target_ss) + 1e-8)
        pred_error = torch.abs(pred - pred_ss) / (torch.abs(pred_ss) + 1e-8)
        
        # Penalize if prediction settles slower than target
        settling_diff = F.relu(pred_error - target_error - self.tolerance)
        
        return settling_diff.mean()


class RippleLoss(nn.Module):
    """
    Loss specifically for matching ripple characteristics.
    Extracts AC component and compares.
    """
    def __init__(self, window_size: int = 64):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Extract ripple using moving average subtraction
        kernel = torch.ones(1, 1, self.window_size, device=pred.device) / self.window_size
        
        pred_expanded = pred.unsqueeze(1)
        target_expanded = target.unsqueeze(1)
        
        # Use 'same' padding to keep output size equal to input
        pad_left = (self.window_size - 1) // 2
        pad_right = self.window_size - 1 - pad_left
        
        pred_padded = F.pad(pred_expanded, (pad_left, pad_right), mode='replicate')
        target_padded = F.pad(target_expanded, (pad_left, pad_right), mode='replicate')
        
        pred_dc = F.conv1d(pred_padded, kernel)
        target_dc = F.conv1d(target_padded, kernel)
        
        pred_ac = pred_expanded - pred_dc
        target_ac = target_expanded - target_dc
        
        return F.mse_loss(pred_ac, target_ac)


class THDLoss(nn.Module):
    """
    Total Harmonic Distortion loss.
    Penalizes harmonic content relative to fundamental.
    """
    def __init__(self, fundamental_idx: int = 1, num_harmonics: int = 10):
        super().__init__()
        self.fundamental_idx = fundamental_idx
        self.num_harmonics = num_harmonics
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Calculate THD for each
        def calc_thd(mag):
            fundamental = mag[:, self.fundamental_idx]
            harmonics = mag[:, self.fundamental_idx+1:self.fundamental_idx+self.num_harmonics+1]
            harmonic_power = (harmonics ** 2).sum(dim=-1)
            thd = torch.sqrt(harmonic_power) / (fundamental + 1e-8)
            return thd
        
        pred_thd = calc_thd(pred_mag)
        target_thd = calc_thd(target_mag)
        
        return F.mse_loss(pred_thd, target_thd)


class PowerElectronicsLoss(nn.Module):
    """
    Combined loss function optimized for power electronics waveforms.
    Weights different components based on engineering importance.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        spectral_weight: float = 0.5,
        derivative_weight: float = 0.3,
        peak_weight: float = 1.0,
        ripple_weight: float = 0.5,
        settling_weight: float = 0.2,
    ):
        super().__init__()
        
        self.weights = {
            'mse': mse_weight,
            'spectral': spectral_weight,
            'derivative': derivative_weight,
            'peak': peak_weight,
            'ripple': ripple_weight,
            'settling': settling_weight,
        }
        
        self.losses = nn.ModuleDict({
            'mse': MSELoss(),
            'spectral': SpectralLoss(),
            'derivative': DerivativeLoss(),
            'peak': PeakLoss(),
            'ripple': RippleLoss(),
            'settling': SettlingTimeLoss(),
        })
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted sum of all loss components.
        
        Returns:
            total_loss: Weighted sum
            loss_dict: Individual components
        """
        loss_dict = {}
        total = torch.tensor(0.0, device=pred.device)
        
        for name, loss_fn in self.losses.items():
            component = loss_fn(pred, target)
            loss_dict[name] = component.item()
            total = total + self.weights[name] * component
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


class DTWLoss(nn.Module):
    """
    Dynamic Time Warping loss.
    Handles time-shifted waveforms gracefully.
    
    Note: This is a soft approximation since true DTW is not differentiable.
    """
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Soft-DTW approximation using pairwise distances
        batch_size, seq_len = pred.shape
        
        # Compute pairwise distance matrix
        pred_expanded = pred.unsqueeze(2)  # (B, N, 1)
        target_expanded = target.unsqueeze(1)  # (B, 1, M)
        
        dist_matrix = (pred_expanded - target_expanded) ** 2  # (B, N, M)
        
        # Soft-min DTW computation
        # This is an approximation - for exact DTW use tslearn or similar
        R = torch.zeros(batch_size, seq_len + 1, seq_len + 1, device=pred.device)
        R[:, 0, 1:] = float('inf')
        R[:, 1:, 0] = float('inf')
        
        for i in range(1, seq_len + 1):
            for j in range(1, seq_len + 1):
                r0 = R[:, i-1, j-1]
                r1 = R[:, i-1, j]
                r2 = R[:, i, j-1]
                
                # Soft minimum
                min_prev = -self.gamma * torch.logsumexp(
                    torch.stack([-r0/self.gamma, -r1/self.gamma, -r2/self.gamma], dim=-1),
                    dim=-1
                )
                
                R[:, i, j] = dist_matrix[:, i-1, j-1] + min_prev
        
        return R[:, -1, -1].mean()


# Test losses
if __name__ == '__main__':
    # Create test waveforms
    t = torch.linspace(0, 1, 512)
    target = 6.0 + 0.1 * torch.sin(2 * np.pi * 10 * t)
    pred = 6.0 + 0.12 * torch.sin(2 * np.pi * 10 * t + 0.1)  # Slightly different
    
    target = target.unsqueeze(0)
    pred = pred.unsqueeze(0)
    
    # Test combined loss
    loss_fn = PowerElectronicsLoss()
    total, components = loss_fn(pred, target)
    
    print("Power Electronics Loss Components:")
    for name, value in components.items():
        print(f"  {name}: {value:.6f}")
