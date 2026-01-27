"""
Inverse Optimizer: Target Waveform → Circuit Parameters

Uses gradient descent through the differentiable forward surrogate
to find circuit parameters that produce the desired waveform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

from .forward_surrogate import ForwardSurrogate


class WaveformLoss(nn.Module):
    """
    Multi-component loss for waveform matching.
    Combines MSE, spectral, and engineering-aware losses.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        spectral_weight: float = 0.5,
        derivative_weight: float = 0.3,
        peak_weight: float = 0.5,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.derivative_weight = derivative_weight
        self.peak_weight = peak_weight
    
    def spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss on FFT magnitude spectrum."""
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Normalize and compare
        pred_mag = pred_mag / (pred_mag.sum(dim=-1, keepdim=True) + 1e-8)
        target_mag = target_mag / (target_mag.sum(dim=-1, keepdim=True) + 1e-8)
        
        return F.mse_loss(pred_mag, target_mag)
    
    def derivative_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss on first derivative (captures slew rate, transitions)."""
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_diff, target_diff)
    
    def peak_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss on peak values (min, max, peak-to-peak)."""
        pred_min = pred.min(dim=-1)[0]
        pred_max = pred.max(dim=-1)[0]
        target_min = target.min(dim=-1)[0]
        target_max = target.max(dim=-1)[0]
        
        loss_min = F.mse_loss(pred_min, target_min)
        loss_max = F.mse_loss(pred_max, target_max)
        loss_pp = F.mse_loss(pred_max - pred_min, target_max - target_min)
        
        return loss_min + loss_max + loss_pp
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss with breakdown.
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        mse = F.mse_loss(pred, target)
        spectral = self.spectral_loss(pred, target)
        derivative = self.derivative_loss(pred, target)
        peak = self.peak_loss(pred, target)
        
        total = (
            self.mse_weight * mse +
            self.spectral_weight * spectral +
            self.derivative_weight * derivative +
            self.peak_weight * peak
        )
        
        loss_dict = {
            'total': total.item(),
            'mse': mse.item(),
            'spectral': spectral.item(),
            'derivative': derivative.item(),
            'peak': peak.item(),
        }
        
        return total, loss_dict


class InverseOptimizer:
    """
    Gradient-based inverse design optimizer.
    Given a target waveform, finds circuit parameters that produce it.
    """
    
    # Parameter bounds (physical constraints)
    PARAM_BOUNDS = {
        'L': (1e-6, 500e-6),        # 1µH to 500µH
        'C': (10e-6, 1000e-6),      # 10µF to 1000µF
        'R_load': (0.5, 100),       # 0.5Ω to 100Ω
        'V_in': (5, 48),            # 5V to 48V
        'f_sw': (20e3, 1e6),        # 20kHz to 1MHz
        'duty': (0.05, 0.95),       # 5% to 95%
    }
    
    PARAM_NAMES = ['L', 'C', 'R_load', 'V_in', 'f_sw', 'duty']
    
    def __init__(
        self,
        surrogate: ForwardSurrogate,
        loss_fn: Optional[WaveformLoss] = None,
        device: str = 'cpu'
    ):
        self.surrogate = surrogate.to(device)
        self.surrogate.eval()  # Freeze surrogate
        self.loss_fn = loss_fn or WaveformLoss()
        self.device = device
        
        # Freeze surrogate parameters
        for param in self.surrogate.parameters():
            param.requires_grad = False
    
    def _params_to_bounded(self, raw_params: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to map unconstrained params to bounded range."""
        bounded = torch.zeros_like(raw_params)
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                # Log-scale for parameters spanning orders of magnitude
                log_low, log_high = np.log(low), np.log(high)
                bounded[:, i] = torch.exp(
                    torch.sigmoid(raw_params[:, i]) * (log_high - log_low) + log_low
                )
            else:
                bounded[:, i] = torch.sigmoid(raw_params[:, i]) * (high - low) + low
        return bounded
    
    def _init_params(self, init_guess: Optional[Dict] = None) -> torch.Tensor:
        """Initialize parameters (in unconstrained space)."""
        if init_guess is None:
            # Start in middle of range
            raw = torch.zeros(1, 6, device=self.device)
        else:
            # Convert to unconstrained space
            raw = torch.zeros(1, 6, device=self.device)
            for i, name in enumerate(self.PARAM_NAMES):
                low, high = self.PARAM_BOUNDS[name]
                val = init_guess.get(name, (low + high) / 2)
                if name in ['L', 'C', 'f_sw']:
                    log_low, log_high = np.log(low), np.log(high)
                    normalized = (np.log(val) - log_low) / (log_high - log_low)
                else:
                    normalized = (val - low) / (high - low)
                # Inverse sigmoid
                normalized = np.clip(normalized, 0.01, 0.99)
                raw[0, i] = np.log(normalized / (1 - normalized))
        
        return raw.requires_grad_(True)
    
    def optimize(
        self,
        target_waveform: torch.Tensor,
        num_iterations: int = 1000,
        lr: float = 0.1,
        init_guess: Optional[Dict] = None,
        verbose: bool = True,
        convergence_threshold: float = 1e-6,
    ) -> Dict:
        """
        Optimize circuit parameters to match target waveform.
        
        Args:
            target_waveform: (waveform_length,) target voltage waveform
            num_iterations: maximum optimization steps
            lr: learning rate
            init_guess: optional initial parameter values
            verbose: print progress
            convergence_threshold: stop if loss change < threshold
            
        Returns:
            Dictionary with optimized parameters and optimization history
        """
        # Ensure target is on device with batch dimension
        if target_waveform.dim() == 1:
            target_waveform = target_waveform.unsqueeze(0)
        target_waveform = target_waveform.to(self.device)
        
        # Initialize optimizable parameters
        raw_params = self._init_params(init_guess)
        
        # Optimizer
        optimizer = torch.optim.Adam([raw_params], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=verbose
        )
        
        # History tracking
        history = {
            'loss': [],
            'params': [],
            'loss_components': [],
        }
        
        best_loss = float('inf')
        best_params = None
        prev_loss = float('inf')
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Map to bounded params
            bounded_params = self._params_to_bounded(raw_params)
            
            # Forward through surrogate
            pred_waveform = self.surrogate(bounded_params, normalize=True)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(pred_waveform, target_waveform)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([raw_params], max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Track history
            current_params = {
                name: bounded_params[0, j].item()
                for j, name in enumerate(self.PARAM_NAMES)
            }
            history['loss'].append(loss.item())
            history['params'].append(current_params)
            history['loss_components'].append(loss_dict)
            
            # Update best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = current_params.copy()
            
            # Logging
            if verbose and (i % 100 == 0 or i == num_iterations - 1):
                print(f"Iter {i:4d} | Loss: {loss.item():.6f} | "
                      f"MSE: {loss_dict['mse']:.6f} | "
                      f"Spectral: {loss_dict['spectral']:.6f}")
            
            # Convergence check
            if abs(prev_loss - loss.item()) < convergence_threshold:
                if verbose:
                    print(f"Converged at iteration {i}")
                break
            prev_loss = loss.item()
        
        # Final prediction with best params
        with torch.no_grad():
            best_params_tensor = torch.tensor(
                [[best_params[name] for name in self.PARAM_NAMES]],
                dtype=torch.float32,
                device=self.device
            )
            final_waveform = self.surrogate(best_params_tensor, normalize=True)
        
        return {
            'best_params': best_params,
            'best_loss': best_loss,
            'final_waveform': final_waveform.squeeze(0).cpu().numpy(),
            'target_waveform': target_waveform.squeeze(0).cpu().numpy(),
            'history': history,
        }
    
    def multi_start_optimize(
        self,
        target_waveform: torch.Tensor,
        num_starts: int = 5,
        **optimize_kwargs
    ) -> Dict:
        """
        Run optimization from multiple random starting points.
        Returns the best result across all runs.
        """
        best_result = None
        best_loss = float('inf')
        
        for start_idx in range(num_starts):
            print(f"\n=== Random start {start_idx + 1}/{num_starts} ===")
            
            # Random initial guess
            init_guess = {
                name: np.random.uniform(low, high)
                if name not in ['L', 'C', 'f_sw'] else
                np.exp(np.random.uniform(np.log(low), np.log(high)))
                for name, (low, high) in self.PARAM_BOUNDS.items()
            }
            
            result = self.optimize(
                target_waveform,
                init_guess=init_guess,
                **optimize_kwargs
            )
            
            if result['best_loss'] < best_loss:
                best_loss = result['best_loss']
                best_result = result
        
        return best_result


def visualize_result(result: Dict, save_path: Optional[str] = None):
    """Visualize optimization result."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Waveform comparison
    ax1 = axes[0, 0]
    ax1.plot(result['target_waveform'], 'b-', label='Target', linewidth=2)
    ax1.plot(result['final_waveform'], 'r--', label='Predicted', linewidth=2)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Waveform Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Loss history
    ax2 = axes[0, 1]
    ax2.semilogy(result['history']['loss'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Optimization Loss')
    ax2.grid(True)
    
    # Parameter evolution
    ax3 = axes[1, 0]
    param_history = result['history']['params']
    for name in ['duty', 'R_load']:  # Plot a subset
        values = [p[name] for p in param_history]
        if name in ['L', 'C', 'f_sw']:
            values = [v * 1e6 for v in values]  # Convert to µ units
        ax3.plot(values, label=name)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Evolution')
    ax3.legend()
    ax3.grid(True)
    
    # Final parameters table
    ax4 = axes[1, 1]
    ax4.axis('off')
    params = result['best_params']
    table_data = [
        ['Parameter', 'Value'],
        ['L (µH)', f"{params['L']*1e6:.2f}"],
        ['C (µF)', f"{params['C']*1e6:.1f}"],
        ['R_load (Ω)', f"{params['R_load']:.2f}"],
        ['V_in (V)', f"{params['V_in']:.2f}"],
        ['f_sw (kHz)', f"{params['f_sw']/1e3:.1f}"],
        ['Duty', f"{params['duty']:.3f}"],
        ['---', '---'],
        ['Final Loss', f"{result['best_loss']:.6f}"],
    ]
    table = ax4.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax4.set_title('Optimized Parameters')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Demo
if __name__ == '__main__':
    # Create surrogate (would normally load trained weights)
    surrogate = ForwardSurrogate()
    
    # Create optimizer
    optimizer = InverseOptimizer(surrogate)
    
    # Create a fake target waveform (normally from SPICE or measurement)
    # This simulates a 6V DC with 0.1V ripple
    t = np.linspace(0, 1, 512)
    target = 6.0 + 0.1 * np.sin(2 * np.pi * 10 * t)  # 6V with ripple
    target_tensor = torch.tensor(target, dtype=torch.float32)
    
    print("=" * 60)
    print("Inverse Design: Finding circuit for target waveform")
    print("=" * 60)
    
    # Optimize
    result = optimizer.optimize(
        target_tensor,
        num_iterations=500,
        lr=0.1,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("OPTIMIZED CIRCUIT PARAMETERS:")
    print("=" * 60)
    for name, value in result['best_params'].items():
        if name == 'L':
            print(f"  {name}: {value*1e6:.2f} µH")
        elif name == 'C':
            print(f"  {name}: {value*1e6:.1f} µF")
        elif name == 'f_sw':
            print(f"  {name}: {value/1e3:.1f} kHz")
        else:
            print(f"  {name}: {value:.4f}")
    
    # Visualize
    visualize_result(result)
