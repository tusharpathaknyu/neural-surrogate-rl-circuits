"""
Uncertainty Quantification for Surrogate Predictions.

Uses Monte Carlo Dropout to estimate prediction confidence.
This helps identify when the model is uncertain about its predictions.

Key features:
1. MC Dropout: Run inference multiple times with dropout enabled
2. Epistemic uncertainty: Model uncertainty from limited training data
3. Confidence intervals: Provide bounds on predictions
4. Out-of-distribution detection: Flag unusual inputs
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using MC Dropout.
    
    MC Dropout: At inference time, keep dropout enabled and run
    multiple forward passes. The variance of outputs indicates
    model uncertainty.
    """
    
    def __init__(self, model, n_samples: int = 30, device: str = 'cpu'):
        """
        Args:
            model: trained surrogate model
            n_samples: number of MC samples (more = better estimate, slower)
            device: computation device
        """
        self.model = model.to(device)
        self.n_samples = n_samples
        self.device = device
    
    def _enable_dropout(self):
        """Enable dropout layers for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def _disable_dropout(self):
        """Disable dropout for normal inference."""
        self.model.eval()
    
    def predict_with_uncertainty(self, params: torch.Tensor, 
                                  topology_ids: torch.Tensor) -> Dict:
        """
        Get prediction with uncertainty estimates.
        
        Args:
            params: (batch, 6) circuit parameters
            topology_ids: (batch,) topology IDs
            
        Returns:
            dict with mean, std, confidence intervals, etc.
        """
        params = params.to(self.device)
        topology_ids = topology_ids.to(self.device)
        
        batch_size = params.shape[0]
        
        # Collect MC samples
        waveform_samples = []
        metrics_samples = []
        
        self._enable_dropout()
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                waveform, metrics = self.model(params, topology_ids, normalize=True)
                waveform_samples.append(waveform.cpu().numpy())
                metrics_samples.append(metrics.cpu().numpy())
        
        self._disable_dropout()
        
        # Stack samples: (n_samples, batch, waveform_len)
        waveform_samples = np.stack(waveform_samples, axis=0)
        metrics_samples = np.stack(metrics_samples, axis=0)
        
        # Compute statistics
        waveform_mean = np.mean(waveform_samples, axis=0)
        waveform_std = np.std(waveform_samples, axis=0)
        
        metrics_mean = np.mean(metrics_samples, axis=0)
        metrics_std = np.std(metrics_samples, axis=0)
        
        # Confidence intervals (95%)
        waveform_ci_low = np.percentile(waveform_samples, 2.5, axis=0)
        waveform_ci_high = np.percentile(waveform_samples, 97.5, axis=0)
        
        # Overall uncertainty score (mean std across waveform)
        uncertainty_score = np.mean(waveform_std, axis=-1)
        
        # Relative uncertainty (std / mean, avoiding division by zero)
        relative_uncertainty = np.mean(waveform_std / (np.abs(waveform_mean) + 1e-6), axis=-1)
        
        # Confidence level: high uncertainty = low confidence
        # Scale to 0-100%
        confidence = np.clip(100 * (1 - relative_uncertainty * 10), 0, 100)
        
        return {
            'waveform_mean': waveform_mean,
            'waveform_std': waveform_std,
            'waveform_ci_low': waveform_ci_low,
            'waveform_ci_high': waveform_ci_high,
            'metrics_mean': metrics_mean,
            'metrics_std': metrics_std,
            'uncertainty_score': uncertainty_score,
            'relative_uncertainty': relative_uncertainty,
            'confidence_pct': confidence,
            'n_samples': self.n_samples,
        }
    
    def is_out_of_distribution(self, params: torch.Tensor,
                                topology_ids: torch.Tensor,
                                threshold: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect if inputs are out-of-distribution (OOD).
        
        High uncertainty often indicates the model hasn't seen
        similar examples during training.
        
        Args:
            params: circuit parameters
            topology_ids: topology IDs
            threshold: relative uncertainty threshold for OOD
            
        Returns:
            is_ood: boolean array
            uncertainty: uncertainty scores
        """
        result = self.predict_with_uncertainty(params, topology_ids)
        
        is_ood = result['relative_uncertainty'] > threshold
        
        return is_ood, result['relative_uncertainty']


class RobustnessAnalyzer:
    """
    Analyze circuit design robustness to component tolerances.
    
    Real components have tolerances (±5% to ±20%). A good design
    should work across these variations.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Typical component tolerances
        self.tolerances = {
            'L': 0.10,      # ±10% for inductors
            'C': 0.20,      # ±20% for electrolytics
            'R_load': 0.05, # ±5% for resistors
            'V_in': 0.05,   # ±5% voltage variation
            'f_sw': 0.02,   # ±2% oscillator tolerance
            'duty': 0.01,   # ±1% PWM accuracy
        }
    
    def monte_carlo_robustness(self, params: torch.Tensor,
                               topology_ids: torch.Tensor,
                               n_samples: int = 100) -> Dict:
        """
        Monte Carlo analysis of component tolerance effects.
        
        Args:
            params: nominal circuit parameters
            topology_ids: topology IDs
            n_samples: number of Monte Carlo samples
            
        Returns:
            dict with statistics on output variation
        """
        params = params.to(self.device)
        topology_ids = topology_ids.to(self.device)
        
        batch_size = params.shape[0]
        
        # Generate perturbed parameters
        waveforms = []
        metrics_list = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_samples):
                # Apply random perturbations within tolerances
                perturbed = params.clone()
                for i, (name, tol) in enumerate(self.tolerances.items()):
                    # Uniform distribution within tolerance
                    perturbation = torch.empty_like(perturbed[:, i]).uniform_(1-tol, 1+tol)
                    perturbed[:, i] = perturbed[:, i] * perturbation
                
                waveform, metrics = self.model(perturbed, topology_ids, normalize=True)
                waveforms.append(waveform.cpu().numpy())
                metrics_list.append(metrics.cpu().numpy())
        
        waveforms = np.stack(waveforms, axis=0)  # (n_samples, batch, 512)
        metrics_arr = np.stack(metrics_list, axis=0)  # (n_samples, batch, 2)
        
        # Compute output statistics
        v_out_samples = np.mean(waveforms, axis=-1)  # DC output for each sample
        ripple_samples = np.max(waveforms, axis=-1) - np.min(waveforms, axis=-1)
        
        v_out_mean = np.mean(v_out_samples, axis=0)
        v_out_std = np.std(v_out_samples, axis=0)
        v_out_range = np.max(v_out_samples, axis=0) - np.min(v_out_samples, axis=0)
        
        ripple_mean = np.mean(ripple_samples, axis=0)
        ripple_std = np.std(ripple_samples, axis=0)
        ripple_worst = np.max(ripple_samples, axis=0)
        
        # Efficiency variation
        eff_samples = metrics_arr[:, :, 0]  # First metric is efficiency
        eff_mean = np.mean(eff_samples, axis=0)
        eff_std = np.std(eff_samples, axis=0)
        eff_worst = np.min(eff_samples, axis=0)
        
        # Robustness score: lower variation = higher robustness
        # Scale relative variation to 0-100
        relative_v_out_var = v_out_std / (np.abs(v_out_mean) + 1e-6)
        robustness_score = np.clip(100 * (1 - relative_v_out_var * 5), 0, 100)
        
        return {
            'v_out_mean': v_out_mean,
            'v_out_std': v_out_std,
            'v_out_range': v_out_range,
            'ripple_mean': ripple_mean,
            'ripple_std': ripple_std,
            'ripple_worst': ripple_worst,
            'efficiency_mean': eff_mean * 100,  # Convert to %
            'efficiency_std': eff_std * 100,
            'efficiency_worst': eff_worst * 100,
            'robustness_score': robustness_score,
            'n_samples': n_samples,
            'tolerances': self.tolerances,
        }
    
    def worst_case_analysis(self, params: torch.Tensor,
                           topology_ids: torch.Tensor) -> Dict:
        """
        Worst-case corner analysis.
        
        Tests all combinations of min/max component values
        to find absolute worst-case performance.
        """
        params = params.to(self.device)
        topology_ids = topology_ids.to(self.device)
        
        batch_size = params.shape[0]
        
        # Generate corners (2^6 = 64 combinations)
        n_params = 6
        n_corners = 2 ** n_params
        
        results = []
        
        self.model.eval()
        with torch.no_grad():
            for corner in range(n_corners):
                perturbed = params.clone()
                for i, (name, tol) in enumerate(self.tolerances.items()):
                    # Bit i determines min or max
                    if (corner >> i) & 1:
                        perturbed[:, i] = perturbed[:, i] * (1 + tol)
                    else:
                        perturbed[:, i] = perturbed[:, i] * (1 - tol)
                
                waveform, metrics = self.model(perturbed, topology_ids, normalize=True)
                v_out = torch.mean(waveform, dim=-1).cpu().numpy()
                ripple = (torch.max(waveform, dim=-1)[0] - torch.min(waveform, dim=-1)[0]).cpu().numpy()
                
                results.append({
                    'corner': corner,
                    'v_out': v_out,
                    'ripple': ripple,
                    'efficiency': metrics[:, 0].cpu().numpy(),
                })
        
        # Find worst cases
        all_v_out = np.array([r['v_out'] for r in results])
        all_ripple = np.array([r['ripple'] for r in results])
        all_eff = np.array([r['efficiency'] for r in results])
        
        return {
            'v_out_min': np.min(all_v_out, axis=0),
            'v_out_max': np.max(all_v_out, axis=0),
            'v_out_nominal': params[:, 3].cpu().numpy() * params[:, 5].cpu().numpy(),  # Approx
            'ripple_worst': np.max(all_ripple, axis=0),
            'efficiency_worst': np.min(all_eff, axis=0) * 100,
            'n_corners': n_corners,
        }


def demonstrate_uncertainty():
    """Demo of uncertainty quantification."""
    from models.multi_topology_surrogate import load_trained_model
    
    print("\n" + "="*60)
    print("Uncertainty Quantification Demo")
    print("="*60)
    
    # Load model
    model = load_trained_model(device='cpu')
    estimator = UncertaintyEstimator(model, n_samples=30)
    analyzer = RobustnessAnalyzer(model)
    
    # Test parameters
    params = torch.tensor([[
        50e-6,   # L = 50µH
        220e-6,  # C = 220µF
        10.0,    # R = 10Ω
        12.0,    # Vin = 12V
        100e3,   # f = 100kHz
        0.5,     # D = 50%
    ]], dtype=torch.float32)
    
    topology_ids = torch.tensor([0])  # Buck
    
    # 1. Uncertainty estimation
    print("\n1. MC Dropout Uncertainty:")
    result = estimator.predict_with_uncertainty(params, topology_ids)
    print(f"   Predicted Vout: {result['waveform_mean'].mean():.2f}V")
    print(f"   Uncertainty (std): {result['waveform_std'].mean():.3f}V")
    print(f"   Confidence: {result['confidence_pct'][0]:.1f}%")
    
    # 2. OOD detection
    print("\n2. Out-of-Distribution Detection:")
    is_ood, uncertainty = estimator.is_out_of_distribution(params, topology_ids)
    print(f"   Is OOD: {is_ood[0]}")
    print(f"   Relative uncertainty: {uncertainty[0]:.4f}")
    
    # 3. Robustness analysis
    print("\n3. Component Tolerance Robustness:")
    robustness = analyzer.monte_carlo_robustness(params, topology_ids, n_samples=100)
    print(f"   Vout range: {robustness['v_out_mean'][0]:.2f}V ± {robustness['v_out_std'][0]:.2f}V")
    print(f"   Worst ripple: {robustness['ripple_worst'][0]:.3f}V")
    print(f"   Robustness score: {robustness['robustness_score'][0]:.1f}%")
    
    # 4. Worst-case corners
    print("\n4. Worst-Case Corner Analysis:")
    worst = analyzer.worst_case_analysis(params, topology_ids)
    print(f"   Vout range: {worst['v_out_min'][0]:.2f}V to {worst['v_out_max'][0]:.2f}V")
    print(f"   Worst ripple: {worst['ripple_worst'][0]:.3f}V")
    print(f"   Worst efficiency: {worst['efficiency_worst'][0]:.1f}%")
    
    print("\n✓ Uncertainty quantification complete!")


if __name__ == '__main__':
    demonstrate_uncertainty()
