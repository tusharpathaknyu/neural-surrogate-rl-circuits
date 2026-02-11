"""
Multi-Topology Forward Surrogate.

Supports 7 circuit topologies:
- Buck converter (step-down)
- Boost converter (step-up)  
- Buck-Boost converter (inverted, step-up/down)
- SEPIC (Single-Ended Primary-Inductor Converter)
- Ćuk converter (inverted output)
- Flyback converter (isolated)
- QR Flyback (Quasi-Resonant Flyback - soft switching)

Architecture matches the trained checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path


class MultiTopologySurrogate(nn.Module):
    """
    Neural surrogate supporting multiple circuit topologies.
    
    This architecture matches the trained checkpoint.
    
    Input: [L, C, R_load, V_in, f_sw, duty] + topology_id
    Output: 512-point waveform + metrics (efficiency, ripple)
    """
    
    PARAM_NAMES = ['L', 'C', 'R_load', 'V_in', 'f_sw', 'duty']
    TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
    
    # Parameter normalization bounds (extended for all topologies)
    PARAM_BOUNDS = {
        'L': (10e-6, 470e-6),
        'C': (47e-6, 1000e-6),
        'R_load': (2, 100),
        'V_in': (5, 48),  # Extended range
        'f_sw': (50e3, 500e3),
        'duty': (0.2, 0.8),
    }
    
    def __init__(self, num_topologies=7, param_dim=6, waveform_len=512, 
                 embed_dim=32, hidden_dim=256):
        super().__init__()
        
        self.num_topologies = num_topologies
        self.param_dim = param_dim
        self.waveform_len = waveform_len
        
        # Topology embedding (direct nn.Embedding to match trained checkpoint)
        self.topology_embedding = nn.Embedding(num_topologies, embed_dim)
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(param_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Waveform generator
        self.waveform_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, waveform_len),
        )
        
        # 1D conv refiner for temporal coherence
        self.refiner = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )
        
        # Metrics head (efficiency, ripple)
        self.metrics_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # efficiency, ripple
            nn.Sigmoid(),
        )
    
    @classmethod
    def get_topology_id(cls, topology) -> int:
        """Get topology ID from name or return if already int."""
        if isinstance(topology, int):
            return topology
        topology_map = {t: i for i, t in enumerate(cls.TOPOLOGIES)}
        return topology_map.get(topology.lower(), 0)
    
    @classmethod
    def get_topology_name(cls, topology) -> str:
        """Get topology name from ID or return if already string."""
        if isinstance(topology, str):
            return topology.lower()
        return cls.TOPOLOGIES[topology] if 0 <= topology < len(cls.TOPOLOGIES) else 'buck'
    
    @classmethod
    def compute_theoretical_vout(cls, v_in: torch.Tensor, duty: torch.Tensor, 
                                  topology) -> torch.Tensor:
        """
        Compute theoretical output voltage based on topology equations.
        
        Args:
            v_in: Input voltage
            duty: Duty cycle (0-1)
            topology: Topology name (str) or ID (int)
            
        Returns:
            Theoretical output voltage
        """
        eps = 1e-6
        duty = duty.clamp(0.1, 0.9)  # Prevent extreme values
        
        # Convert to string if int
        topo_name = cls.get_topology_name(topology)
        
        if topo_name == 'buck':
            return v_in * duty
        elif topo_name == 'boost':
            return v_in / (1 - duty + eps)
        elif topo_name in ['buck_boost', 'cuk']:
            return v_in * duty / (1 - duty + eps)
        elif topo_name == 'sepic':
            return v_in * duty / (1 - duty + eps)
        elif topo_name == 'flyback':
            # Assuming 1:1 turns ratio
            return v_in * duty / (1 - duty + eps)
        else:
            return v_in * duty  # Default to buck
    
    def denormalize_waveform(self, waveform: torch.Tensor, 
                              params: torch.Tensor,
                              topology) -> torch.Tensor:
        """
        Denormalize waveform from model output to actual voltage.
        
        FIXED (v2): Uses per-topology normalization stats stored in checkpoint.
        
        Old approach REPLACED the model's learned output with theoretical DC
        values + 5% synthetic ripple — the model's actual prediction was
        completely thrown away. This made every topology output exactly the
        theoretical formula (V_in*D for buck, etc.), hiding the fact that
        the model hadn't learned anything useful for complex topologies.
        
        New approach: Reverse the per-topology (mean, std) normalization
        used during training. If stats aren't available (old checkpoint),
        fall back to theoretical scaling as before.
        
        Args:
            waveform: Normalized waveform from model output
            params: Original (non-normalized) circuit parameters
            topology: Topology name (str) or ID (int)
            
        Returns:
            Waveform scaled to actual voltage values
        """
        topo_id = self.get_topology_id(topology)
        
        # Use stored per-topology stats if available (new checkpoints)
        if hasattr(self, '_waveform_stats') and self._waveform_stats is not None:
            stats = self._waveform_stats.get(topo_id, self._waveform_stats.get(str(topo_id)))
            if stats is not None:
                mean = stats['mean']
                std = stats['std']
                return waveform * std + mean
        
        # Fallback for old checkpoints: use theoretical scaling
        # (This is the old buggy approach — kept only for backward compat)
        v_in = params[:, 3]  # V_in is at index 3
        duty = params[:, 5]  # duty is at index 5
        v_out_theoretical = self.compute_theoretical_vout(v_in, duty, topology)
        waveform_mean = waveform.mean(dim=-1, keepdim=True)
        waveform_centered = waveform - waveform_mean
        ripple_scale = 0.05 * v_out_theoretical.unsqueeze(-1)
        v_out_theoretical = v_out_theoretical.unsqueeze(-1)
        denorm_waveform = v_out_theoretical + waveform_centered * ripple_scale
        return denorm_waveform
    
    def normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize parameters to [0, 1] range."""
        normalized = torch.zeros_like(params)
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                # Log scale normalization
                log_val = torch.log(params[:, i].clamp(min=low))
                normalized[:, i] = (log_val - np.log(low)) / (np.log(high) - np.log(low))
            else:
                normalized[:, i] = (params[:, i] - low) / (high - low)
        return normalized.clamp(0, 1)
    
    def forward(self, params: torch.Tensor, topology_ids: torch.Tensor,
                normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            params: (batch, param_dim) - circuit parameters
            topology_ids: (batch,) - integer topology IDs
            normalize: whether to normalize inputs (False if already normalized)
        
        Returns:
            waveforms: (batch, waveform_len)
            metrics: (batch, 2) - efficiency, ripple
        """
        # Normalize if needed
        if normalize:
            params = self.normalize_params(params)
        
        # Get topology embedding
        topo_embed = self.topology_embedding(topology_ids)  # (batch, embed_dim)
        
        # Concatenate params and topology
        x = torch.cat([params, topo_embed], dim=1)  # (batch, param_dim + embed_dim)
        
        # Encode
        features = self.encoder(x)  # (batch, hidden_dim)
        
        # Generate waveform
        waveform = self.waveform_head(features)  # (batch, waveform_len)
        
        # Refine with conv
        waveform = waveform.unsqueeze(1)  # (batch, 1, waveform_len)
        waveform = waveform + self.refiner(waveform)  # residual
        waveform = waveform.squeeze(1)  # (batch, waveform_len)
        
        # Predict metrics
        metrics = self.metrics_head(features)
        
        return waveform, metrics
    
    def predict_waveform(self, params: torch.Tensor, topology: str,
                         normalize: bool = True) -> torch.Tensor:
        """
        Convenience method to predict waveform for a single topology.
        
        Args:
            params: (batch, 6) circuit parameters
            topology: topology name string
            normalize: whether to normalize inputs
            
        Returns:
            waveform: (batch, 512)
        """
        device = params.device
        batch_size = params.shape[0]
        topology_id = self.get_topology_id(topology)
        topology_ids = torch.full((batch_size,), topology_id, dtype=torch.long, device=device)
        
        waveform, _ = self.forward(params, topology_ids, normalize=normalize)
        return waveform
    
    def predict_voltage(self, params: torch.Tensor, topology: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict actual output voltage waveform (denormalized).
        
        This is the recommended method for getting realistic voltage predictions.
        Uses physics-based scaling to convert model output to actual voltages.
        
        Args:
            params: (batch, 6) circuit parameters [L, C, R_load, V_in, f_sw, duty]
                   Should be in physical units (not normalized)
            topology: topology name string
            
        Returns:
            waveform: (batch, 512) - actual voltage waveform
            v_out: (batch,) - predicted DC output voltage
        """
        device = params.device
        batch_size = params.shape[0]
        topology_id = self.get_topology_id(topology)
        topology_ids = torch.full((batch_size,), topology_id, dtype=torch.long, device=device)
        
        # Get normalized waveform
        waveform, _ = self.forward(params, topology_ids, normalize=True)
        
        # Denormalize to actual voltage
        waveform = self.denormalize_waveform(waveform, params, topology)
        
        # Compute DC output
        v_out = waveform.mean(dim=-1)
        
        return waveform, v_out


class MultiTopologySurrogateWithMetrics(MultiTopologySurrogate):
    """Extended surrogate with additional engineering metrics computation."""
    
    def compute_engineering_metrics(self, waveform: torch.Tensor, 
                                    v_in: torch.Tensor,
                                    duty: torch.Tensor,
                                    topology: str) -> Dict[str, torch.Tensor]:
        """
        Compute power electronics metrics from waveform.
        
        Args:
            waveform: (batch, 512) output voltage waveform
            v_in: (batch,) input voltage
            duty: (batch,) duty cycle
            topology: topology name
        
        Returns:
            Dictionary of engineering metrics
        """
        metrics = {}
        
        # DC component (mean output voltage)
        v_dc = waveform.mean(dim=-1)
        metrics['v_out'] = v_dc
        
        # Ripple (peak-to-peak)
        ripple_pp = waveform.max(dim=-1)[0] - waveform.min(dim=-1)[0]
        metrics['ripple_pp'] = ripple_pp
        
        # Ripple percentage
        metrics['ripple_pct'] = ripple_pp / (v_dc.abs() + 1e-6) * 100
        
        # RMS voltage
        metrics['v_rms'] = torch.sqrt((waveform ** 2).mean(dim=-1))
        
        # Theoretical output voltage based on topology
        if topology == 'buck':
            v_out_ideal = v_in * duty
        elif topology == 'boost':
            v_out_ideal = v_in / (1 - duty + 1e-6)
        elif topology in ['buck_boost', 'cuk']:
            v_out_ideal = v_in * duty / (1 - duty + 1e-6)
        elif topology == 'sepic':
            v_out_ideal = v_in * duty / (1 - duty + 1e-6)
        elif topology == 'flyback':
            # Assuming 1:1 turns ratio
            v_out_ideal = v_in * duty / (1 - duty + 1e-6)
        else:
            v_out_ideal = v_dc
            
        metrics['v_out_ideal'] = v_out_ideal
        
        # Regulation accuracy (how close to ideal)
        metrics['regulation_error'] = (v_dc - v_out_ideal).abs() / (v_out_ideal.abs() + 1e-6) * 100
        
        return metrics
    
    def forward_with_metrics(
        self,
        params: torch.Tensor,
        topology_ids: torch.Tensor,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with full metrics.
        
        Returns:
            waveform: (batch, 512)
            metrics: dict with efficiency, ripple, v_out, etc.
        """
        waveform, raw_metrics = self.forward(params, topology_ids, normalize)
        
        # Extract efficiency and ripple from network
        metrics = {
            'efficiency': raw_metrics[:, 0],  # 0-1 range
            'ripple_norm': raw_metrics[:, 1],  # normalized ripple
        }
        
        # Add waveform-based metrics
        v_dc = waveform.mean(dim=-1)
        ripple_pp = waveform.max(dim=-1)[0] - waveform.min(dim=-1)[0]
        
        metrics['v_out'] = v_dc
        metrics['ripple_pp'] = ripple_pp
        metrics['ripple_pct'] = ripple_pp / (v_dc.abs() + 1e-6) * 100
        
        return waveform, metrics


def load_trained_model(checkpoint_path: str = None, device: str = 'cpu') -> MultiTopologySurrogate:
    """
    Load the trained multi-topology surrogate model.
    
    Args:
        checkpoint_path: path to checkpoint file (default: auto-detect)
        device: device to load model on
        
    Returns:
        Loaded MultiTopologySurrogate model
    """
    if checkpoint_path is None:
        # Auto-detect checkpoint
        default_paths = [
            Path(__file__).parent.parent / 'checkpoints' / 'multi_topology_surrogate.pt',
            Path('checkpoints/multi_topology_surrogate.pt'),
        ]
        for path in default_paths:
            if path.exists():
                checkpoint_path = str(path)
                break
    
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False needed for numpy scalars in older checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Infer model architecture from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Get dimensions from checkpoint weights
    num_topologies = state_dict['topology_embedding.weight'].shape[0]
    embed_dim = state_dict['topology_embedding.weight'].shape[1]
    hidden_dim = state_dict['encoder.0.weight'].shape[0]  # First encoder layer output
    waveform_len = state_dict['waveform_head.2.weight'].shape[0]  # Waveform output length
    
    # Create model with correct architecture
    model = MultiTopologySurrogate(
        num_topologies=num_topologies,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        waveform_len=waveform_len
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    # Load normalization stats if available (needed for denormalization)
    waveform_stats = checkpoint.get('waveform_stats', None)
    model._waveform_stats = waveform_stats
    if waveform_stats:
        print(f"  Loaded per-topology waveform normalization stats")
    else:
        print(f"  Warning: No waveform stats in checkpoint, using theoretical denorm (legacy)")
    
    model.to(device)
    model.eval()
    
    return model


def test_multi_topology():
    """Test multi-topology surrogate."""
    print("Testing Multi-Topology Surrogate...")
    
    model = MultiTopologySurrogate()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test each topology
    for i, topo in enumerate(MultiTopologySurrogate.TOPOLOGIES):
        params = torch.randn(4, 6).abs() * 0.5 + 0.25  # Random normalized params
        topology_ids = torch.full((4,), i, dtype=torch.long)
        
        waveform, metrics = model(params, topology_ids)
        
        print(f"\n{topo}:")
        print(f"  Output shape: {waveform.shape}")
        print(f"  V_dc: {waveform.mean():.2f}V")
        print(f"  Efficiency: {metrics[:, 0].mean():.1%}")
    
    print("\n✓ Multi-topology surrogate test passed!")


if __name__ == '__main__':
    test_multi_topology()
