"""
Multi-Topology Forward Surrogate.

Extends the base surrogate to handle multiple circuit topologies:
- Buck converter (step-down)
- Boost converter (step-up)
- Buck-Boost converter (inverted, step-up/down)

Uses topology embedding + shared backbone architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
from pathlib import Path


class TopologyEmbedding(nn.Module):
    """Learnable embedding for circuit topology."""
    
    def __init__(self, num_topologies: int = 3, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_topologies, embed_dim)
        self.topology_map = {'buck': 0, 'boost': 1, 'buck_boost': 2}
    
    def forward(self, topology_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(topology_ids)
    
    def get_id(self, topology: str) -> int:
        return self.topology_map.get(topology, 0)


class MultiTopologySurrogate(nn.Module):
    """
    Neural surrogate for multiple circuit topologies.
    
    Input: [L, C, R_load, V_in, f_sw, duty] + topology_id
    Output: 512-point waveform
    """
    
    PARAM_NAMES = ['L', 'C', 'R_load', 'V_in', 'f_sw', 'duty']
    TOPOLOGIES = ['buck', 'boost', 'buck_boost']
    
    # Parameter normalization bounds (combined for all topologies)
    PARAM_BOUNDS = {
        'L': (10e-6, 470e-6),
        'C': (47e-6, 1000e-6),
        'R_load': (2, 100),
        'V_in': (5, 24),
        'f_sw': (50e3, 500e3),
        'duty': (0.2, 0.8),
    }
    
    def __init__(
        self,
        param_dim: int = 6,
        output_points: int = 512,
        hidden_dim: int = 256,
        num_topologies: int = 3,
        topology_embed_dim: int = 32,
    ):
        super().__init__()
        
        self.param_dim = param_dim
        self.output_points = output_points
        
        # Topology embedding
        self.topology_embed = TopologyEmbedding(num_topologies, topology_embed_dim)
        
        # Parameter encoder (includes topology embedding)
        input_dim = param_dim + topology_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
        )
        
        # Topology-specific heads (learned specialization)
        self.topology_heads = nn.ModuleDict({
            'buck': nn.Linear(hidden_dim * 2, hidden_dim),
            'boost': nn.Linear(hidden_dim * 2, hidden_dim),
            'buck_boost': nn.Linear(hidden_dim * 2, hidden_dim),
        })
        
        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_points),
        )
        
        # Waveform refinement (1D conv)
        self.refine = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )
        
        # Output scaling per topology
        self.output_scale = nn.ParameterDict({
            'buck': nn.Parameter(torch.ones(1)),
            'boost': nn.Parameter(torch.ones(1) * 1.5),
            'buck_boost': nn.Parameter(torch.ones(1)),
        })
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize parameters to [0, 1]."""
        normalized = torch.zeros_like(params)
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                # Log scale
                normalized[:, i] = (torch.log(params[:, i]) - np.log(low)) / (np.log(high) - np.log(low))
            else:
                normalized[:, i] = (params[:, i] - low) / (high - low)
        return normalized.clamp(0, 1)
    
    def forward(
        self,
        params: torch.Tensor,
        topology_ids: Optional[torch.Tensor] = None,
        topology_names: Optional[list] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            params: [batch, 6] circuit parameters
            topology_ids: [batch] integer topology IDs (0=buck, 1=boost, 2=buck_boost)
            topology_names: list of topology name strings (alternative to ids)
            normalize: whether to normalize inputs
        """
        batch_size = params.shape[0]
        device = params.device
        
        # Handle topology specification
        if topology_ids is None and topology_names is None:
            topology_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif topology_names is not None:
            topology_ids = torch.tensor(
                [self.topology_embed.get_id(t) for t in topology_names],
                dtype=torch.long, device=device
            )
        
        # Normalize parameters
        if normalize:
            params = self.normalize_params(params)
        
        # Get topology embedding
        topo_embed = self.topology_embed(topology_ids)
        
        # Combine params with topology
        x = torch.cat([params, topo_embed], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Apply topology-specific heads (soft routing based on topology)
        outputs = []
        for i in range(batch_size):
            topo_name = self.TOPOLOGIES[topology_ids[i].item()]
            head_out = self.topology_heads[topo_name](features[i:i+1])
            outputs.append(head_out)
        features = torch.cat(outputs, dim=0)
        
        # Decode to waveform
        waveform = self.decoder(features)
        
        # Refine with conv
        waveform = waveform.unsqueeze(1)  # [batch, 1, 512]
        waveform = waveform + self.refine(waveform)
        waveform = waveform.squeeze(1)  # [batch, 512]
        
        # Scale by topology
        for i in range(batch_size):
            topo_name = self.TOPOLOGIES[topology_ids[i].item()]
            waveform[i] = waveform[i] * self.output_scale[topo_name]
        
        return waveform


class MultiTopologySurrogateWithMetrics(MultiTopologySurrogate):
    """Extended surrogate with engineering metrics computation."""
    
    def compute_metrics(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute power electronics metrics from waveform."""
        metrics = {}
        
        # DC component (mean)
        metrics['v_dc'] = waveform.mean(dim=-1)
        
        # Ripple (peak-to-peak)
        metrics['ripple_pp'] = waveform.max(dim=-1)[0] - waveform.min(dim=-1)[0]
        
        # RMS
        metrics['v_rms'] = torch.sqrt((waveform ** 2).mean(dim=-1))
        
        # Ripple percentage
        metrics['ripple_pct'] = metrics['ripple_pp'] / (metrics['v_dc'].abs() + 1e-6) * 100
        
        return metrics
    
    def forward_with_metrics(
        self,
        params: torch.Tensor,
        topology_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple:
        waveform = self.forward(params, topology_ids, **kwargs)
        metrics = self.compute_metrics(waveform)
        return waveform, metrics


def test_multi_topology():
    """Test multi-topology surrogate."""
    print("Testing Multi-Topology Surrogate...")
    
    model = MultiTopologySurrogateWithMetrics()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test each topology
    for topo in ['buck', 'boost', 'buck_boost']:
        params = torch.randn(4, 6).abs() * 0.5 + 0.25  # Random normalized params
        topology_names = [topo] * 4
        
        waveform, metrics = model.forward_with_metrics(
            params, topology_names=topology_names, normalize=False
        )
        
        print(f"\n{topo}:")
        print(f"  Output shape: {waveform.shape}")
        print(f"  V_dc: {metrics['v_dc'].mean():.2f}V")
        print(f"  Ripple: {metrics['ripple_pct'].mean():.1f}%")
    
    print("\n✓ Multi-topology surrogate test passed!")


if __name__ == '__main__':
    test_multi_topology()
