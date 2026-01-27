#!/usr/bin/env python3
"""Quick test of surrogate model"""

import torch
from models.multi_topology_surrogate import load_trained_model

model = load_trained_model(device='cpu')
model.eval()

print("="*50)
print("SURROGATE MODEL QUICK TEST")
print("="*50)

# Test each topology
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']

for topo_idx, topo_name in enumerate(TOPOLOGIES):
    # Test with mid-range parameters
    params = torch.tensor([[0.5, 0.3, 0.5, 0.5, 0.5, 0.5]])
    topology_ids = torch.tensor([topo_idx])
    
    with torch.no_grad():
        waveform, metrics = model(params, topology_ids)
    
    eff = metrics[0, 0].item()
    ripple = metrics[0, 1].item()
    
    print(f"\n{topo_name.upper()}:")
    print(f"  Efficiency: {eff:.1%}")
    print(f"  Ripple: {ripple:.2%}")
    
# Check raw outputs (unscaled)
print("\n" + "="*50)
print("RAW OUTPUT CHECK (metrics_head output)")
print("="*50)

# Get the raw output before any scaling
params = torch.tensor([[0.5, 0.3, 0.5, 0.5, 0.5, 0.5]])
topology_ids = torch.tensor([0])

with torch.no_grad():
    waveform, metrics = model(params, topology_ids)

print(f"Metrics tensor: {metrics}")
print(f"Metrics shape: {metrics.shape}")
print(f"Waveform stats: min={waveform.min():.3f}, max={waveform.max():.3f}, mean={waveform.mean():.3f}")
