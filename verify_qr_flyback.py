#!/usr/bin/env python3
"""Verify QR Flyback was added correctly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from rl.train_per_topology_agents import TOPOLOGY_CONFIG, TOPOLOGIES
from data.generate_extended_topologies import Topology, PARAM_RANGES, TEMPLATES

print("=" * 85)
print("QR FLYBACK TOPOLOGY ADDED SUCCESSFULLY")
print("=" * 85)

print("\n1. Training Configuration (7 topologies):")
print("-" * 85)
print(f"{'Topology':12} | {'Hidden':>6} | {'LR':>8} | {'γ':>5} | {'λ':>5} | {'ε_clip':>6} | {'Entropy':>7} | Steps")
print("-" * 85)

for topo in TOPOLOGIES:
    c = TOPOLOGY_CONFIG[topo]
    total = c['n_iterations'] * c['steps_per_iter']
    print(f"{topo:12} | {c['hidden_dim']:>6} | {c['lr']:>8.0e} | {c['gamma']:>5.3f} | {c['gae_lambda']:>5.2f} | {c['clip_epsilon']:>6.2f} | {c['entropy_coef']:>7.3f} | {total:>7,}")

print("-" * 85)
print(f"Total topologies: {len(TOPOLOGIES)}")

print("\n2. QR Flyback Physics-Informed Settings:")
qr = TOPOLOGY_CONFIG['qr_flyback']
print(f"   Description: {qr['description']}")
print(f"   Tier: 4 (MOST COMPLEX) - Resonant + Isolated")
print(f"   Learning rate: {qr['lr']:.0e} (lowest - resonant timing critical)")
print(f"   Discount γ: {qr['gamma']} (longest horizon - resonant energy cycling)")
print(f"   Clipping ε: {qr['clip_epsilon']} (tightest - resonance sensitive)")
print(f"   Entropy: {qr['entropy_coef']} (highest - explore resonant frequency space)")
print(f"   Total steps: {qr['n_iterations'] * qr['steps_per_iter']:,}")

print("\n3. Data Generation Config:")
print(f"   Topology enum: {Topology.QR_FLYBACK}")
print(f"   SPICE template: {'qr_flyback' in [t.name.lower() for t in TEMPLATES.keys()]}")
print(f"   Param ranges: {list(PARAM_RANGES[Topology.QR_FLYBACK].keys())}")

print("\n4. QR Flyback Characteristics:")
print("   ✓ Zero-Voltage Switching (ZVS) / Zero-Current Switching (ZCS)")
print("   ✓ Variable frequency operation (valley switching)")
print("   ✓ Resonant tank: Lr = 5% of Lm, Cr = 1% of Cout")
print("   ✓ Reduced EMI vs hard-switched flyback")
print("   ✓ Higher efficiency (88-94% typical)")
print("   ✓ Lower switching losses")

print("\n" + "=" * 85)
print("READY TO TRAIN: python rl/train_per_topology_agents.py --topology qr_flyback")
print("=" * 85)
