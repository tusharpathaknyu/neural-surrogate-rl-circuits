#!/usr/bin/env python3
"""
Final Comparison: Original vs Enhanced Models
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']


def compute_waveform_quality(waveform: np.ndarray) -> dict:
    """Compute quality metrics from waveform."""
    mean_v = np.mean(waveform)
    std_v = np.std(waveform)
    
    stability = 1 - min(std_v / (abs(mean_v) + 1e-6), 1)
    ripple = (np.max(waveform) - np.min(waveform)) / (abs(mean_v) + 1e-6)
    
    # FFT for THD
    fft = np.abs(np.fft.rfft(waveform))
    if len(fft) > 1 and fft[1] > 0:
        thd = np.sqrt(np.sum(fft[2:11]**2)) / (fft[1] + 1e-6)
    else:
        thd = 0
    
    quality = 0.4 * stability + 0.3 * (1 - min(ripple, 1)) + 0.3 * (1 - min(thd, 1))
    
    return {'quality': quality, 'stability': stability, 'ripple': ripple, 'thd': thd}


def evaluate_original():
    """Evaluate original surrogate model."""
    print("\n📊 ORIGINAL MODEL")
    print("-"*40)
    
    from models.multi_topology_surrogate import load_trained_model
    model = load_trained_model(device=DEVICE)
    model.eval()
    
    results = {}
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        params = torch.rand(100, 6).to(DEVICE)
        topo_ids = torch.full((100,), topo_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            waveforms, _ = model(params, topo_ids)
        
        qualities = [compute_waveform_quality(w.cpu().numpy())['quality'] for w in waveforms]
        avg_q = np.mean(qualities)
        results[topo_name] = avg_q
        
        status = "✅" if avg_q > 0.7 else "⚠️" if avg_q > 0.5 else "❌"
        print(f"  {status} {topo_name:<12}: {avg_q:.1%}")
    
    return results


def evaluate_enhanced():
    """Evaluate enhanced surrogate model."""
    print("\n📊 ENHANCED MODEL")
    print("-"*40)
    
    from retrain_surrogate_enhanced import MultiTopologySurrogate
    
    model = MultiTopologySurrogate().to(DEVICE)
    checkpoint = torch.load(
        'checkpoints/multi_topology_surrogate_enhanced.pt',
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    results = {}
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        params = torch.rand(100, 6).to(DEVICE)
        topo_ids = torch.full((100,), topo_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            waveforms, _ = model(params, topo_ids)
        
        qualities = [compute_waveform_quality(w.cpu().numpy())['quality'] for w in waveforms]
        avg_q = np.mean(qualities)
        results[topo_name] = avg_q
        
        status = "✅" if avg_q > 0.7 else "⚠️" if avg_q > 0.5 else "❌"
        print(f"  {status} {topo_name:<12}: {avg_q:.1%}")
    
    return results


def main():
    print("="*60)
    print("🔬 FINAL COMPARISON: ORIGINAL vs ENHANCED")
    print("="*60)
    
    orig_results = evaluate_original()
    enh_results = evaluate_enhanced()
    
    print("\n" + "="*60)
    print("📈 IMPROVEMENT SUMMARY")
    print("="*60)
    print(f"\n{'Topology':<12} {'Original':<12} {'Enhanced':<12} {'Change':<10}")
    print("-"*50)
    
    for topo in TOPOLOGIES:
        orig = orig_results.get(topo, 0)
        enh = enh_results.get(topo, 0)
        delta = enh - orig
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"{topo:<12} {orig:>10.1%}   {enh:>10.1%}   {arrow} {abs(delta):>6.1%}")
    
    # Overall
    orig_avg = np.mean(list(orig_results.values()))
    enh_avg = np.mean(list(enh_results.values()))
    
    print("-"*50)
    print(f"{'AVERAGE':<12} {orig_avg:>10.1%}   {enh_avg:>10.1%}   {enh_avg - orig_avg:>+7.1%}")
    
    # Challenging topology focus
    print("\n🎯 CHALLENGING TOPOLOGIES:")
    for topo in ['cuk', 'flyback', 'qr_flyback']:
        orig = orig_results.get(topo, 0)
        enh = enh_results.get(topo, 0)
        delta = enh - orig
        print(f"  {topo.upper()}: {orig:.1%} → {enh:.1%} ({delta:+.1%})")


if __name__ == '__main__':
    main()
