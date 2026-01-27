#!/usr/bin/env python3
"""
Final evaluation of all trained models with proper metrics
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.multi_topology_surrogate import MultiTopologySurrogate
from rl.ppo_agent import ActorCritic
from rl.topology_rewards import TOPOLOGY_REWARD_CONFIG

DEVICE = 'cpu'
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
BASE_PATH = Path('/Users/tushardhananjaypathak/Desktop/MLEntry')


def create_target_waveform(topology: str, params: torch.Tensor) -> torch.Tensor:
    batch_size = params.shape[0]
    t = torch.linspace(0, 1, 32)
    
    vin = params[:, 3:4] * 24 + 12
    duty = params[:, 5:6] * 0.7 + 0.1
    
    if topology == 'buck':
        vout = vin * duty
    elif topology == 'boost':
        vout = vin / (1 - duty + 0.01)
    elif topology == 'buck_boost':
        vout = -vin * duty / (1 - duty + 0.01)
    elif topology == 'sepic':
        vout = vin * duty / (1 - duty + 0.01)
    elif topology == 'cuk':
        vout = -vin * duty / (1 - duty + 0.01)
    else:  # flyback, qr_flyback
        vout = vin * 0.5 * duty / (1 - duty + 0.01)
    
    ripple = 0.03
    base = vout.expand(-1, 32)
    ripple_wave = ripple * torch.sin(2 * np.pi * 8 * t.unsqueeze(0).expand(batch_size, -1))
    return base + ripple_wave


if __name__ == "__main__":
    print("=" * 60)
    print("📊 FINAL MODEL EVALUATION")
    print("=" * 60)
    
    # Load surrogate
    print("\n📦 Loading models...")
    model = MultiTopologySurrogate(
        num_topologies=7, param_dim=6, waveform_len=32,
        embed_dim=64, hidden_dim=512
    ).to(DEVICE)
    
    ckpt = torch.load(BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt', 
                     map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"   Surrogate val_loss: {ckpt.get('val_loss', 'N/A'):.4f}")
    
    # Also load SPICE data for ground truth comparison
    data = dict(np.load(BASE_PATH / 'data' / 'spice_validated_data.npz'))
    spice_params = torch.FloatTensor(data['params'])
    spice_waveforms = torch.FloatTensor(data['waveforms'])
    spice_topo_ids = torch.LongTensor(data['topology_ids'])
    
    print("\n" + "=" * 60)
    print("📊 TOPOLOGY RESULTS")
    print("=" * 60)
    
    results = []
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        is_inv = config.get('inverted', False)
        
        # Get SPICE samples for this topology
        mask = spice_topo_ids == topo_idx
        topo_params = spice_params[mask][:100]
        topo_wf_true = spice_waveforms[mask][:100]
        topo_ids = torch.full((len(topo_params),), topo_idx, dtype=torch.long)
        
        # Surrogate predictions
        with torch.no_grad():
            pred_wf, _ = model(topo_params.to(DEVICE), topo_ids.to(DEVICE))
        
        pred_wf = pred_wf.cpu()
        
        # Metrics
        if is_inv:
            # For inverted topologies, compare absolute values
            mse = ((pred_wf.abs() - topo_wf_true.abs()) ** 2).mean().item()
            mae = (pred_wf.abs() - topo_wf_true.abs()).abs().mean().item()
            
            # Check sign accuracy
            pred_sign = (pred_wf.mean(dim=1) < 0).float()
            true_sign = (topo_wf_true.mean(dim=1) < 0).float()
            sign_acc = (pred_sign == true_sign).float().mean().item() * 100
        else:
            mse = ((pred_wf - topo_wf_true) ** 2).mean().item()
            mae = (pred_wf - topo_wf_true).abs().mean().item()
            sign_acc = 100.0  # N/A for non-inverted
        
        # Correlation
        pred_flat = pred_wf.flatten().numpy()
        true_flat = topo_wf_true.flatten().numpy()
        corr = np.corrcoef(pred_flat, true_flat)[0, 1]
        
        # DC accuracy
        pred_dc = pred_wf.mean(dim=1).abs().numpy()
        true_dc = topo_wf_true.mean(dim=1).abs().numpy()
        dc_error = np.abs(pred_dc - true_dc) / (np.abs(true_dc) + 1e-6)
        dc_accuracy = (1 - dc_error.mean()) * 100
        
        # Overall quality score
        quality = max(0, min(100, corr * 100))
        
        inv_label = "⚡INV" if is_inv else "    "
        results.append({
            'name': topo_name,
            'mse': mse,
            'mae': mae,
            'corr': corr,
            'dc_acc': dc_accuracy,
            'sign_acc': sign_acc,
            'quality': quality,
            'inverted': is_inv
        })
        
        print(f"\n   {topo_name.upper():12} {inv_label}")
        print(f"      MSE: {mse:.4f}")
        print(f"      MAE: {mae:.4f}")
        print(f"      Correlation: {corr:.4f}")
        print(f"      DC Accuracy: {dc_accuracy:.1f}%")
        if is_inv:
            print(f"      Sign Accuracy: {sign_acc:.1f}%")
        print(f"      Quality Score: {quality:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    avg_quality = np.mean([r['quality'] for r in results])
    avg_corr = np.mean([r['corr'] for r in results])
    avg_dc = np.mean([r['dc_acc'] for r in results])
    
    print(f"\n   Average Correlation: {avg_corr:.4f}")
    print(f"   Average DC Accuracy: {avg_dc:.1f}%")
    print(f"   Average Quality: {avg_quality:.1f}%")
    
    print("\n   Per-Topology Quality:")
    for r in sorted(results, key=lambda x: -x['quality']):
        inv = "⚡" if r['inverted'] else " "
        print(f"      {inv} {r['name'].upper():12} {r['quality']:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)
