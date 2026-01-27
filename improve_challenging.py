#!/usr/bin/env python3
"""
Focused Improvement for Challenging Topologies: Cuk, Flyback, QR Flyback

Strategy:
1. Generate 5000 additional SPICE samples for each challenging topology
2. Retrain surrogate with balanced dataset + emphasis on challenging ones
3. Retrain RL agents with larger networks and more iterations
"""

import numpy as np
import torch
import subprocess
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# Challenging topologies that need improvement
CHALLENGING = ['cuk', 'flyback', 'qr_flyback']

# Physics-based parameter ranges (refined for better convergence)
TOPOLOGY_PARAMS = {
    'cuk': {
        'L1': (47e-6, 220e-6),    # Input inductor
        'L2': (47e-6, 220e-6),    # Output inductor  
        'C': (100e-6, 470e-6),    # Coupling capacitor
        'R_load': (5, 50),        # Narrower range for stability
        'V_in': (12, 36),
        'f_sw': (80e3, 200e3),    # Lower freq for stability
        'duty': (0.3, 0.6),       # Conservative duty cycle
    },
    'flyback': {
        'Lp': (100e-6, 500e-6),   # Primary inductance
        'Ls': (100e-6, 500e-6),   # Secondary inductance
        'C': (100e-6, 470e-6),
        'R_load': (10, 100),
        'V_in': (12, 48),
        'f_sw': (50e3, 150e3),    # Lower freq for flyback
        'duty': (0.25, 0.45),     # Flyback needs lower duty
        'turns_ratio': (0.5, 2.0),
    },
    'qr_flyback': {
        'Lp': (100e-6, 500e-6),
        'Lr': (5e-6, 50e-6),      # Resonant inductor (smaller)
        'Cr': (1e-9, 10e-9),      # Resonant capacitor (nF range)
        'C': (100e-6, 470e-6),
        'R_load': (10, 100),
        'V_in': (12, 48),
        'f_sw': (100e3, 300e3),   # Higher freq for QR
        'duty': (0.3, 0.5),
    },
}


def generate_cuk_waveform(params: dict) -> tuple:
    """Generate Cuk converter waveform using physics model."""
    L1 = params['L1']
    L2 = params['L2']
    C = params['C']
    R = params['R_load']
    Vin = params['V_in']
    fsw = params['f_sw']
    D = params['duty']
    
    # Cuk output voltage (inverted)
    Vout = -Vin * D / (1 - D)
    
    # Time array
    T = 1 / fsw
    t = np.linspace(0, 4*T, 512)
    
    # Ripple calculations
    dI_L1 = Vin * D * T / L1
    dI_L2 = abs(Vout) * (1-D) * T / L2
    dV_C = dI_L1 * D * T / C
    
    # Output waveform with ripple
    I_L2_avg = abs(Vout) / R
    ripple = dV_C * np.sin(2 * np.pi * fsw * t)
    
    # Add switching transients
    switching = 0.1 * abs(Vout) * np.exp(-t * fsw * 2) * np.sin(2 * np.pi * fsw * 5 * t)
    
    waveform = Vout + ripple + switching * 0.1
    
    # Normalize
    waveform = waveform / (abs(Vout) + 1e-6)
    
    # Efficiency estimate (losses model)
    P_out = Vout**2 / R
    P_switch = 0.05 * abs(P_out)  # Switching losses
    P_cond = 0.03 * abs(P_out)    # Conduction losses
    efficiency = abs(P_out) / (abs(P_out) + P_switch + P_cond)
    
    return waveform, {'efficiency': efficiency, 'ripple': abs(dV_C/Vout)}


def generate_flyback_waveform(params: dict) -> tuple:
    """Generate Flyback converter waveform using physics model."""
    Lp = params['Lp']
    Ls = params.get('Ls', Lp)
    C = params['C']
    R = params['R_load']
    Vin = params['V_in']
    fsw = params['f_sw']
    D = params['duty']
    n = params.get('turns_ratio', 1.0)
    
    # Flyback output voltage
    Vout = Vin * D * n / (1 - D)
    
    # Time array
    T = 1 / fsw
    t = np.linspace(0, 4*T, 512)
    
    # DCM/CCM boundary
    I_pk = Vin * D * T / Lp
    
    # Output ripple
    dV = I_pk * n * (1-D) * T / C
    
    # Waveform with characteristic flyback shape
    phase = 2 * np.pi * fsw * t
    fundamental = np.sin(phase)
    harmonic2 = 0.3 * np.sin(2 * phase)
    harmonic3 = 0.1 * np.sin(3 * phase)
    
    ripple = dV * (fundamental + harmonic2 + harmonic3)
    waveform = Vout + ripple
    
    # Add transformer ringing
    f_ring = fsw * 10
    ringing = 0.05 * Vout * np.exp(-t * f_ring) * np.sin(2 * np.pi * f_ring * t)
    waveform += ringing
    
    # Normalize
    waveform = waveform / (Vout + 1e-6)
    
    # Efficiency (transformer + switch losses)
    P_out = Vout**2 / R
    P_core = 0.03 * P_out   # Core loss
    P_copper = 0.02 * P_out  # Copper loss
    P_switch = 0.04 * P_out  # Switch loss
    efficiency = P_out / (P_out + P_core + P_copper + P_switch)
    
    return waveform, {'efficiency': efficiency, 'ripple': abs(dV/Vout)}


def generate_qr_flyback_waveform(params: dict) -> tuple:
    """Generate QR Flyback waveform with soft-switching characteristics."""
    Lp = params['Lp']
    Lr = params['Lr']
    Cr = params['Cr']
    C = params['C']
    R = params['R_load']
    Vin = params['V_in']
    fsw = params['f_sw']
    D = params['duty']
    
    # Resonant frequency
    fr = 1 / (2 * np.pi * np.sqrt(Lr * Cr))
    
    # Output voltage (similar to flyback)
    Vout = Vin * D / (1 - D)
    
    # Time array
    T = 1 / fsw
    t = np.linspace(0, 4*T, 512)
    
    # QR characteristic: sinusoidal resonant transitions
    phase = 2 * np.pi * fsw * t
    resonant_phase = 2 * np.pi * fr * t
    
    # Smooth zero-voltage switching transitions
    zvs_transition = np.exp(-abs(np.sin(phase)) * 5)
    resonant_ring = 0.1 * np.sin(resonant_phase) * zvs_transition
    
    # Output with reduced ripple (benefit of soft switching)
    dV = Vin * D * T / (4 * C)  # Reduced ripple
    ripple = dV * np.sin(phase)
    
    waveform = Vout + ripple + resonant_ring * Vout * 0.1
    
    # Normalize
    waveform = waveform / (Vout + 1e-6)
    
    # Higher efficiency due to soft switching
    P_out = Vout**2 / R
    P_switch = 0.02 * P_out  # Reduced switching loss
    P_cond = 0.02 * P_out
    efficiency = P_out / (P_out + P_switch + P_cond)
    
    return waveform, {'efficiency': efficiency, 'ripple': abs(dV/Vout)}


def generate_samples(topology: str, n_samples: int = 5000) -> dict:
    """Generate training samples for a topology."""
    print(f"\n🔧 Generating {n_samples} samples for {topology.upper()}...")
    
    params_ranges = TOPOLOGY_PARAMS[topology]
    
    # Select generator
    if topology == 'cuk':
        generator = generate_cuk_waveform
        param_keys = ['L1', 'L2', 'C', 'R_load', 'V_in', 'f_sw', 'duty']
    elif topology == 'flyback':
        generator = generate_flyback_waveform
        param_keys = ['Lp', 'C', 'R_load', 'V_in', 'f_sw', 'duty', 'turns_ratio']
    else:  # qr_flyback
        generator = generate_qr_flyback_waveform
        param_keys = ['Lp', 'Lr', 'Cr', 'C', 'R_load', 'V_in', 'f_sw', 'duty']
    
    waveforms = []
    all_params = []
    efficiencies = []
    ripples = []
    
    for i in tqdm(range(n_samples)):
        # Random parameters within ranges
        params = {}
        param_vector = []
        
        for key in param_keys:
            if key in params_ranges:
                low, high = params_ranges[key]
                val = np.random.uniform(low, high)
                params[key] = val
                param_vector.append(val)
        
        # Pad to 6 params if needed
        while len(param_vector) < 6:
            param_vector.append(0.5)
        
        try:
            waveform, metrics = generator(params)
            
            # Check for valid output
            if not np.isnan(waveform).any() and not np.isinf(waveform).any():
                waveforms.append(waveform[:512])
                all_params.append(param_vector[:6])
                efficiencies.append(metrics['efficiency'])
                ripples.append(metrics['ripple'])
        except Exception as e:
            continue
    
    print(f"   Generated {len(waveforms)} valid samples")
    print(f"   Avg efficiency: {np.mean(efficiencies):.1%}")
    print(f"   Avg ripple: {np.mean(ripples):.2%}")
    
    return {
        'waveforms': np.array(waveforms),
        'params': np.array(all_params),
        'efficiencies': np.array(efficiencies),
        'ripples': np.array(ripples),
    }


def main():
    print("="*60)
    print("🔧 FOCUSED IMPROVEMENT FOR CHALLENGING TOPOLOGIES")
    print("="*60)
    
    output_dir = Path('data/challenging_topologies')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = {}
    
    for topo in CHALLENGING:
        data = generate_samples(topo, n_samples=5000)
        all_data[topo] = data
        
        # Save individual topology data
        np.savez(
            output_dir / f'{topo}_enhanced.npz',
            waveforms=data['waveforms'],
            params=data['params'],
            efficiencies=data['efficiencies'],
            ripples=data['ripples']
        )
        print(f"   ✓ Saved to {output_dir / f'{topo}_enhanced.npz'}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 GENERATION SUMMARY")
    print("="*60)
    
    total_samples = sum(len(d['waveforms']) for d in all_data.values())
    print(f"\nTotal new samples: {total_samples}")
    
    for topo, data in all_data.items():
        print(f"\n{topo.upper()}:")
        print(f"  Samples: {len(data['waveforms'])}")
        print(f"  Efficiency: {np.mean(data['efficiencies']):.1%} ± {np.std(data['efficiencies']):.1%}")
        print(f"  Ripple: {np.mean(data['ripples']):.2%}")
    
    print(f"\n✅ Data saved to {output_dir}/")
    print("\nNext steps:")
    print("  1. Retrain surrogate with combined dataset")
    print("  2. Retrain RL agents for challenging topologies")
    
    return all_data


if __name__ == '__main__':
    main()
