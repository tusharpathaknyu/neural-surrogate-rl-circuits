"""
Validation: Compare surrogate predictions against real SPICE simulations.

This script validates the trained models by:
1. Running real ngspice simulations for test circuits
2. Comparing against surrogate predictions
3. Verifying RL-designed circuits work in real SPICE

This is the GROUND TRUTH check - ensuring our surrogate
actually matches real physics!
"""

import sys
from pathlib import Path
import subprocess
import tempfile
import time

sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.forward_surrogate import ForwardSurrogate
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent


# SPICE netlist template (same as training)
NETLIST_TEMPLATE = """* Buck Converter Validation
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

* Input voltage
Vin input 0 DC {{V_val}}

* Switch (voltage-controlled)
.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})
S1 input sw_node ctrl 0 sw_model

* Diode
D1 0 sw_node dmodel
.model dmodel d is=1e-14 n=1.05

* LC filter
L1 sw_node output {{L_val}} ic=0
C1 output 0 {{C_val}} ic=0

* Load
Rload output 0 {{R_val}}

* Analysis
.tran 1u 10m 5m uic

* Output
.control
run
set filetype=ascii
wrdata /tmp/spice_output.txt v(output)
.endc
.end
"""


def run_spice_simulation(params, timeout=10):
    """Run actual ngspice simulation and return waveform."""
    L, C, R_load, V_in, f_sw, duty = params
    
    netlist = NETLIST_TEMPLATE.format(
        L=L, C=C, R_load=R_load, V_in=V_in, f_sw=f_sw, duty=duty
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
        f.write(netlist)
        netlist_path = f.name
    
    try:
        start = time.time()
        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        spice_time = (time.time() - start) * 1000
        
        # Parse output
        output_file = '/tmp/spice_output.txt'
        if Path(output_file).exists():
            data = np.loadtxt(output_file)
            if len(data.shape) == 2 and data.shape[1] >= 2:
                waveform = data[:, 1]
                # Resample to 512 points
                indices = np.linspace(0, len(waveform)-1, 512).astype(int)
                waveform = waveform[indices]
                return waveform, spice_time
    except Exception as e:
        print(f"SPICE error: {e}")
    
    return None, 0


def validate_surrogate(surrogate, device, n_tests=20):
    """Compare surrogate predictions against real SPICE."""
    print("\n" + "="*60)
    print("Validating Surrogate Against Real SPICE")
    print("="*60)
    
    param_bounds = {
        'L': (10e-6, 100e-6),
        'C': (47e-6, 470e-6),
        'R_load': (2, 50),
        'V_in': (10, 24),
        'f_sw': (50e3, 500e3),
        'duty': (0.2, 0.8),
    }
    
    results = []
    surrogate_times = []
    spice_times = []
    
    for i in range(n_tests):
        # Random parameters
        params = np.array([
            np.random.uniform(*param_bounds['L']),
            np.random.uniform(*param_bounds['C']),
            np.random.uniform(*param_bounds['R_load']),
            np.random.uniform(*param_bounds['V_in']),
            np.random.uniform(*param_bounds['f_sw']),
            np.random.uniform(*param_bounds['duty']),
        ])
        
        # Surrogate prediction
        start = time.time()
        with torch.no_grad():
            params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(device)
            pred = surrogate(params_tensor, normalize=True).cpu().numpy().squeeze()
        surrogate_time = (time.time() - start) * 1000
        surrogate_times.append(surrogate_time)
        
        # Real SPICE simulation
        spice_wave, spice_time = run_spice_simulation(params)
        spice_times.append(spice_time)
        
        if spice_wave is not None:
            mse = np.mean((pred - spice_wave) ** 2)
            mae = np.mean(np.abs(pred - spice_wave))
            results.append({
                'params': params,
                'mse': mse,
                'mae': mae,
                'surrogate': pred,
                'spice': spice_wave,
            })
            print(f"Test {i+1}/{n_tests}: MSE={mse:.4f}, MAE={mae:.4f}")
        else:
            print(f"Test {i+1}/{n_tests}: SPICE failed")
    
    # Summary
    if results:
        avg_mse = np.mean([r['mse'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        avg_surrogate_time = np.mean(surrogate_times)
        avg_spice_time = np.mean([t for t in spice_times if t > 0])
        
        print(f"\n{'='*40}")
        print("VALIDATION SUMMARY")
        print(f"{'='*40}")
        print(f"Tests passed: {len(results)}/{n_tests}")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average MAE: {avg_mae:.4f} V")
        print(f"Surrogate time: {avg_surrogate_time:.3f}ms")
        print(f"SPICE time: {avg_spice_time:.1f}ms")
        print(f"Speedup: {avg_spice_time/avg_surrogate_time:.0f}x")
        
        return results
    
    return []


def validate_rl_design(agent, n_tests=5):
    """Verify RL-designed circuits work in real SPICE."""
    print("\n" + "="*60)
    print("Validating RL Designs in Real SPICE")
    print("="*60)
    
    t = np.linspace(0, 1, 512)
    
    for i in range(n_tests):
        # Create target
        target_v = np.random.uniform(8, 18)
        target = np.ones(512) * target_v
        target[:25] = target_v * (1 - np.exp(-t[:25] * 60))
        
        # RL design
        result = agent.design_circuit(target, max_steps=50)
        params = np.array([result['params'][n] for n in agent.env.PARAM_NAMES])
        
        # Surrogate prediction
        predicted = agent.env._simulate(params)
        
        # Real SPICE
        spice_wave, spice_time = run_spice_simulation(params)
        
        print(f"\nTest {i+1}: Target {target_v:.1f}V DC")
        print(f"  RL design MSE (surrogate): {result['mse']:.4f}")
        
        if spice_wave is not None:
            spice_mse = np.mean((target - spice_wave) ** 2)
            surrogate_vs_spice = np.mean((predicted - spice_wave) ** 2)
            print(f"  RL design MSE (real SPICE): {spice_mse:.4f}")
            print(f"  Surrogate vs SPICE: {surrogate_vs_spice:.4f}")
            print(f"  SPICE output mean: {np.mean(spice_wave):.2f}V (target: {target_v:.1f}V)")
        else:
            print("  SPICE simulation failed")


def plot_validation(results, save_path='checkpoints/validation.png'):
    """Plot surrogate vs SPICE comparison."""
    if not results:
        return
    
    n_plots = min(4, len(results))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    t = np.linspace(0, 1, 512)
    
    for i in range(n_plots):
        r = results[i]
        ax = axes[i]
        
        ax.plot(t, r['spice'], 'b-', label='Real SPICE', linewidth=2)
        ax.plot(t, r['surrogate'], 'r--', label='Surrogate', linewidth=2)
        ax.set_xlabel('Time (normalized)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Validation {i+1}\nMSE: {r["mse"]:.4f}, MAE: {r["mae"]:.4f}V')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved validation plots to {save_path}")


def main():
    print("="*60)
    print("VALIDATION: Checking Models Against Real SPICE")
    print("="*60)
    
    # Check ngspice
    try:
        subprocess.run(['ngspice', '--version'], capture_output=True, check=True)
        print("✓ ngspice found")
    except:
        print("✗ ngspice not found! Install with: brew install ngspice")
        return
    
    # Load models
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    surrogate = ForwardSurrogate()
    ckpt = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
    surrogate.load_state_dict(ckpt['model_state_dict'])
    surrogate.to(device)
    surrogate.eval()
    print("✓ Loaded surrogate")
    
    env = CircuitDesignEnv(surrogate, device=device)
    agent = PPOAgent(env, device=device)
    agent.load('checkpoints/rl_agent.pt')
    print("✓ Loaded RL agent")
    
    # Validate surrogate
    results = validate_surrogate(surrogate, device, n_tests=10)
    
    # Plot validation
    plot_validation(results)
    
    # Validate RL designs
    validate_rl_design(agent, n_tests=3)
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
