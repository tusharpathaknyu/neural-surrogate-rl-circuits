"""
Inference Demo: Design circuits using the trained RL agent.

This script demonstrates how to use the trained models to:
1. Design circuits for various target waveforms
2. Compare RL-designed circuits with random designs
3. Measure inference speed

Run after completing all 3 training phases.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.forward_surrogate import ForwardSurrogate
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent


def load_models():
    """Load trained surrogate and RL agent."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load surrogate
    surrogate = ForwardSurrogate()
    ckpt = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
    surrogate.load_state_dict(ckpt['model_state_dict'])
    surrogate.eval()
    print(f"✓ Loaded surrogate (val MSE: {ckpt.get('val_loss', 'N/A'):.4f})")
    
    # Create environment and agent
    env = CircuitDesignEnv(surrogate, device=device)
    agent = PPOAgent(env, device=device)
    agent.load('checkpoints/rl_agent.pt')
    print("✓ Loaded RL agent")
    
    return surrogate, env, agent, device


def create_target_waveforms():
    """Create various target waveforms for testing."""
    t = np.linspace(0, 1, 512)
    targets = {}
    
    # 1. Clean 12V DC output
    target_12v = np.ones(512) * 12.0
    target_12v[:25] = 12.0 * (1 - np.exp(-t[:25] * 60))  # Fast rise
    targets['12V_DC'] = target_12v
    
    # 2. 5V DC output
    target_5v = np.ones(512) * 5.0
    target_5v[:25] = 5.0 * (1 - np.exp(-t[:25] * 60))
    targets['5V_DC'] = target_5v
    
    # 3. 18V DC output
    target_18v = np.ones(512) * 18.0
    target_18v[:30] = 18.0 * (1 - np.exp(-t[:30] * 50))
    targets['18V_DC'] = target_18v
    
    # 4. With ripple (0.5V p-p)
    target_ripple = np.ones(512) * 12.0 + 0.25 * np.sin(2 * np.pi * 20 * t)
    target_ripple[:25] = target_12v[:25]
    targets['12V_with_ripple'] = target_ripple
    
    return targets


def design_circuit(agent, target, name, verbose=True):
    """Design a circuit for a target waveform."""
    start_time = time.time()
    result = agent.design_circuit(target, max_steps=50)
    elapsed = (time.time() - start_time) * 1000  # ms
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Target: {name}")
        print(f"{'='*50}")
        print(f"Design time: {elapsed:.1f}ms")
        print(f"MSE: {result['mse']:.6f}")
        print(f"Steps: {result['steps']}")
        print(f"\nDesigned parameters:")
        print(f"  L     = {result['params']['L']*1e6:.2f} µH")
        print(f"  C     = {result['params']['C']*1e6:.2f} µF")
        print(f"  R_load = {result['params']['R_load']:.2f} Ω")
        print(f"  V_in  = {result['params']['V_in']:.2f} V")
        print(f"  f_sw  = {result['params']['f_sw']/1e3:.1f} kHz")
        print(f"  duty  = {result['params']['duty']*100:.1f}%")
    
    return result, elapsed


def compare_with_random(agent, env, target, n_random=100):
    """Compare RL design with random parameter search."""
    # RL design
    rl_result, rl_time = design_circuit(agent, target, "RL Design", verbose=False)
    
    # Random search
    best_random_mse = float('inf')
    random_times = []
    
    for _ in range(n_random):
        start = time.time()
        params = env._random_params()
        waveform = env._simulate(params)
        mse = np.mean((waveform - target) ** 2)
        random_times.append((time.time() - start) * 1000)
        
        if mse < best_random_mse:
            best_random_mse = mse
    
    avg_random_time = np.mean(random_times) * n_random
    
    print(f"\n{'='*50}")
    print("RL vs Random Search Comparison")
    print(f"{'='*50}")
    print(f"RL Design:")
    print(f"  MSE: {rl_result['mse']:.6f}")
    print(f"  Time: {rl_time:.1f}ms")
    print(f"\nRandom Search ({n_random} samples):")
    print(f"  Best MSE: {best_random_mse:.6f}")
    print(f"  Time: {avg_random_time:.1f}ms")
    print(f"\nRL Advantage:")
    print(f"  MSE improvement: {(best_random_mse - rl_result['mse']) / best_random_mse * 100:.1f}% better")
    print(f"  Speed improvement: {avg_random_time / rl_time:.1f}x faster")
    
    return rl_result, best_random_mse


def benchmark_speed(agent, n_designs=100):
    """Benchmark inference speed."""
    targets = [np.random.randn(512) * 2 + 12 for _ in range(n_designs)]
    
    print(f"\n{'='*50}")
    print(f"Speed Benchmark ({n_designs} designs)")
    print(f"{'='*50}")
    
    start = time.time()
    for target in targets:
        agent.design_circuit(target, max_steps=30)
    total_time = time.time() - start
    
    avg_time = total_time / n_designs * 1000
    designs_per_sec = n_designs / total_time
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per design: {avg_time:.1f}ms")
    print(f"Throughput: {designs_per_sec:.1f} designs/second")
    
    # Compare to SPICE
    spice_time_per_design = 100  # ms (approximate)
    spice_total = n_designs * spice_time_per_design / 1000
    
    print(f"\nCompared to SPICE optimization:")
    print(f"  SPICE would take: {spice_total * 100:.0f}s (assuming 100 iterations)")
    print(f"  Speedup: {(spice_total * 100) / total_time:.0f}x")


def plot_designs(agent, targets, save_path='checkpoints/inference_demo.png'):
    """Plot RL designs for all target waveforms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    t = np.linspace(0, 1, 512)
    
    for idx, (name, target) in enumerate(targets.items()):
        if idx >= 4:
            break
            
        result, _ = design_circuit(agent, target, name, verbose=False)
        
        # Get predicted waveform
        params = np.array([result['params'][n] for n in agent.env.PARAM_NAMES])
        predicted = agent.env._simulate(params)
        
        ax = axes[idx]
        ax.plot(t, target, 'b-', label='Target', linewidth=2, alpha=0.7)
        ax.plot(t, predicted, 'r--', label='RL Design', linewidth=2)
        ax.set_xlabel('Time (normalized)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'{name}\nMSE: {result["mse"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved design plots to {save_path}")


def main():
    print("="*60)
    print("INFERENCE DEMO: Neural Surrogate + RL Circuit Design")
    print("="*60)
    
    # Load models
    surrogate, env, agent, device = load_models()
    
    # Create target waveforms
    targets = create_target_waveforms()
    
    # Design circuits for each target
    print("\n" + "="*60)
    print("Designing circuits for various targets...")
    print("="*60)
    
    for name, target in targets.items():
        design_circuit(agent, target, name)
    
    # Compare with random search
    compare_with_random(agent, env, targets['12V_DC'])
    
    # Benchmark speed
    benchmark_speed(agent, n_designs=50)
    
    # Plot all designs
    plot_designs(agent, targets)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
