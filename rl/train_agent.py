"""
Phase C: Train the RL Agent.

This script:
1. Loads the trained surrogate model (from Phase B)
2. Creates the RL environment
3. Trains PPO agent to design circuits

The key insight: Training uses SURROGATE (0.001ms per simulation)
instead of ngspice (100ms per simulation) = 100,000x speedup!
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from models.forward_surrogate import ForwardSurrogate
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent


def load_surrogate(model_path: str) -> ForwardSurrogate:
    """Load trained surrogate model."""
    model = ForwardSurrogate()
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded surrogate from {model_path}")
    else:
        print(f"⚠ No trained surrogate found at {model_path}")
        print("  Using random weights - results will be meaningless!")
        print("  Run Phase B (train_surrogate.py) first!")
    
    model.eval()
    return model


def create_target_waveform(output_points: int = 512) -> np.ndarray:
    """
    Create a target waveform for the RL agent to aim for.
    
    For a buck converter, we want:
    - Stable DC output at target voltage
    - Minimal ripple
    - Fast rise time
    - No overshoot
    """
    t = np.linspace(0, 1, output_points)
    
    # Target: Clean 12V DC output with fast rise
    target_voltage = 12.0
    rise_time = 0.05  # 5% of period
    
    # Create ideal step response
    target = np.ones(output_points) * target_voltage
    
    # Add realistic rise (smooth step)
    rise_mask = t < rise_time
    target[rise_mask] = target_voltage * (1 - np.exp(-t[rise_mask] * 60))
    
    # Small acceptable ripple (0.1V peak-to-peak)
    ripple = 0.05 * np.sin(2 * np.pi * 10 * t)
    target = target + ripple
    
    return target


def test_agent(agent: PPOAgent, n_tests: int = 10):
    """Test trained agent on design tasks."""
    print("\n" + "="*60)
    print("Testing Trained Agent")
    print("="*60)
    
    results = []
    
    for i in range(n_tests):
        # Create random target
        target = create_target_waveform()
        target += np.random.randn(len(target)) * 0.1  # Add some noise
        
        # Design circuit
        result = agent.design_circuit(target, max_steps=50)
        results.append(result)
        
        print(f"\nTest {i+1}:")
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Designed parameters:")
        for name, val in result['params'].items():
            print(f"    {name}: {val:.4f}")
    
    # Summary
    mean_mse = np.mean([r['mse'] for r in results])
    mean_steps = np.mean([r['steps'] for r in results])
    
    print("\n" + "-"*40)
    print(f"Average MSE: {mean_mse:.6f}")
    print(f"Average Steps: {mean_steps:.1f}")
    
    return results


def plot_design_example(agent: PPOAgent, save_path: str):
    """Plot an example circuit design."""
    target = create_target_waveform()
    result = agent.design_circuit(target, max_steps=50)
    
    # Get predicted waveform
    params = np.array([result['params'][name] for name in agent.env.PARAM_NAMES])
    params_tensor = torch.FloatTensor(params).unsqueeze(0)
    
    with torch.no_grad():
        predicted = agent.env.surrogate(params_tensor).squeeze().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    t = np.linspace(0, 1, len(target))
    
    axes[0].plot(t, target, 'b-', label='Target', linewidth=2)
    axes[0].plot(t, predicted, 'r--', label='Designed', linewidth=2)
    axes[0].set_xlabel('Time (normalized)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title(f'RL-Designed Circuit (MSE: {result["mse"]:.6f})')
    axes[0].legend()
    axes[0].grid(True)
    
    # Show parameters
    param_text = "Designed Parameters:\n"
    param_text += f"L = {result['params']['L']*1e6:.1f} µH\n"
    param_text += f"C = {result['params']['C']*1e6:.1f} µF\n"
    param_text += f"R_load = {result['params']['R_load']:.1f} Ω\n"
    param_text += f"V_in = {result['params']['V_in']:.1f} V\n"
    param_text += f"f_sw = {result['params']['f_sw']/1e3:.0f} kHz\n"
    param_text += f"duty = {result['params']['duty']:.2f}"
    
    axes[1].text(0.1, 0.5, param_text, fontsize=14, family='monospace',
                 verticalalignment='center', transform=axes[1].transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].axis('off')
    axes[1].set_title('Designed Circuit Parameters')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved design example to {save_path}")


def main():
    print("="*60)
    print("PHASE C: Train RL Agent for Circuit Design")
    print("="*60)
    print()
    
    # Paths
    project_root = Path(__file__).parent.parent
    surrogate_path = project_root / 'checkpoints' / 'best_model.pt'
    agent_path = project_root / 'checkpoints' / 'rl_agent.pt'
    
    # Load surrogate
    print("Loading surrogate model...")
    surrogate = load_surrogate(str(surrogate_path))
    
    # Create environment
    print("\nCreating RL environment...")
    env = CircuitDesignEnv(
        surrogate=surrogate,
        max_steps=50,
        target_waveforms=create_target_waveform(),
    )
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action dimension: {env.action_dim}")
    
    # Create agent
    print("\nCreating PPO agent...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    agent = PPOAgent(
        env=env,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        device=device,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print()
    
    total_timesteps = 500_000  # Full training for good exploration
    
    history = agent.train(
        total_timesteps=total_timesteps,
        n_steps_per_update=2048,
        n_epochs=10,
        batch_size=64,
        log_interval=10,
        save_path=str(agent_path),
    )
    
    # Test
    test_results = test_agent(agent, n_tests=5)
    
    # Plot example
    plot_path = project_root / 'checkpoints' / 'design_example.png'
    plot_design_example(agent, str(plot_path))
    
    print("\n" + "="*60)
    print("Phase C Complete!")
    print("="*60)
    print(f"\nAgent saved to: {agent_path}")
    print(f"\nThe RL agent can now design circuits in milliseconds!")
    print("Traditional SPICE would take hours for the same optimization.")


if __name__ == '__main__':
    main()
