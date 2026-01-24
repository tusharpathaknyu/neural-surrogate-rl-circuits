"""
Train RL Agent on Multi-Topology Surrogate.

This script trains a PPO agent that can design circuits for all 6 topologies:
- Buck, Boost, Buck-Boost, SEPIC, Ćuk, Flyback

Key improvements over single-topology training:
1. Uses the 6-topology surrogate (30k training samples)
2. Samples different topologies during training
3. Topology-aware reward function
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json

from models.multi_topology_surrogate import MultiTopologySurrogate, load_trained_model
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent


# Device
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Topologies to train on
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback']


def create_target_waveform(topology: str, v_in: float = 12.0, 
                           duty: float = 0.5, output_points: int = 512) -> np.ndarray:
    """
    Create a target waveform for the RL agent based on topology.
    
    Different topologies have different ideal output characteristics.
    """
    t = np.linspace(0, 1, output_points)
    rise_time = 0.05
    
    # Calculate expected output voltage based on topology
    if topology == 'buck':
        # Vout = Vin * D
        v_out = v_in * duty
    elif topology == 'boost':
        # Vout = Vin / (1 - D)
        v_out = v_in / (1 - duty + 0.01)
    elif topology in ['buck_boost', 'cuk']:
        # Vout = Vin * D / (1 - D) (inverted)
        v_out = abs(v_in * duty / (1 - duty + 0.01))
    elif topology == 'sepic':
        # Vout = Vin * D / (1 - D) (non-inverted)
        v_out = v_in * duty / (1 - duty + 0.01)
    elif topology == 'flyback':
        # Vout = Vin * D / (1 - D) * N (assuming N=1)
        v_out = v_in * duty / (1 - duty + 0.01)
    else:
        v_out = v_in * duty
    
    # Clamp to reasonable range
    v_out = np.clip(v_out, 1, 60)
    
    # Create ideal waveform with fast rise and minimal ripple
    target = np.ones(output_points) * v_out
    
    # Add realistic rise
    rise_mask = t < rise_time
    target[rise_mask] = v_out * (1 - np.exp(-t[rise_mask] * 60))
    
    # Add small acceptable ripple (1% of output)
    ripple = 0.01 * v_out * np.sin(2 * np.pi * 10 * t)
    target = target + ripple
    
    return target.astype(np.float32)


class MultiTopologyEnv(CircuitDesignEnv):
    """Extended environment that randomly samples topologies during training."""
    
    def __init__(self, surrogate, device='cpu', **kwargs):
        super().__init__(surrogate, device=device, **kwargs)
        self.current_topology_idx = 0
        self.is_multi_topology = True
    
    def reset(self, topology: str = None):
        """Reset with optional topology specification."""
        # If no topology specified, sample randomly
        if topology is None:
            self.current_topology_idx = np.random.randint(len(TOPOLOGIES))
            self.topology = TOPOLOGIES[self.current_topology_idx]
        else:
            self.topology = topology
            self.current_topology_idx = TOPOLOGIES.index(topology)
        
        # Random input voltage and duty cycle for diversity
        v_in = np.random.uniform(8, 36)
        duty = np.random.uniform(0.3, 0.7)
        
        # Create target waveform for this topology
        self.target_waveform = create_target_waveform(
            self.topology, v_in=v_in, duty=duty
        )
        
        # Random initial parameters (normalized)
        self.current_params = np.array([
            np.random.uniform(20e-6, 200e-6),   # L
            np.random.uniform(100e-6, 470e-6),  # C
            np.random.uniform(5, 50),           # R_load
            v_in,                                # V_in
            np.random.uniform(100e3, 300e3),    # f_sw
            duty,                                # duty
        ], dtype=np.float32)
        
        self.current_step = 0
        self.prev_mse = None
        
        return self._get_state()


def train_multi_topology_agent(
    n_iterations: int = 500,
    steps_per_iter: int = 4096,
    save_every: int = 100,
):
    """Train PPO agent on multi-topology surrogate with improved hyperparameters."""
    
    print("\n" + "="*60)
    print("Training Multi-Topology RL Agent (Extended)")
    print("="*60)
    
    # Load multi-topology surrogate
    print("\nLoading multi-topology surrogate...")
    surrogate = load_trained_model(device=DEVICE)
    print(f"  ✓ Loaded surrogate with {sum(p.numel() for p in surrogate.parameters()):,} params")
    
    # Create multi-topology environment
    env = MultiTopologyEnv(surrogate, device=DEVICE)
    print(f"  ✓ Created environment with {len(TOPOLOGIES)} topologies")
    
    # Create agent with improved hyperparameters
    agent = PPOAgent(
        env,
        hidden_dim=512,          # Larger network
        lr=1e-4,                 # Lower learning rate for stability
        gamma=0.995,             # Longer horizon
        gae_lambda=0.97,         # Better advantage estimation
        clip_epsilon=0.15,       # Tighter clipping
        entropy_coef=0.005,      # Less exploration as training progresses
        value_coef=0.5,
        device=DEVICE,
    )
    print(f"  ✓ Created PPO agent (hidden=512, lr=1e-4)")
    
    # Training tracking
    all_rewards = []
    all_mses = []
    topology_performance = {t: [] for t in TOPOLOGIES}
    best_mean_reward = float('-inf')
    
    # Training loop
    print(f"\nTraining for {n_iterations} iterations ({n_iterations * steps_per_iter:,} total steps)")
    print("-"*60)
    
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Collect rollouts
        rollout = agent.collect_rollouts(steps_per_iter)
        
        # Update policy
        update_info = agent.update(rollout, n_epochs=10, batch_size=64)
        
        # Track per-topology performance
        if len(agent.episode_rewards) > 0:
            all_rewards.extend(agent.episode_rewards[-10:])
            all_mses.extend(agent.episode_mses[-10:])
        
        # Logging
        if (iteration + 1) % 10 == 0:
            recent_rewards = agent.episode_rewards[-50:] if agent.episode_rewards else [0]
            recent_mses = agent.episode_mses[-50:] if agent.episode_mses else [float('inf')]
            
            mean_reward = np.mean(recent_rewards)
            mean_mse = np.mean(recent_mses)
            
            tqdm.write(f"Iter {iteration+1:4d} | Reward: {mean_reward:7.2f} | MSE: {mean_mse:.4f}")
            
            # Save best
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                agent.save('checkpoints/multi_topo_rl_agent_best.pt')
        
        # Periodic save
        if (iteration + 1) % save_every == 0:
            agent.save(f'checkpoints/multi_topo_rl_agent_iter{iteration+1}.pt')
    
    # Final save
    agent.save('checkpoints/multi_topo_rl_agent.pt')
    print(f"\n✓ Training complete! Best reward: {best_mean_reward:.2f}")
    
    # Save training history
    history = {
        'rewards': [float(r) for r in all_rewards],
        'mses': [float(m) for m in all_mses if m < float('inf')],
        'best_reward': float(best_mean_reward),
        'n_iterations': n_iterations,
        'steps_per_iter': steps_per_iter,
        'topologies': TOPOLOGIES,
    }
    with open('checkpoints/multi_topo_rl_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(all_rewards, all_mses)
    
    return agent


def plot_training_curves(rewards, mses):
    """Plot training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Smooth rewards
    if len(rewards) > 0:
        window = min(50, len(rewards) // 5 + 1)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, 'b-', linewidth=1)
        axes[0].set_title('Episode Reward (Smoothed)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True, alpha=0.3)
    
    # Smooth MSEs
    if len(mses) > 0:
        window = min(50, len(mses) // 5 + 1)
        smoothed = np.convolve(mses, np.ones(window)/window, mode='valid')
        axes[1].semilogy(smoothed, 'r-', linewidth=1)
        axes[1].set_title('Design MSE (Smoothed)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('MSE (log)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoints/multi_topo_rl_training.png', dpi=150)
    plt.close()
    print("  ✓ Saved training curves to checkpoints/multi_topo_rl_training.png")


def test_agent_all_topologies(agent_path: str = 'checkpoints/multi_topo_rl_agent.pt'):
    """Test trained agent on all topologies."""
    
    print("\n" + "="*60)
    print("Testing Agent on All Topologies")
    print("="*60)
    
    # Load surrogate and agent
    surrogate = load_trained_model(device=DEVICE)
    env = MultiTopologyEnv(surrogate, device=DEVICE)
    
    agent = PPOAgent(env, device=DEVICE)
    agent.load(agent_path)
    
    results = {}
    
    for topology in TOPOLOGIES:
        mses = []
        for _ in range(10):
            state = env.reset(topology=topology)
            
            for step in range(50):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action, _, _ = agent.policy.get_action(state_tensor, deterministic=True)
                action_np = action.cpu().numpy().squeeze()
                state, _, done, info = env.step(action_np)
                if done:
                    break
            
            mses.append(info['mse'])
        
        mean_mse = np.mean(mses)
        std_mse = np.std(mses)
        results[topology] = {'mean': mean_mse, 'std': std_mse}
        print(f"  {topology:12s}: MSE = {mean_mse:.4f} ± {std_mse:.4f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    
    if args.test_only:
        test_agent_all_topologies()
    else:
        agent = train_multi_topology_agent(
            n_iterations=args.iterations,
            steps_per_iter=args.steps,
        )
        test_agent_all_topologies()
