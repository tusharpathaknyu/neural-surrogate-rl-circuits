"""
Train Separate RL Agents Per Topology.

Each topology gets its own specialized agent for optimal performance.
This approach typically gives 2-5x better results than a single multi-topology agent.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

from models.multi_topology_surrogate import load_trained_model
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent


# Device - prioritize CUDA (Colab), then MPS (Mac), then CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

# All topologies
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback']

# Per-topology hyperparameters (tuned for each topology's characteristics)
TOPOLOGY_CONFIG = {
    'buck': {
        'hidden_dim': 256,
        'lr': 3e-4,
        'n_iterations': 300,
        'steps_per_iter': 2048,
        'description': 'Simple step-down, easiest to optimize',
    },
    'boost': {
        'hidden_dim': 512,
        'lr': 1e-4,
        'n_iterations': 500,
        'steps_per_iter': 4096,
        'description': 'Step-up, needs more training due to instability at high duty',
    },
    'buck_boost': {
        'hidden_dim': 256,
        'lr': 2e-4,
        'n_iterations': 400,
        'steps_per_iter': 2048,
        'description': 'Inverting topology, moderate complexity',
    },
    'sepic': {
        'hidden_dim': 512,
        'lr': 1e-4,
        'n_iterations': 500,
        'steps_per_iter': 4096,
        'description': 'Non-inverting buck-boost, needs coupled inductor handling',
    },
    'cuk': {
        'hidden_dim': 256,
        'lr': 2e-4,
        'n_iterations': 400,
        'steps_per_iter': 2048,
        'description': 'Continuous current, good for low ripple',
    },
    'flyback': {
        'hidden_dim': 512,
        'lr': 1e-4,
        'n_iterations': 500,
        'steps_per_iter': 4096,
        'description': 'Isolated, complex transformer dynamics',
    },
}


def create_target_waveform(topology: str, v_in: float = 12.0, 
                           duty: float = 0.5, output_points: int = 512) -> np.ndarray:
    """Create target waveform for given topology."""
    t = np.linspace(0, 1, output_points)
    rise_time = 0.05
    
    # Calculate expected output voltage
    if topology == 'buck':
        v_out = v_in * duty
    elif topology == 'boost':
        v_out = v_in / (1 - duty + 0.01)
    elif topology in ['buck_boost', 'cuk']:
        v_out = abs(v_in * duty / (1 - duty + 0.01))
    elif topology == 'sepic':
        v_out = v_in * duty / (1 - duty + 0.01)
    elif topology == 'flyback':
        v_out = v_in * duty / (1 - duty + 0.01)
    else:
        v_out = v_in * duty
    
    v_out = np.clip(v_out, 1, 60)
    
    # Create ideal waveform
    target = np.ones(output_points) * v_out
    rise_mask = t < rise_time
    target[rise_mask] = v_out * (1 - np.exp(-t[rise_mask] * 60))
    
    # Add small ripple
    ripple = 0.01 * v_out * np.sin(2 * np.pi * 10 * t)
    target = target + ripple
    
    return target.astype(np.float32)


class SingleTopologyEnv(CircuitDesignEnv):
    """Environment for training on a single topology."""
    
    def __init__(self, surrogate, topology: str, device='cpu', **kwargs):
        super().__init__(surrogate, device=device, **kwargs)
        self.topology = topology
        self.topology_idx = TOPOLOGIES.index(topology)
        self.is_multi_topology = True  # Enables topology_id in simulation
    
    def reset(self):
        """Reset environment for this topology."""
        # Random input voltage and duty cycle
        v_in = np.random.uniform(8, 36)
        duty = np.random.uniform(0.3, 0.7)
        
        # Create target for this topology
        self.target_waveform = create_target_waveform(
            self.topology, v_in=v_in, duty=duty
        )
        
        # Random initial parameters
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


def train_single_topology(topology: str, surrogate, quick_mode: bool = False):
    """Train an agent for a single topology."""
    
    config = TOPOLOGY_CONFIG[topology]
    
    # Reduce iterations in quick mode
    n_iterations = config['n_iterations'] // 5 if quick_mode else config['n_iterations']
    steps_per_iter = config['steps_per_iter'] // 2 if quick_mode else config['steps_per_iter']
    
    print(f"\n{'='*60}")
    print(f"Training Agent for: {topology.upper()}")
    print(f"{'='*60}")
    print(f"  {config['description']}")
    print(f"  Hidden dim: {config['hidden_dim']}, LR: {config['lr']}")
    print(f"  Iterations: {n_iterations}, Steps/iter: {steps_per_iter}")
    print(f"  Total steps: {n_iterations * steps_per_iter:,}")
    print('-'*60)
    
    # Create environment
    env = SingleTopologyEnv(surrogate, topology, device=DEVICE)
    
    # Create agent with topology-specific config
    agent = PPOAgent(
        env,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=0.995,
        gae_lambda=0.97,
        clip_epsilon=0.15,
        entropy_coef=0.005,
        value_coef=0.5,
        device=DEVICE,
    )
    
    # Training tracking
    all_rewards = []
    all_mses = []
    best_mean_reward = float('-inf')
    best_mse = float('inf')
    
    # Training loop
    for iteration in tqdm(range(n_iterations), desc=f"Training {topology}"):
        # Collect rollouts
        rollout = agent.collect_rollouts(steps_per_iter)
        
        # Update policy
        agent.update(rollout, n_epochs=10, batch_size=64)
        
        # Track performance
        if len(agent.episode_rewards) > 0:
            all_rewards.extend(agent.episode_rewards[-10:])
            all_mses.extend(agent.episode_mses[-10:])
        
        # Logging
        if (iteration + 1) % max(1, n_iterations // 10) == 0:
            recent_rewards = agent.episode_rewards[-50:] if agent.episode_rewards else [0]
            recent_mses = agent.episode_mses[-50:] if agent.episode_mses else [float('inf')]
            
            mean_reward = np.mean(recent_rewards)
            mean_mse = np.mean(recent_mses)
            
            # Save best
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_mse = mean_mse
                agent.save(f'checkpoints/rl_agent_{topology}.pt')
    
    # Final save
    agent.save(f'checkpoints/rl_agent_{topology}.pt')
    
    return {
        'topology': topology,
        'best_reward': float(best_mean_reward),
        'best_mse': float(best_mse),
        'hidden_dim': config['hidden_dim'],
        'n_iterations': n_iterations,
        'total_steps': n_iterations * steps_per_iter,
    }


def test_topology_agent(topology: str, surrogate, n_tests: int = 20):
    """Test a trained topology-specific agent."""
    
    config = TOPOLOGY_CONFIG[topology]
    
    # Create environment and agent
    env = SingleTopologyEnv(surrogate, topology, device=DEVICE)
    agent = PPOAgent(env, hidden_dim=config['hidden_dim'], device=DEVICE)
    
    # Load trained agent
    agent_path = f'checkpoints/rl_agent_{topology}.pt'
    if not Path(agent_path).exists():
        return {'topology': topology, 'error': 'Agent not found'}
    
    agent.load(agent_path)
    
    # Test
    mses = []
    for _ in range(n_tests):
        state = env.reset()
        for step in range(50):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, _ = agent.policy.get_action(state_tensor, deterministic=True)
            action_np = action.cpu().numpy().squeeze()
            state, _, done, info = env.step(action_np)
            if done:
                break
        mses.append(info['mse'])
    
    return {
        'topology': topology,
        'mean_mse': float(np.mean(mses)),
        'std_mse': float(np.std(mses)),
        'rmse': float(np.sqrt(np.mean(mses))),
    }


def train_all_topologies(quick_mode: bool = False):
    """Train agents for all topologies."""
    
    print("\n" + "="*60)
    print("Per-Topology RL Agent Training")
    print("="*60)
    print(f"Mode: {'Quick (reduced iterations)' if quick_mode else 'Full'}")
    print(f"Topologies: {', '.join(TOPOLOGIES)}")
    
    # Load surrogate
    print("\nLoading multi-topology surrogate...")
    surrogate = load_trained_model(device=DEVICE)
    print(f"  ✓ Loaded surrogate with {sum(p.numel() for p in surrogate.parameters()):,} params")
    
    # Train each topology
    results = {}
    for topology in TOPOLOGIES:
        result = train_single_topology(topology, surrogate, quick_mode)
        results[topology] = result
        print(f"  ✓ {topology}: Best MSE = {result['best_mse']:.2f}")
    
    # Test all agents
    print("\n" + "="*60)
    print("Testing All Topology Agents")
    print("="*60)
    
    test_results = {}
    for topology in TOPOLOGIES:
        test_result = test_topology_agent(topology, surrogate)
        test_results[topology] = test_result
        if 'error' not in test_result:
            print(f"  {topology:12s}: MSE = {test_result['mean_mse']:7.1f} ± {test_result['std_mse']:6.1f}  (RMSE = {test_result['rmse']:.1f}V)")
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'quick_mode': quick_mode,
        'training_results': results,
        'test_results': test_results,
    }
    
    with open('checkpoints/per_topology_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"  Saved {len(TOPOLOGIES)} agents to checkpoints/rl_agent_<topology>.pt")
    print(f"  Summary: checkpoints/per_topology_training_summary.json")
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode with reduced iterations')
    parser.add_argument('--topology', type=str, default=None,
                        help='Train only a specific topology')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test existing agents')
    args = parser.parse_args()
    
    if args.test_only:
        surrogate = load_trained_model(device=DEVICE)
        for topo in TOPOLOGIES:
            result = test_topology_agent(topo, surrogate)
            if 'error' not in result:
                print(f"{topo:12s}: MSE = {result['mean_mse']:7.1f} ± {result['std_mse']:6.1f}")
    elif args.topology:
        surrogate = load_trained_model(device=DEVICE)
        train_single_topology(args.topology, surrogate, args.quick)
        test_topology_agent(args.topology, surrogate)
    else:
        train_all_topologies(quick_mode=args.quick)
