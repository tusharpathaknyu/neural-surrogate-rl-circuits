"""
Train Separate RL Agents Per Topology.

Each topology gets its own specialized agent for optimal performance.
This approach typically gives 2-5x better results than a single multi-topology agent.

NEW: Supports SPICE-enhanced training for ground-truth validation.
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

# All topologies (7 total, including QR Flyback)
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']

# ============================================================================
# PHYSICS-INFORMED HYPERPARAMETER TUNING
# ============================================================================
# Each topology has unique dynamics that require different optimization strategies:
#
# COMPLEXITY TIERS:
#   Tier 1 (Simple):    Buck - linear V_out = V_in * D, stable across duty range
#   Tier 2 (Moderate):  Buck-Boost, Ćuk - inverting, discontinuous currents
#   Tier 3 (Complex):   Boost, SEPIC, Flyback - nonlinear, unstable at high duty
#
# KEY CONSIDERATIONS:
#   - High duty cycle instability (Boost, SEPIC, Flyback approach infinity as D→1)
#   - Coupled inductors (SEPIC, Ćuk, Flyback) need larger networks
#   - Transformer dynamics (Flyback) have additional magnetizing inductance
#   - Inverting topologies (Buck-Boost, Ćuk) have polarity considerations
# ============================================================================

TOPOLOGY_CONFIG = {
    'buck': {
        # === TIER 1: SIMPLE ===
        # Linear relationship: V_out = V_in × D
        # Stable across entire duty cycle range
        # Fastest to converge, smallest network sufficient
        'hidden_dim': 256,
        'lr': 3e-4,           # Higher LR - simple landscape
        'n_iterations': 300,
        'steps_per_iter': 2048,
        'gamma': 0.99,        # Standard discount
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,  # Standard clipping
        'entropy_coef': 0.01, # Standard exploration
        'description': 'Simple step-down, linear V_out=V_in×D, stable across all duty cycles',
    },
    'boost': {
        # === TIER 3: COMPLEX ===
        # Nonlinear: V_out = V_in / (1-D), approaches infinity as D→1
        # Unstable at high duty cycles (>0.8)
        # Right-half-plane zero causes control challenges
        # Needs conservative learning, more exploration
        'hidden_dim': 512,
        'lr': 1e-4,           # Lower LR - unstable dynamics
        'n_iterations': 500,
        'steps_per_iter': 4096,
        'gamma': 0.995,       # Higher gamma - delayed rewards from instability
        'gae_lambda': 0.97,
        'clip_epsilon': 0.15, # Tighter clipping - prevent wild updates
        'entropy_coef': 0.02, # More exploration for nonlinear landscape
        'description': 'Nonlinear V_out=V_in/(1-D), RHP zero, unstable at high duty (>0.8)',
    },
    'buck_boost': {
        # === TIER 2: MODERATE ===
        # Inverting: V_out = -V_in × D/(1-D)
        # Combines buck and boost challenges
        # Discontinuous input and output currents
        'hidden_dim': 256,
        'lr': 2e-4,           # Moderate LR
        'n_iterations': 400,
        'steps_per_iter': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.18,
        'entropy_coef': 0.015,
        'description': 'Inverting, V_out=-V_in×D/(1-D), discontinuous currents',
    },
    'sepic': {
        # === TIER 3: COMPLEX ===
        # Non-inverting buck-boost: V_out = V_in × D/(1-D)
        # Coupled inductors add energy storage dynamics
        # Capacitor C1 transfers energy between stages
        # More components = more degrees of freedom
        'hidden_dim': 512,
        'lr': 1e-4,           # Lower LR - coupled dynamics
        'n_iterations': 500,
        'steps_per_iter': 4096,
        'gamma': 0.995,       # Longer horizon for energy transfer
        'gae_lambda': 0.97,
        'clip_epsilon': 0.15,
        'entropy_coef': 0.02, # Explore coupled inductor space
        'description': 'Coupled inductors, energy transfer capacitor, 4+ reactive elements',
    },
    'cuk': {
        # === TIER 2: MODERATE ===
        # Inverting like buck-boost but with CONTINUOUS currents
        # Better EMI characteristics, less filter needed
        # Two inductors share energy via coupling capacitor
        'hidden_dim': 256,
        'lr': 2e-4,
        'n_iterations': 400,
        'steps_per_iter': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.18,
        'entropy_coef': 0.015,
        'description': 'Continuous I/O currents, capacitive energy transfer, low ripple',
    },
    'flyback': {
        # === TIER 3: COMPLEX (ISOLATED) ===
        # Transformer provides galvanic isolation
        # Magnetizing inductance stores energy (not just transfers)
        # Leakage inductance causes voltage spikes
        # Turns ratio (N) adds another design dimension
        # Most complex dynamics due to transformer modeling
        'hidden_dim': 512,
        'lr': 8e-5,           # Lowest LR - transformer dynamics
        'n_iterations': 600,  # Most iterations - complex search space
        'steps_per_iter': 4096,
        'gamma': 0.997,       # Longest horizon - energy storage in transformer
        'gae_lambda': 0.98,
        'clip_epsilon': 0.12, # Tightest clipping - prevent divergence
        'entropy_coef': 0.025,# Most exploration - transformer + converter params
        'description': 'Isolated, transformer magnetics, leakage inductance, turns ratio N',
    },
    'qr_flyback': {
        # === TIER 4: MOST COMPLEX (RESONANT + ISOLATED) ===
        # Quasi-Resonant (QR) operation for soft switching
        # Zero-Voltage Switching (ZVS) or Zero-Current Switching (ZCS)
        # Resonant tank (Lr, Cr) creates sinusoidal transitions
        # Variable frequency operation (valley switching)
        # Reduced EMI and switching losses vs hard-switched flyback
        # Additional resonant components add design complexity
        'hidden_dim': 512,
        'lr': 5e-5,           # Lowest LR - resonant + transformer dynamics
        'n_iterations': 700,  # Most iterations - resonant timing critical
        'steps_per_iter': 4096,
        'gamma': 0.998,       # Longest horizon - resonant energy cycling
        'gae_lambda': 0.98,
        'clip_epsilon': 0.10, # Tightest clipping - resonance sensitive
        'entropy_coef': 0.03, # High exploration - resonant frequency space
        'description': 'QR soft-switching, ZVS/ZCS, variable freq, resonant tank Lr/Cr',
    },
}

# Validate all required keys exist
REQUIRED_KEYS = ['hidden_dim', 'lr', 'n_iterations', 'steps_per_iter', 
                 'gamma', 'gae_lambda', 'clip_epsilon', 'entropy_coef']
for topo, config in TOPOLOGY_CONFIG.items():
    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing {key} in {topo} config")

# SPICE validation settings
USE_SPICE_GLOBALLY = False  # Will be set via command line


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
    elif topology in ['flyback', 'qr_flyback']:
        # QR Flyback has same DC transfer function, but lower ripple
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
    
    def __init__(self, surrogate, topology: str, device='cpu', 
                 use_spice: bool = False, spice_freq: int = 10, **kwargs):
        """
        Args:
            surrogate: Neural surrogate model
            topology: Topology name (buck, boost, etc.)
            device: CPU/CUDA/MPS
            use_spice: Enable SPICE-based ground truth validation
            spice_freq: Validate with SPICE every N steps (default 10)
        """
        super().__init__(
            surrogate, 
            device=device, 
            use_spice_reward=use_spice,
            spice_validation_freq=spice_freq,
            **kwargs
        )
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


def train_single_topology(topology: str, surrogate, quick_mode: bool = False, 
                          use_spice: bool = False, spice_freq: int = 10):
    """Train an agent for a single topology with physics-informed hyperparameters."""
    
    config = TOPOLOGY_CONFIG[topology]
    
    # Reduce iterations in quick mode
    n_iterations = config['n_iterations'] // 5 if quick_mode else config['n_iterations']
    steps_per_iter = config['steps_per_iter'] // 2 if quick_mode else config['steps_per_iter']
    
    print(f"\n{'='*60}")
    print(f"Training Agent for: {topology.upper()}")
    print(f"{'='*60}")
    print(f"  Physics: {config['description']}")
    print(f"  Network: hidden_dim={config['hidden_dim']}")
    print(f"  Learning: lr={config['lr']:.0e}, γ={config['gamma']}, λ={config['gae_lambda']}")
    print(f"  PPO: clip_ε={config['clip_epsilon']}, entropy={config['entropy_coef']}")
    print(f"  Training: {n_iterations} iters × {steps_per_iter} steps = {n_iterations * steps_per_iter:,} total")
    if use_spice:
        print(f"  SPICE: Validating every {spice_freq} steps (ground truth)")
    print('-'*60)
    
    # Create environment with SPICE support
    env = SingleTopologyEnv(
        surrogate, topology, device=DEVICE,
        use_spice=use_spice, spice_freq=spice_freq
    )
    
    # Create agent with TOPOLOGY-SPECIFIC hyperparameters
    # These are physics-informed based on each converter's dynamics
    agent = PPOAgent(
        env,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma'],              # Topology-specific discount
        gae_lambda=config['gae_lambda'],    # Topology-specific advantage
        clip_epsilon=config['clip_epsilon'],# Topology-specific clipping
        entropy_coef=config['entropy_coef'],# Topology-specific exploration
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
        'spice_enhanced': use_spice,
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


def train_all_topologies(quick_mode: bool = False, use_spice: bool = False, 
                         spice_freq: int = 10):
    """Train agents for all topologies."""
    
    print("\n" + "="*60)
    print("Per-Topology RL Agent Training")
    print("="*60)
    print(f"Mode: {'Quick (reduced iterations)' if quick_mode else 'Full'}")
    if use_spice:
        print(f"SPICE: Enabled (validation every {spice_freq} steps)")
    print(f"Topologies: {', '.join(TOPOLOGIES)}")
    
    # Load surrogate
    print("\nLoading multi-topology surrogate...")
    surrogate = load_trained_model(device=DEVICE)
    print(f"  ✓ Loaded surrogate with {sum(p.numel() for p in surrogate.parameters()):,} params")
    
    # Train each topology
    results = {}
    for topology in TOPOLOGIES:
        result = train_single_topology(
            topology, surrogate, quick_mode,
            use_spice=use_spice, spice_freq=spice_freq
        )
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
    
    parser = argparse.ArgumentParser(description='Train RL agents for DC-DC converter topologies')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode with reduced iterations')
    parser.add_argument('--topology', type=str, default=None,
                        help='Train only a specific topology')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test existing agents')
    parser.add_argument('--spice', action='store_true',
                        help='Enable SPICE-based ground truth validation (slower but more accurate)')
    parser.add_argument('--spice-freq', type=int, default=10,
                        help='Validate with SPICE every N steps (default: 10)')
    args = parser.parse_args()
    
    if args.test_only:
        surrogate = load_trained_model(device=DEVICE)
        for topo in TOPOLOGIES:
            result = test_topology_agent(topo, surrogate)
            if 'error' not in result:
                print(f"{topo:12s}: MSE = {result['mean_mse']:7.1f} ± {result['std_mse']:6.1f}")
    elif args.topology:
        surrogate = load_trained_model(device=DEVICE)
        train_single_topology(
            args.topology, surrogate, args.quick,
            use_spice=args.spice, spice_freq=args.spice_freq
        )
        test_topology_agent(args.topology, surrogate)
    else:
        train_all_topologies(
            quick_mode=args.quick, 
            use_spice=args.spice, 
            spice_freq=args.spice_freq
        )
