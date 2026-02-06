#!/usr/bin/env python3
"""
INTENSIVE SPICE-IN-THE-LOOP TRAINING
====================================
Full production training with:
1. Per-topology SPICE reward calculations (ground truth)
2. Physics-informed hyperparameters per topology
3. Topology-aware reward shaping (inverted, resonant, isolated)
4. Full iteration counts (300-700 per topology, ~11.7M steps total)

This is the proper training that takes ~18 hours.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import time
import os
import signal

from models.multi_topology_surrogate import load_trained_model
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent
from rl.topology_rewards import compute_topology_aware_reward, TOPOLOGY_REWARD_CONFIG

# Force CPU to avoid MPS freezing
DEVICE = 'cpu'
print(f"Using device: {DEVICE} (forced for stability)")

# Resume from flyback (buck, boost, buck_boost, sepic, cuk already done)
TOPOLOGIES = ['flyback', 'qr_flyback']

# ============================================================================
# PHYSICS-INFORMED HYPERPARAMETERS (from original 18-hour training)
# ============================================================================
TOPOLOGY_CONFIG = {
    'buck': {
        'hidden_dim': 256,
        'lr': 3e-4,
        'n_iterations': 300,
        'steps_per_iter': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'spice_freq': 5,  # SPICE validate every 5 steps (within episode)
        'description': 'Simple step-down, linear V_out=V_in×D',
    },
    'boost': {
        'hidden_dim': 512,
        'lr': 1e-4,
        'n_iterations': 500,
        'steps_per_iter': 4096,
        'gamma': 0.995,
        'gae_lambda': 0.97,
        'clip_epsilon': 0.15,
        'entropy_coef': 0.02,
        'spice_freq': 5,
        'description': 'Nonlinear V_out=V_in/(1-D), RHP zero',
    },
    'buck_boost': {
        'hidden_dim': 256,
        'lr': 2e-4,
        'n_iterations': 400,
        'steps_per_iter': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.18,
        'entropy_coef': 0.015,
        'spice_freq': 5,
        'description': 'Inverting, V_out=-V_in×D/(1-D)',
    },
    'sepic': {
        'hidden_dim': 512,
        'lr': 1e-4,
        'n_iterations': 500,
        'steps_per_iter': 4096,
        'gamma': 0.995,
        'gae_lambda': 0.97,
        'clip_epsilon': 0.15,
        'entropy_coef': 0.02,
        'spice_freq': 5,
        'description': 'Coupled inductors, 4+ reactive elements',
    },
    'cuk': {
        'hidden_dim': 256,
        'lr': 2e-4,
        'n_iterations': 400,
        'steps_per_iter': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.18,
        'entropy_coef': 0.015,
        'spice_freq': 5,
        'description': 'Continuous currents, capacitive transfer',
    },
    'flyback': {
        'hidden_dim': 512,
        'lr': 8e-5,
        'n_iterations': 600,
        'steps_per_iter': 4096,
        'gamma': 0.997,
        'gae_lambda': 0.98,
        'clip_epsilon': 0.12,
        'entropy_coef': 0.025,
        'spice_freq': 5,
        'description': 'Isolated, transformer magnetics',
    },
    'qr_flyback': {
        'hidden_dim': 512,
        'lr': 5e-5,
        'n_iterations': 700,
        'steps_per_iter': 4096,
        'gamma': 0.998,
        'gae_lambda': 0.98,
        'clip_epsilon': 0.10,
        'entropy_coef': 0.03,
        'spice_freq': 5,
        'description': 'QR soft-switching, ZVS/ZCS, resonant tank',
    },
}


def create_target_waveform(topology: str, v_in: float = 12.0, duty: float = 0.5,
                           output_points: int = 32) -> np.ndarray:
    """Create physics-based target waveform for each topology.
    
    Note: output_points=32 to match surrogate model output shape.
    """
    
    t = np.linspace(0, 0.01, output_points)
    rise_time = 0.0005
    
    # Calculate expected V_out based on topology transfer function
    if topology == 'buck':
        v_out = v_in * duty
    elif topology == 'boost':
        v_out = v_in / (1 - duty) if duty < 0.95 else v_in / 0.05
    elif topology == 'buck_boost':
        v_out = -v_in * duty / (1 - duty) if duty < 0.95 else -v_in * duty / 0.05
    elif topology == 'sepic':
        v_out = v_in * duty / (1 - duty) if duty < 0.95 else v_in * duty / 0.05
    elif topology == 'cuk':
        v_out = -v_in * duty / (1 - duty) if duty < 0.95 else -v_in * duty / 0.05
    elif topology == 'flyback':
        n = 1.0  # Turns ratio
        v_out = v_in * duty * n / (1 - duty) if duty < 0.95 else v_in * duty * n / 0.05
    elif topology == 'qr_flyback':
        n = 1.0
        # QR has slightly different effective duty due to resonant period
        eff_duty = duty * 0.95
        v_out = v_in * eff_duty * n / (1 - eff_duty) if eff_duty < 0.95 else v_in * n
    else:
        v_out = v_in * duty
    
    # Clip to reasonable range
    v_out = np.clip(v_out, -60, 60)
    
    # Create waveform with rise time
    target = np.ones(output_points) * v_out
    rise_mask = t < rise_time
    if v_out >= 0:
        target[rise_mask] = v_out * (1 - np.exp(-t[rise_mask] * 60))
    else:
        target[rise_mask] = v_out * (1 - np.exp(-t[rise_mask] * 60))
    
    # Add topology-specific ripple
    ripple_config = TOPOLOGY_REWARD_CONFIG.get(topology, {})
    ripple_pct = ripple_config.get('ripple_target', 0.03)
    ripple = ripple_pct * abs(v_out) * np.sin(2 * np.pi * 10 * t)
    target = target + ripple
    
    return target.astype(np.float32)


class TopologySpecificEnv(CircuitDesignEnv):
    """Environment with topology-specific SPICE rewards."""
    
    def __init__(self, surrogate, topology: str, device='cpu',
                 use_spice: bool = True, spice_freq: int = 50, **kwargs):
        """
        Args:
            surrogate: Neural surrogate model
            topology: Topology name
            device: CPU/CUDA
            use_spice: Enable SPICE validation
            spice_freq: Run SPICE every N steps
        """
        super().__init__(
            surrogate,
            device=device,
            topology=topology,
            use_spice_reward=use_spice,
            spice_validation_freq=spice_freq,
            **kwargs
        )
        self.topology = topology
        self.topology_idx = TOPOLOGIES.index(topology)
        self.is_multi_topology = True
        
        # Track SPICE metrics
        self.spice_call_count = 0
        self.spice_mse_history = []
        self.step_count = 0
        
    def reset(self):
        """Reset environment with topology-appropriate target."""
        v_in = np.random.uniform(8, 36)
        duty = np.random.uniform(0.3, 0.7)
        
        self.target_waveform = create_target_waveform(
            self.topology, v_in=v_in, duty=duty
        )
        
        # Random starting parameters
        self.current_params = np.array([
            np.random.uniform(*self.PARAM_BOUNDS['L']),
            np.random.uniform(*self.PARAM_BOUNDS['C']),
            np.random.uniform(*self.PARAM_BOUNDS['R_load']),
            v_in,
            np.random.uniform(*self.PARAM_BOUNDS['f_sw']),
            duty,
        ], dtype=np.float32)
        
        self.current_step = 0
        self.prev_mse = None
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action: np.ndarray):
        """Take a step with topology-aware rewards."""
        # Apply action (clipped deltas)
        action = np.clip(action, -1, 1)
        
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                log_low, log_high = np.log(low), np.log(high)
                current_log = np.log(self.current_params[i])
                delta = action[i] * 0.1 * (log_high - log_low)
                new_log = np.clip(current_log + delta, log_low, log_high)
                self.current_params[i] = np.exp(new_log)
            else:
                delta = action[i] * 0.1 * (high - low)
                self.current_params[i] = np.clip(
                    self.current_params[i] + delta, low, high
                )
        
        # Simulate with surrogate
        predicted = self._simulate(self.current_params)
        
        # SPICE validation every N steps (proper ground truth rewards)
        self.step_count += 1
        use_spice_this_step = (
            self.use_spice_reward and 
            self.spice_calculator is not None and
            self.step_count % self.spice_validation_freq == 0
        )
        
        if use_spice_this_step:
            try:
                spice_waveform = self._run_spice_simulation()
                if spice_waveform is not None:
                    # Use SPICE ground truth for reward
                    reward, info = compute_topology_aware_reward(
                        spice_waveform, self.target_waveform,
                        self.topology, self.prev_mse
                    )
                    info['spice_validated'] = True
                    self.spice_call_count += 1
                    self.spice_mse_history.append(info['mse'])
                else:
                    reward, info = compute_topology_aware_reward(
                        predicted, self.target_waveform,
                        self.topology, self.prev_mse
                    )
                    info['spice_validated'] = False
            except Exception as e:
                reward, info = compute_topology_aware_reward(
                    predicted, self.target_waveform,
                    self.topology, self.prev_mse
                )
                info['spice_validated'] = False
        else:
            # Normal surrogate-based reward with topology awareness
            reward, info = compute_topology_aware_reward(
                predicted, self.target_waveform,
                self.topology, self.prev_mse
            )
            info['spice_validated'] = False
        
        # Update state
        self.prev_mse = info['mse']
        self.current_step += 1
        
        done = (self.current_step >= self.max_steps) or (info['mse'] < 0.1)
        
        return self._get_state(), reward, done, info
    
    def _run_spice_simulation(self):
        """Run actual SPICE simulation."""
        if self.spice_calculator is None:
            return None
        
        try:
            # Pass numpy array directly (not dict)
            waveform = self.spice_calculator.simulate(self.current_params)
            
            # Resample to match target length
            if waveform is not None and len(waveform) != len(self.target_waveform):
                indices = np.linspace(0, len(waveform)-1, len(self.target_waveform)).astype(int)
                waveform = waveform[indices]
            
            return waveform
            
        except Exception as e:
            print(f"SPICE error: {e}")
            return None


def train_topology_agent(topology: str, surrogate, config: dict) -> dict:
    """Train a single topology agent with SPICE validation."""
    
    print(f"\n{'='*60}")
    print(f"Training {topology.upper()} Agent")
    print(f"{'='*60}")
    print(f"  Config: {config['n_iterations']} iters × {config['steps_per_iter']} steps")
    print(f"  = {config['n_iterations'] * config['steps_per_iter']:,} total steps")
    print(f"  SPICE validation every {config['spice_freq']} iterations")
    print(f"  Description: {config['description']}")
    
    start_time = time.time()
    
    # Create environment with SPICE
    env = TopologySpecificEnv(
        surrogate, topology, device=DEVICE,
        use_spice=True, spice_freq=config['spice_freq']
    )
    
    # Create PPO agent with topology-specific hyperparameters
    agent = PPOAgent(
        env,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        entropy_coef=config['entropy_coef'],
        value_coef=0.5,
        device=DEVICE,
    )
    
    # Training tracking
    all_rewards = []
    all_mses = []
    best_reward = float('-inf')
    best_mse = float('inf')
    spice_validations = 0
    
    # Progress bar
    pbar = tqdm(range(config['n_iterations']), desc=f"{topology}")
    
    for iteration in pbar:
        # Collect rollouts
        rollout = agent.collect_rollouts(config['steps_per_iter'])
        
        # Update policy
        agent.update(rollout, n_epochs=10, batch_size=64)
        
        # Track performance
        if len(agent.episode_rewards) > 0:
            all_rewards.extend(agent.episode_rewards[-10:])
        if len(agent.episode_mses) > 0:
            all_mses.extend(agent.episode_mses[-10:])
        
        # SPICE validation iteration
        if (iteration + 1) % config['spice_freq'] == 0:
            spice_validations += 1
            
            # Test with multiple SPICE runs
            spice_mses = []
            for _ in range(5):
                state = env.reset()
                for step in range(20):
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        action, _, _ = agent.policy.get_action(state_t, deterministic=True)
                    action_np = action.cpu().numpy().squeeze()
                    state, _, done, info = env.step(action_np)
                    if info.get('spice_validated', False):
                        spice_mses.append(info['mse'])
                    if done:
                        break
            
            if spice_mses:
                spice_mse = np.mean(spice_mses)
                pbar.set_postfix({
                    'spice_mse': f'{spice_mse:.1f}',
                    'best': f'{best_mse:.1f}'
                })
        
        # Logging and checkpointing
        log_freq = max(1, config['n_iterations'] // 20)
        if (iteration + 1) % log_freq == 0:
            recent_rewards = agent.episode_rewards[-50:] if agent.episode_rewards else [0]
            recent_mses = agent.episode_mses[-50:] if agent.episode_mses else [float('inf')]
            
            mean_reward = np.mean(recent_rewards)
            mean_mse = np.mean(recent_mses)
            
            # Save best
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_mse = mean_mse
                agent.save(f'checkpoints/rl_agent_{topology}.pt')
            
            pbar.set_postfix({
                'reward': f'{mean_reward:.2f}',
                'mse': f'{mean_mse:.1f}',
                'best': f'{best_mse:.1f}'
            })
    
    # Final save
    agent.save(f'checkpoints/rl_agent_{topology}.pt')
    
    elapsed = time.time() - start_time
    
    result = {
        'topology': topology,
        'best_reward': float(best_reward),
        'best_mse': float(best_mse),
        'total_steps': config['n_iterations'] * config['steps_per_iter'],
        'training_time_sec': elapsed,
        'training_time_str': f"{elapsed/60:.1f} min",
        'spice_validations': spice_validations,
        'spice_calls': env.spice_call_count,
    }
    
    print(f"\n  ✓ {topology}: Best MSE = {best_mse:.2f}, Time = {elapsed/60:.1f} min")
    print(f"    SPICE calls: {env.spice_call_count}")
    
    return result


def test_topology_agent(topology: str, surrogate, n_tests: int = 30) -> dict:
    """Test a trained agent with SPICE ground truth."""
    
    config = TOPOLOGY_CONFIG[topology]
    
    # Create env with SPICE enabled
    env = TopologySpecificEnv(
        surrogate, topology, device=DEVICE,
        use_spice=True, spice_freq=1  # SPICE every step for testing
    )
    
    # Create and load agent
    agent = PPOAgent(env, hidden_dim=config['hidden_dim'], device=DEVICE)
    agent_path = f'checkpoints/rl_agent_{topology}.pt'
    
    if not Path(agent_path).exists():
        return {'topology': topology, 'error': 'Agent not found'}
    
    agent.load(agent_path)
    
    # Test
    surrogate_mses = []
    spice_mses = []
    
    for _ in range(n_tests):
        state = env.reset()
        for step in range(50):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, _ = agent.policy.get_action(state_t, deterministic=True)
            action_np = action.cpu().numpy().squeeze()
            state, _, done, info = env.step(action_np)
            if done:
                break
        
        surrogate_mses.append(info['mse'])
        
        # Final SPICE validation
        if info.get('spice_validated', False):
            spice_mses.append(info['mse'])
    
    result = {
        'topology': topology,
        'surrogate_mse': float(np.mean(surrogate_mses)),
        'surrogate_std': float(np.std(surrogate_mses)),
    }
    
    if spice_mses:
        result['spice_mse'] = float(np.mean(spice_mses))
        result['spice_std'] = float(np.std(spice_mses))
    
    return result


def main():
    """Run full intensive training."""
    
    print("\n" + "="*70)
    print("INTENSIVE SPICE-IN-THE-LOOP TRAINING")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Topologies: {', '.join(TOPOLOGIES)}")
    
    # Calculate total steps
    total_steps = sum(
        cfg['n_iterations'] * cfg['steps_per_iter'] 
        for cfg in TOPOLOGY_CONFIG.values()
    )
    print(f"Total training steps: {total_steps:,}")
    print(f"Estimated time: ~12-18 hours")
    
    # Load surrogate
    print("\nLoading multi-topology surrogate...")
    surrogate = load_trained_model(device=DEVICE)
    print(f"  ✓ Loaded surrogate: {sum(p.numel() for p in surrogate.parameters()):,} params")
    
    # Ensure checkpoint directory exists
    Path('checkpoints').mkdir(exist_ok=True)
    
    # Train all topologies
    training_results = {}
    overall_start = time.time()
    
    for i, topology in enumerate(TOPOLOGIES):
        print(f"\n[{i+1}/{len(TOPOLOGIES)}] Training {topology}...")
        config = TOPOLOGY_CONFIG[topology]
        result = train_topology_agent(topology, surrogate, config)
        training_results[topology] = result
        
        # Save intermediate results
        intermediate = {
            'timestamp': datetime.now().isoformat(),
            'completed': list(training_results.keys()),
            'results': training_results,
        }
        with open('checkpoints/training_progress.json', 'w') as f:
            json.dump(intermediate, f, indent=2)
    
    # Test all agents
    print("\n" + "="*70)
    print("TESTING ALL AGENTS WITH SPICE")
    print("="*70)
    
    test_results = {}
    for topology in TOPOLOGIES:
        result = test_topology_agent(topology, surrogate)
        test_results[topology] = result
        
        if 'error' not in result:
            msg = f"  {topology:12s}: MSE = {result['surrogate_mse']:7.1f} ± {result['surrogate_std']:5.1f}"
            if 'spice_mse' in result:
                msg += f"  (SPICE: {result['spice_mse']:.1f})"
            print(msg)
    
    # Final summary
    total_time = time.time() - overall_start
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': DEVICE,
        'total_steps': total_steps,
        'total_time_sec': total_time,
        'total_time_str': f"{total_time/3600:.2f} hours",
        'training_results': training_results,
        'test_results': test_results,
    }
    
    with open('checkpoints/intensive_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Saved {len(TOPOLOGIES)} agents to checkpoints/rl_agent_<topology>.pt")
    print(f"  Summary: checkpoints/intensive_training_summary.json")
    
    # Print final quality summary
    print("\nFinal Quality Summary:")
    for topo in TOPOLOGIES:
        if topo in test_results and 'error' not in test_results[topo]:
            mse = test_results[topo]['surrogate_mse']
            quality = "Excellent" if mse < 5 else "Good" if mse < 20 else "Fair" if mse < 50 else "Needs work"
            print(f"  {topo:12s}: {quality} (MSE={mse:.1f})")
    
    return summary


if __name__ == '__main__':
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Saving current progress...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    main()
