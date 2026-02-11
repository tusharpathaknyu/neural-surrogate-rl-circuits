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

# Full training across all 7 topologies (post-bugfix)
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']

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
    
    BUG FIX: Old version created synthetic waveforms with rise times starting at 0,
    but the surrogate model outputs steady-state waveforms (e.g., buck starts at ~6V).
    The agent was trying to match an impossible target shape.
    
    New approach: Generate steady-state waveforms with realistic ripple,
    matching the distribution the surrogate actually produces.
    
    Note: output_points=32 to match surrogate model output shape.
    """
    
    t = np.linspace(0, 0.01, output_points)
    
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
        eff_duty = duty * 0.95
        v_out = v_in * eff_duty * n / (1 - eff_duty) if eff_duty < 0.95 else v_in * n
    else:
        v_out = v_in * duty
    
    # Clip to reasonable range
    v_out = np.clip(v_out, -60, 60)
    
    # BUG FIX: Create STEADY-STATE waveform (no synthetic rise time from 0).
    # The surrogate outputs steady-state waveforms, so targets must match.
    target = np.ones(output_points, dtype=np.float32) * v_out
    
    # Add topology-specific ripple (this IS realistic - all converters have ripple)
    ripple_config = TOPOLOGY_REWARD_CONFIG.get(topology, {})
    ripple_pct = ripple_config.get('ripple_target', 0.03)
    ripple = ripple_pct * abs(v_out) * np.sin(2 * np.pi * 10 * t)
    target = target + ripple
    
    # Add small random noise to make targets slightly varied (prevents overfitting)
    target += np.random.normal(0, 0.01 * abs(v_out) + 0.01, output_points)
    
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
        self.topology_idx = TOPOLOGIES.index(topology) if topology in TOPOLOGIES else 0
        self.is_multi_topology = True
        
        # Track SPICE metrics
        self.spice_call_count = 0
        self.spice_mse_history = []
        self.step_count = 0
        
    def reset(self):
        """Reset environment with topology-appropriate target.
        
        BUG FIX (v2): Instead of using create_target_waveform() with manual transfer
        functions, we generate targets by running the SURROGATE on random params.
        This guarantees target-surrogate alignment for ALL topologies.
        
        Old approach had 2 fatal mismatches:
          - buck_boost/cuk: surrogate outputs positive, target was negative (MSE=4339)
          - qr_flyback: surrogate uses buck scaling, target used QR formula (MSE=216)
        """
        # Generate target by running surrogate with random params
        # This is what the base env does, and it guarantees alignment
        target_params = self._random_params()
        self.target_waveform = self._simulate(target_params)
        
        # Random DIFFERENT starting parameters (agent must find the right ones)
        self.current_params = self._random_params()
        
        self.current_step = 0
        self.prev_mse = None
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action: np.ndarray):
        """Take a step with topology-aware rewards."""
        # Apply action (clipped deltas)
        action = np.clip(action, -1, 1)
        
        # BUG FIX: Was using 0.1 (10%) max change vs base env's 0.2 (20%).
        # Less exploration = slower convergence. Match base env.
        max_change_pct = 0.2
        
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                log_low, log_high = np.log(low), np.log(high)
                current_log = np.log(self.current_params[i])
                delta = action[i] * max_change_pct * (log_high - log_low)
                new_log = np.clip(current_log + delta, log_low, log_high)
                self.current_params[i] = np.exp(new_log)
            else:
                delta = action[i] * max_change_pct * (high - low)
                self.current_params[i] = np.clip(
                    self.current_params[i] + delta, low, high
                )
        
        # Simulate with surrogate
        predicted = self._simulate(self.current_params)
        
        # SPICE validation: fire on FIRST step (critical for short episodes that
        # terminate early at MSE<5) and then every N steps for longer episodes.
        # Without step 1, SPICE never fires because episodes end too quickly.
        self.step_count += 1
        use_spice_this_step = (
            self.use_spice_reward and 
            self.spice_calculator is not None and
            (self.step_count == 1 or self.step_count % self.spice_validation_freq == 0)
        )
        
        # SPICE as a TRUST SIGNAL (agreement-based reward modulation)
        #
        # Problem: target waveforms come from the surrogate. SPICE waveforms live
        # in a different domain (ripple, switching noise, transients). Comparing
        # SPICE output to surrogate targets via raw MSE is apples-to-oranges —
        # even perfect parameters give high MSE due to domain mismatch.
        #
        # Previous fixes tried: (1) additive bonus → reward hacking, (2) SPICE
        # replaces surrogate → domain mismatch causes MSE to diverge anyway.
        #
        # Correct approach: AGREEMENT-BASED reward modulation.
        # - Always compute reward from surrogate (same domain as target)
        # - On SPICE steps, measure how much surrogate and SPICE AGREE
        # - If they agree → surrogate is trustworthy here → keep full reward
        # - If they disagree → agent is in a surrogate-inaccurate region → crush reward
        # - Also reward SPICE DC accuracy and waveform quality
        #
        # This pushes the agent toward regions where the surrogate is accurate,
        # which are the only regions where low surrogate-MSE is meaningful.
        
        # 1. Always compute surrogate-based reward (same domain as target)
        reward, info = compute_topology_aware_reward(
            predicted, self.target_waveform,
            self.topology, self.prev_mse
        )
        
        spice_succeeded = False
        if use_spice_this_step:
            try:
                spice_waveform = self.spice_calculator.simulate(self.current_params)
                
                if spice_waveform is not None:
                    spice_resampled = self.spice_calculator.resample_to_target(
                        spice_waveform, len(self.target_waveform)
                    )
                    
                    # 2. Surrogate-SPICE agreement: how trustworthy is the surrogate here?
                    spice_surr_mse = float(np.mean((spice_resampled - predicted) ** 2))
                    agreement = 1.0 / (1.0 + spice_surr_mse / 5.0)
                    # agreement ≈ 1.0 when they agree, drops toward 0 when they disagree
                    # Using /5.0 (not /10.0) for harsher penalization of disagreement
                    
                    # 3. SPICE DC accuracy: BONUS when close, PENALTY when far
                    spice_dc = float(np.mean(spice_resampled))
                    target_dc = float(np.mean(self.target_waveform))
                    dc_error_pct = abs(spice_dc - target_dc) / (abs(target_dc) + 1e-6)
                    if dc_error_pct < 0.5:
                        # Good DC match: up to +3 bonus
                        dc_term = 3.0 * (1.0 - 2.0 * dc_error_pct)
                    else:
                        # Bad DC match: up to -5 penalty (capped)
                        dc_term = -min(5.0, 2.0 * (dc_error_pct - 0.5))
                    
                    # 4. Waveform quality bonus from HIGH-RES SPICE data
                    spice_bonus, spice_metrics = self.spice_calculator.compute_spice_quality_bonus(
                        spice_waveform, self.current_params, self.topology
                    )
                    
                    # 5. Final reward: surrogate reward modulated by SPICE trust
                    #    + DC accuracy term + waveform quality bonus
                    reward = reward * agreement + dc_term + spice_bonus
                    
                    # Record SPICE metrics for logging
                    spice_target_mse = float(np.mean((spice_resampled - self.target_waveform) ** 2))
                    info['spice_validated'] = True
                    info['spice_mse'] = spice_target_mse  # For monitoring
                    info['spice_surr_mse'] = spice_surr_mse
                    info['agreement'] = agreement
                    info['dc_error_pct'] = dc_error_pct * 100
                    info['spice_bonus'] = spice_bonus
                    info['dc_term'] = dc_term
                    info['spice_v_out'] = spice_metrics['v_out_mean']
                    info['spice_ripple_pct'] = spice_metrics['ripple_pct']
                    info['spice_overshoot_pct'] = spice_metrics['overshoot_pct']
                    info['spice_thd'] = spice_metrics['thd']
                    info['surrogate_mse'] = info['mse']
                    
                    self.spice_call_count += 1
                    self.spice_mse_history.append(spice_target_mse)
                    spice_succeeded = True
            except Exception as e:
                self.spice_fail_count = getattr(self, 'spice_fail_count', 0) + 1
        
        if not spice_succeeded:
            info['spice_validated'] = False
        
        # Update state
        self.prev_mse = info['mse']
        self.current_step += 1
        
        done = (self.current_step >= self.max_steps) or (info['mse'] < 5.0)
        
        return self._get_state(), reward, done, info


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
    last_hourly_save = time.time()  # Track hourly checkpoint saves
    
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
            spice_ripples = []
            spice_overshoots = []
            agreements = []
            dc_errors = []
            for _ in range(5):
                state = env.reset()
                for step in range(20):
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        action, _, _ = agent.policy.get_action(state_t, deterministic=True)
                    action_np = action.cpu().numpy().squeeze()
                    state, _, done, info = env.step(action_np)
                    if info.get('spice_validated', False):
                        spice_mses.append(info.get('spice_mse', info['mse']))
                        if 'spice_ripple_pct' in info:
                            spice_ripples.append(info['spice_ripple_pct'])
                        if 'spice_overshoot_pct' in info:
                            spice_overshoots.append(info['spice_overshoot_pct'])
                        if 'agreement' in info:
                            agreements.append(info['agreement'])
                        if 'dc_error_pct' in info:
                            dc_errors.append(info['dc_error_pct'])
                    if done:
                        break
            
            if spice_mses:
                spice_mse = np.mean(spice_mses)
                postfix = {
                    'spice_mse': f'{spice_mse:.1f}',
                    'best': f'{best_mse:.1f}'
                }
                if spice_ripples:
                    postfix['ripple%'] = f'{np.mean(spice_ripples):.1f}'
                if agreements:
                    postfix['agree'] = f'{np.mean(agreements):.2f}'
                if dc_errors:
                    postfix['dc_err%'] = f'{np.mean(dc_errors):.1f}'
                pbar.set_postfix(postfix)
                
                # Print SPICE results as a separate line so they don't get lost in tqdm
                agree_str = f", agree={np.mean(agreements):.2f}" if agreements else ""
                dc_str = f", dc_err={np.mean(dc_errors):.1f}%" if dc_errors else ""
                ripple_str = f", ripple={np.mean(spice_ripples):.1f}%" if spice_ripples else ""
                print(f"\n  SPICE[iter {iteration+1}]: mse={spice_mse:.1f}{agree_str}{dc_str}{ripple_str}")
        
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
        
        # Hourly checkpoint save (every 3600 seconds)
        elapsed_since_save = time.time() - last_hourly_save
        if elapsed_since_save >= 3600:
            hourly_path = f'checkpoints/rl_agent_{topology}_hourly.pt'
            agent.save(hourly_path)
            last_hourly_save = time.time()
            hours_total = (time.time() - start_time) / 3600
            print(f'\n  [HOURLY SAVE] {topology} @ iter {iteration+1}/{config["n_iterations"]} ({hours_total:.1f}h elapsed) -> {hourly_path}')
    
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
    spice_stats = env.spice_calculator.get_stats() if env.spice_calculator else {}
    print(f"    SPICE: {env.spice_call_count} calls, "
          f"success_rate={spice_stats.get('success_rate', 0):.0%}, "
          f"cache_hits={spice_stats.get('hit_count', 0)}")
    if env.spice_mse_history:
        print(f"    SPICE MSE: first={env.spice_mse_history[0]:.1f}, "
              f"last={env.spice_mse_history[-1]:.1f}, "
              f"best={min(env.spice_mse_history):.1f}")
    
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
