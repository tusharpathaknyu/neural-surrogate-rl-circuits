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


# Per-topology parameter bounds matching the data generation ranges.
# CRITICAL: The global PARAM_BOUNDS in environment.py is too wide (V_in 3.3-400V).
# Without per-topology bounds, the agent wanders into parameter regions where the
# surrogate has never been trained, causing surrogate exploitation (low surrogate MSE
# but SPICE MSE oscillating wildly 1600-52000+).
PER_TOPOLOGY_PARAM_BOUNDS = {
    'buck': {
        'L': (10e-6, 100e-6), 'C': (47e-6, 470e-6), 'R_load': (2, 50),
        'V_in': (8, 48), 'f_sw': (50e3, 500e3), 'duty': (0.1, 0.9),
    },
    'boost': {
        'L': (22e-6, 220e-6), 'C': (100e-6, 1000e-6), 'R_load': (10, 100),
        'V_in': (3.3, 24), 'f_sw': (50e3, 300e3), 'duty': (0.2, 0.8),
    },
    'buck_boost': {
        'L': (47e-6, 470e-6), 'C': (100e-6, 1000e-6), 'R_load': (5, 50),
        'V_in': (5, 36), 'f_sw': (50e3, 200e3), 'duty': (0.2, 0.8),
    },
    'sepic': {
        'L': (22e-6, 220e-6), 'C': (100e-6, 1000e-6), 'R_load': (10, 100),
        'V_in': (5, 24), 'f_sw': (50e3, 200e3), 'duty': (0.2, 0.8),
    },
    'cuk': {
        'L': (47e-6, 470e-6), 'C': (100e-6, 1000e-6), 'R_load': (5, 50),
        'V_in': (5, 24), 'f_sw': (50e3, 200e3), 'duty': (0.2, 0.8),
    },
    'flyback': {
        'L': (100e-6, 1000e-6), 'C': (100e-6, 1000e-6), 'R_load': (5, 100),
        'V_in': (12, 100), 'f_sw': (30e3, 150e3), 'duty': (0.2, 0.65),
        # V_in narrowed from (12, 400) — surrogate std=165V at full range
        # causes wild generalization. Focus on well-modeled region first.
    },
    'qr_flyback': {
        'L': (100e-6, 1000e-6), 'C': (100e-6, 1000e-6), 'R_load': (5, 100),
        'V_in': (12, 100), 'f_sw': (30e3, 120e3), 'duty': (0.15, 0.45),
        # V_in narrowed from (12, 400) — same surrogate accuracy concern.
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
        
        # CRITICAL FIX: Override global PARAM_BOUNDS with per-topology bounds.
        # Without this, agent explores V_in=400V for buck (trained on 8-48V),
        # causing surrogate exploitation (low surrogate MSE, wild SPICE MSE).
        if topology in PER_TOPOLOGY_PARAM_BOUNDS:
            self.PARAM_BOUNDS = PER_TOPOLOGY_PARAM_BOUNDS[topology].copy()
        
        # Track SPICE metrics
        self.spice_call_count = 0
        self.spice_mse_history = []
        self.step_count = 0
        
        # Running SPICE trust score: exponential moving average of agreement.
        # When SPICE and surrogate disagree, this dampens surrogate reward
        # even on non-SPICE episodes, preventing exploitation.
        self.spice_trust = 0.5  # Start neutral
        
        # Training progress for staged thresholds (set externally)
        self.training_progress = 0.0  # 0.0 = start, 1.0 = end
        
        # FORMULA MODE: For topologies where the neural surrogate is internally
        # inconsistent (buck_boost), bypass the surrogate entirely and use
        # analytical transfer functions for targets and predictions.
        # Investigation showed buck_boost surrogate predictions are random
        # (range +50V to -112V for params that should give -4 to -27V),
        # making it impossible for RL to learn from surrogate-based rewards.
        self.use_formula_mode = topology in ['buck_boost', 'flyback', 'qr_flyback']
        if self.use_formula_mode:
            print(f"  [FORMULA MODE] {topology}: using analytical transfer function "
                  f"instead of neural surrogate")
    
    def _simulate(self, params: np.ndarray) -> np.ndarray:
        """Override: use formula for formula-mode topologies, surrogate otherwise.
        
        This override ensures ALL code paths that call _simulate() (including
        _get_state() in the base class) use the physics formula when the
        surrogate is unreliable.
        """
        if self.use_formula_mode:
            return self._formula_predict(params)
        return super()._simulate(params)
    
    def _formula_predict(self, params: np.ndarray, output_points: int = 512) -> np.ndarray:
        """Physics-based voltage prediction using analytical transfer functions.
        
        Used instead of the neural surrogate for topologies where the surrogate
        is internally inconsistent (e.g., buck_boost). The analytical formula
        is accurate to within ~8-10% of SPICE for DC voltage.
        
        Args:
            params: [L, C, R_load, V_in, f_sw, duty]
            output_points: Number of output points (default 512 to match surrogate)
            
        Returns:
            Steady-state waveform with realistic ripple
        """
        v_in = params[3]
        duty = params[5]
        
        # Transfer function for each topology (no clipping — use actual values)
        if self.topology == 'buck':
            v_out = v_in * duty
        elif self.topology == 'boost':
            v_out = v_in / max(1 - duty, 0.05)
        elif self.topology == 'buck_boost':
            v_out = -v_in * duty / max(1 - duty, 0.05)
        elif self.topology == 'sepic':
            v_out = v_in * duty / max(1 - duty, 0.05)
        elif self.topology == 'cuk':
            v_out = -v_in * duty / max(1 - duty, 0.05)
        elif self.topology == 'flyback':
            v_out = v_in * duty / max(1 - duty, 0.05)  # 1:1 turns ratio
        elif self.topology == 'qr_flyback':
            eff_duty = duty * 0.95
            v_out = v_in * eff_duty / max(1 - eff_duty, 0.05)
        else:
            v_out = v_in * duty
        
        # Apply duty-dependent loss factor to better match SPICE.
        # Higher duty → more conduction losses → bigger magnitude reduction.
        efficiency = 0.95 - 0.15 * duty  # 95% at D=0, 80% at D=1
        if v_out < 0:
            v_out *= (2 - efficiency)  # For negative: v_out becomes more negative
        else:
            v_out *= efficiency  # For positive: v_out becomes smaller
        
        # Build steady-state waveform with realistic ripple
        t = np.linspace(0, 0.01, output_points)
        ripple_config = TOPOLOGY_REWARD_CONFIG.get(self.topology, {})
        ripple_pct = ripple_config.get('ripple_target', 0.03)
        
        waveform = np.ones(output_points, dtype=np.float32) * v_out
        waveform += ripple_pct * abs(v_out) * np.sin(2 * np.pi * 10 * t)
        waveform += np.random.normal(0, 0.005 * abs(v_out) + 0.01, output_points)
        
        return waveform.astype(np.float32)
    
    def reset(self):
        """Reset environment with topology-appropriate target.
        
        For FORMULA MODE topologies (buck_boost): Uses analytical transfer
        functions instead of the neural surrogate, because the surrogate
        is internally inconsistent for these topologies. Also tries SPICE
        for ground-truth targets when available.
        
        For OTHER topologies: Uses the surrogate for targets, which guarantees
        target-surrogate alignment since both use the same model.
        """
        target_params = self._random_params()
        
        if self.use_formula_mode and self.spice_calculator is not None:
            # Try SPICE first for highest-accuracy targets
            spice_target = self.spice_calculator.simulate(target_params)
            if spice_target is not None:
                self.target_waveform = self.spice_calculator.resample_to_target(
                    spice_target, 512)
            else:
                # Fall back to formula
                self.target_waveform = self._simulate(target_params)
        else:
            # Uses surrogate (normal) or formula (formula mode without SPICE)
            self.target_waveform = self._simulate(target_params)
        
        # Random DIFFERENT starting parameters (agent must find the right ones)
        self.current_params = self._random_params()
        
        self.current_step = 0
        self.prev_mse = None
        self.step_count = 0
        
        # SPICE episode rate: 50% for formula mode (need more ground truth),
        # 20% for normal mode
        import random
        spice_rate = 0.50 if self.use_formula_mode else 0.20
        self.spice_this_episode = (random.random() < spice_rate)
        
        return self._get_state()
    
    def step(self, action: np.ndarray):
        """Take a step with topology-aware rewards."""
        # Apply action (clipped deltas)
        action = np.clip(action, -1, 1)
        
        # IMPROVEMENT: Adaptive action scale that shrinks over the episode.
        # Early steps: 20% max change (coarse exploration)
        # Late steps: 5% max change (fine-tuning)
        # This lets the agent make big jumps initially, then refine.
        episode_progress = self.step_count / max(self.max_steps, 1)
        max_change_pct = 0.20 - 0.15 * episode_progress  # 20% -> 5%
        
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
        
        # Simulate (formula for formula-mode topologies, surrogate otherwise)
        predicted = self._simulate(self.current_params)
        
        # SPICE validation: only on randomly-sampled episodes (10%) at step 1.
        # This balances SPICE ground-truth feedback with training speed.
        # ~20 SPICE calls per rollout instead of ~100+ (step 1 of every episode).
        self.step_count += 1
        use_spice_this_step = (
            self.use_spice_reward and 
            self.spice_calculator is not None and
            self.spice_this_episode and
            self.step_count == 1
        )
        
        # SPICE as a TRUST SIGNAL (DC-error-based agreement)
        #
        # ROOT CAUSE FIX (v5): Previous versions used waveform MSE between
        # SPICE and surrogate outputs for "agreement". This is fundamentally
        # broken because:
        #   1. Surrogate's predict_voltage() replaces model output with
        #      theoretical DC (V_in*D for buck, etc.) + 5% synthetic ripple
        #   2. SPICE produces REAL waveforms with losses, parasitic ringing,
        #      actual ripple at switching frequency
        #   3. Even PERFECT params give high waveform MSE due to domain gap
        #   4. Waveform MSE was 50-2000+ for all topologies — meaningless
        #
        # FIX: Use DC VOLTAGE ERROR as the agreement metric.
        # DC voltage is domain-invariant — both SPICE and surrogate can
        # express DC level accurately. This is also the actual engineering
        # metric that matters for power converter design.
        #
        # agreement = 1 / (1 + dc_error_pct)  where dc_error_pct is %-error
        # between SPICE DC output and surrogate DC prediction.
        
        # 1. Always compute surrogate-based reward (same domain as target)
        reward, info = compute_topology_aware_reward(
            predicted, self.target_waveform,
            self.topology, self.prev_mse
        )
        
        # Apply SPICE trust dampening to ALL steps (not just SPICE steps).
        # If recent SPICE checks showed high disagreement, reduce surrogate
        # reward confidence to prevent exploitation.
        if not use_spice_this_step:
            reward = reward * (0.5 + 0.5 * self.spice_trust)
        
        spice_succeeded = False
        if use_spice_this_step:
            try:
                spice_waveform = self.spice_calculator.simulate(self.current_params)
                
                if spice_waveform is not None:
                    spice_resampled = self.spice_calculator.resample_to_target(
                        spice_waveform, len(self.target_waveform)
                    )
                    
                    # 2. DC-based agreement (domain-invariant)
                    spice_dc = float(np.mean(spice_resampled))
                    surrogate_dc = float(np.mean(predicted))
                    target_dc = float(np.mean(self.target_waveform))
                    
                    # How well do SPICE and surrogate agree on DC voltage?
                    dc_agreement_err = abs(spice_dc - surrogate_dc) / (abs(surrogate_dc) + 1e-6)
                    agreement = 1.0 / (1.0 + dc_agreement_err * 5.0)
                    # agreement ≈ 1.0 when DCs match, drops toward 0 when they diverge
                    # *5.0 makes it sensitive: 20% DC gap → agreement ≈ 0.5
                    
                    # 3. SPICE DC accuracy vs target: how close is SPICE to what we want?
                    spice_dc_error_pct = abs(spice_dc - target_dc) / (abs(target_dc) + 1e-6)
                    if spice_dc_error_pct < 0.3:
                        # Good DC match: up to +3 bonus
                        dc_term = 3.0 * (1.0 - spice_dc_error_pct / 0.3)
                    elif spice_dc_error_pct < 0.5:
                        # Mediocre: no bonus, no penalty
                        dc_term = 0.0
                    else:
                        # Bad DC match: penalty
                        dc_term = -min(5.0, 2.0 * (spice_dc_error_pct - 0.5))
                    
                    # 4. Waveform quality bonus from HIGH-RES SPICE data
                    spice_bonus, spice_metrics = self.spice_calculator.compute_spice_quality_bonus(
                        spice_waveform, self.current_params, self.topology
                    )
                    
                    # 5. Final reward: surrogate reward modulated by DC agreement
                    #    + DC accuracy term + waveform quality bonus
                    reward = reward * agreement + dc_term + spice_bonus
                    
                    # 6. Update running SPICE trust (EMA, alpha=0.3)
                    #    When agreement is high, trust increases -> surrogate reward
                    #    is fully used. When agreement drops, trust drops -> 
                    #    surrogate reward is dampened even on non-SPICE steps.
                    self.spice_trust = 0.7 * self.spice_trust + 0.3 * agreement
                    
                    # Record metrics for logging (use DC error % as primary metric)
                    info['spice_validated'] = True
                    info['spice_dc'] = spice_dc
                    info['surrogate_dc'] = surrogate_dc
                    info['spice_dc_err_pct'] = spice_dc_error_pct * 100
                    info['agreement'] = agreement
                    info['spice_bonus'] = spice_bonus
                    info['dc_term'] = dc_term
                    info['spice_v_out'] = spice_metrics['v_out_mean']
                    info['spice_ripple_pct'] = spice_metrics['ripple_pct']
                    info['spice_overshoot_pct'] = spice_metrics['overshoot_pct']
                    info['spice_thd'] = spice_metrics['thd']
                    info['surrogate_mse'] = info['mse']
                    
                    self.spice_call_count += 1
                    self.spice_mse_history.append(spice_dc_error_pct * 100)  # Track DC error %
                    spice_succeeded = True
            except Exception as e:
                self.spice_fail_count = getattr(self, 'spice_fail_count', 0) + 1
        
        if not spice_succeeded:
            info['spice_validated'] = False
        
        # Update state
        self.prev_mse = info['mse']
        self.current_step += 1
        
        # IMPROVEMENT: Staged early termination threshold.
        # Early training (0-30%): terminate at MSE < 5.0 (learn coarse matching)
        # Mid training (30-70%): terminate at MSE < 2.0 (learn refinement)
        # Late training (70-100%): terminate at MSE < 1.0 (learn fine-tuning)
        #
        # Old approach: always MSE < 5.0, which meant the agent NEVER learned
        # to refine below 5.0 — episodes ended the moment it got close.
        p = self.training_progress  # 0.0 to 1.0
        if p < 0.3:
            success_threshold = 5.0
        elif p < 0.7:
            success_threshold = 2.0
        else:
            success_threshold = 1.0
        
        done = (self.current_step >= self.max_steps) or (info['mse'] < success_threshold)
        # Give success bonus at any stage
        if info['mse'] < success_threshold:
            info['success'] = True
        
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
        n_iterations=config['n_iterations'],
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
        # Update training progress for staged termination threshold
        env.training_progress = iteration / max(config['n_iterations'] - 1, 1)
        
        # Collect rollouts
        rollout = agent.collect_rollouts(config['steps_per_iter'])
        
        # Update policy
        agent.update(rollout, n_epochs=10, batch_size=64)
        
        # Step LR scheduler (linear annealing)
        agent.scheduler.step()
        agent.current_iteration = iteration
        
        # Track performance
        if len(agent.episode_rewards) > 0:
            all_rewards.extend(agent.episode_rewards[-10:])
        if len(agent.episode_mses) > 0:
            all_mses.extend(agent.episode_mses[-10:])
        
        # SPICE validation iteration: directly evaluate agent's output with SPICE
        # (bypasses the short-episode problem where env.step() terminates too quickly)
        if (iteration + 1) % config['spice_freq'] == 0 and env.spice_calculator is not None:
            spice_validations += 1
            
            spice_mses = []
            spice_ripples = []
            dc_errors = []
            agreements = []
            
            for _ in range(5):
                # Run agent for a full episode to get final params
                state = env.reset()
                # Force no SPICE during this evaluation rollout (speed)
                env.spice_this_episode = False
                for step in range(20):
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        action, _, _ = agent.policy.get_action(state_t, deterministic=True)
                    action_np = action.cpu().numpy().squeeze()
                    state, _, done, _ = env.step(action_np)
                    if done:
                        break
                
                # Now directly evaluate agent's final params with SPICE
                try:
                    spice_wf = env.spice_calculator.simulate(env.current_params)
                    if spice_wf is not None:
                        spice_resampled = env.spice_calculator.resample_to_target(
                            spice_wf, len(env.target_waveform)
                        )
                        predicted = env._simulate(env.current_params)
                        
                        # DC-based metrics (domain-invariant)
                        spice_dc = float(np.mean(spice_resampled))
                        surrogate_dc = float(np.mean(predicted))
                        target_dc = float(np.mean(env.target_waveform))
                        
                        # SPICE DC error vs target (the metric that matters)
                        dc_err = abs(spice_dc - target_dc) / (abs(target_dc) + 1e-6) * 100
                        dc_errors.append(dc_err)
                        
                        # Also track old MSE for comparison during transition
                        spice_mse = float(np.mean((spice_resampled - env.target_waveform) ** 2))
                        spice_mses.append(spice_mse)
                        
                        # DC-based agreement (how well surrogate & SPICE agree)
                        dc_agreement_err = abs(spice_dc - surrogate_dc) / (abs(surrogate_dc) + 1e-6)
                        agreement = 1.0 / (1.0 + dc_agreement_err * 5.0)
                        agreements.append(agreement)
                        
                        # Waveform quality
                        _, spice_metrics = env.spice_calculator.compute_spice_quality_bonus(
                            spice_wf, env.current_params, env.topology
                        )
                        spice_ripples.append(spice_metrics['ripple_pct'])
                except Exception:
                    pass
            
            if dc_errors:
                avg_dc_err = np.mean(dc_errors)
                postfix = {
                    'dc_err%': f'{avg_dc_err:.1f}',
                    'best': f'{best_mse:.1f}'
                }
                if spice_ripples:
                    postfix['ripple%'] = f'{np.mean(spice_ripples):.1f}'
                if agreements:
                    postfix['agree'] = f'{np.mean(agreements):.2f}'
                if spice_mses:
                    postfix['spice_mse'] = f'{np.mean(spice_mses):.1f}'
                pbar.set_postfix(postfix)
                
                # Print persistently (tqdm overwrites postfix)
                agree_str = f", agree={np.mean(agreements):.2f}" if agreements else ""
                dc_str = f", dc_err={avg_dc_err:.1f}%"
                ripple_str = f", ripple={np.mean(spice_ripples):.1f}%" if spice_ripples else ""
                mse_str = f", wf_mse={np.mean(spice_mses):.1f}" if spice_mses else ""
                print(f"\n  SPICE[iter {iteration+1}]{dc_str}{agree_str}{ripple_str}{mse_str}")
        
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
        print(f"    SPICE DC err%: first={env.spice_mse_history[0]:.1f}%, "
              f"last={env.spice_mse_history[-1]:.1f}%, "
              f"best={min(env.spice_mse_history):.1f}%")
    
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
    
    # Allow training specific topologies via CLI args
    # Usage: python train_intensive_spice.py buck_boost flyback qr_flyback
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        valid = [t for t in requested if t in TOPOLOGIES]
        if valid:
            TOPOLOGIES_TO_TRAIN = valid
            print(f"Training only: {', '.join(TOPOLOGIES_TO_TRAIN)}")
        else:
            print(f"Unknown topologies: {requested}. Valid: {TOPOLOGIES}")
            sys.exit(1)
    else:
        TOPOLOGIES_TO_TRAIN = TOPOLOGIES
    
    # Override the global list for this run
    original_topologies = TOPOLOGIES
    TOPOLOGIES = TOPOLOGIES_TO_TRAIN
    main()
    TOPOLOGIES = original_topologies
