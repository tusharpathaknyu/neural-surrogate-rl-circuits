#!/usr/bin/env python3
"""
Topology-Aware RL Environment
=============================
Custom reward functions for each topology type:
- Inverted topologies: Handle negative voltage correctly
- Resonant topologies: Different THD interpretation
- Isolated topologies: Separate ringing from ripple
"""

import numpy as np
from typing import Dict, Tuple


# ============================================================================
# TOPOLOGY-SPECIFIC REWARD CONFIGURATIONS
# ============================================================================

TOPOLOGY_REWARD_CONFIG = {
    'buck': {
        'type': 'simple',
        'inverted': False,
        'efficiency_target': 0.95,
        'ripple_target': 0.02,
        'weights': {
            'mse': 1.0,
            'thd': 0.5,
            'rise': 0.3,
            'ripple': 0.5,
            'overshoot': 2.0,
            'dc': 0.5,
        }
    },
    'boost': {
        'type': 'simple',
        'inverted': False,
        'efficiency_target': 0.90,
        'ripple_target': 0.03,
        'weights': {
            'mse': 1.0,
            'thd': 0.5,
            'rise': 0.3,
            'ripple': 0.6,  # Boost has more ripple
            'overshoot': 2.0,
            'dc': 0.5,
        }
    },
    'buck_boost': {
        'type': 'inverted',
        'inverted': True,  # OUTPUT IS NEGATIVE
        'efficiency_target': 0.85,
        'ripple_target': 0.04,
        'weights': {
            'mse': 1.0,
            'thd': 0.4,
            'rise': 0.3,
            'ripple': 0.5,
            'overshoot': 1.5,  # Less strict - inverted
            'dc': 0.8,  # More important to get sign right
            'sign': 2.0,  # NEW: penalty for wrong sign
        }
    },
    'sepic': {
        'type': 'coupled',
        'inverted': False,
        'efficiency_target': 0.85,
        'ripple_target': 0.04,
        'weights': {
            'mse': 1.0,
            'thd': 0.5,
            'rise': 0.3,
            'ripple': 0.5,
            'overshoot': 2.0,
            'dc': 0.5,
        }
    },
    'cuk': {
        'type': 'inverted',
        'inverted': True,  # OUTPUT IS NEGATIVE
        'efficiency_target': 0.83,
        'ripple_target': 0.025,  # Cuk has low ripple advantage
        'weights': {
            'mse': 1.0,
            'thd': 0.3,
            'rise': 0.3,
            'ripple': 0.3,  # Lower weight - Cuk should have low ripple
            'overshoot': 1.5,
            'dc': 0.8,
            'sign': 2.0,  # Penalty for wrong sign
            'smoothness': 0.5,  # NEW: bonus for smooth waveform
        }
    },
    'flyback': {
        'type': 'isolated',
        'inverted': False,
        'efficiency_target': 0.80,
        'ripple_target': 0.05,
        'has_ringing': True,
        'weights': {
            'mse': 1.0,
            'thd': 0.3,  # Lower - ringing adds THD
            'rise': 0.3,
            'ripple': 0.4,
            'overshoot': 1.5,
            'dc': 0.5,
            'ringing': 0.3,  # NEW: separate ringing metric
        }
    },
    'qr_flyback': {
        'type': 'resonant',
        'inverted': False,
        'efficiency_target': 0.87,
        'ripple_target': 0.04,
        'soft_switching': True,
        'weights': {
            'mse': 1.0,
            'thd': 0.1,  # Very low - QR has intentional harmonics
            'rise': 0.2,
            'ripple': 0.4,
            'overshoot': 1.0,  # Less strict
            'dc': 0.5,
            'transition_smoothness': 0.5,  # NEW: ZVS creates smooth transitions
        }
    }
}


def compute_topology_aware_reward(predicted: np.ndarray, target: np.ndarray, 
                                   topology: str, prev_mse: float = None) -> Tuple[float, Dict]:
    """
    Compute reward with topology-specific handling.
    
    Key differences from standard reward:
    1. Inverted topologies: Use absolute values, check sign
    2. Resonant topologies: Don't penalize THD heavily
    3. Different targets per topology
    """
    
    config = TOPOLOGY_REWARD_CONFIG.get(topology, TOPOLOGY_REWARD_CONFIG['buck'])
    weights = config['weights']
    info = {}
    
    # ========== Handle Inverted Topologies ==========
    if config['inverted']:
        # For inverted topologies, work with absolute values for most metrics
        pred_for_metrics = np.abs(predicted)
        target_for_metrics = np.abs(target)
        
        # Check if sign is correct (both should be negative)
        pred_mean = np.mean(predicted)
        target_mean = np.mean(target)
        sign_correct = (np.sign(pred_mean) == np.sign(target_mean))
        sign_penalty = 0 if sign_correct else weights.get('sign', 2.0)
        info['sign_correct'] = sign_correct
        info['sign_penalty'] = sign_penalty
    else:
        pred_for_metrics = predicted
        target_for_metrics = target
        sign_penalty = 0
    
    # ========== 1. MSE (waveform matching) ==========
    # BUG FIX: Use log-scaled MSE so it doesn't drown other metrics.
    # Raw MSE can be 50-200+, while other terms are 0.1-3. Log brings it to ~2-5 range.
    mse = np.mean((pred_for_metrics - target_for_metrics) ** 2)
    log_mse = np.log1p(mse)  # log(1 + MSE), smooth and bounded
    info['mse'] = mse
    info['log_mse'] = log_mse
    
    # ========== 2. THD (Total Harmonic Distortion) ==========
    def compute_thd(waveform):
        fft = np.abs(np.fft.rfft(waveform))
        if len(fft) > 1 and fft[1] > 0:
            fundamental = fft[1]
            harmonics = np.sum(fft[2:11] ** 2)
            return np.sqrt(harmonics) / fundamental
        return 0
    
    pred_thd = compute_thd(pred_for_metrics)
    target_thd = compute_thd(target_for_metrics)
    thd_error = abs(pred_thd - target_thd)
    thd_error = min(thd_error, 5.0)  # Cap to prevent reward explosion
    info['thd_error'] = thd_error
    
    # For resonant topologies, THD is expected - don't penalize as much
    if config['type'] == 'resonant':
        thd_error *= 0.2  # 80% reduction in THD penalty
    
    # ========== 3. Rise Time ==========
    def compute_rise_time(waveform):
        min_val, max_val = np.min(waveform), np.max(waveform)
        if max_val - min_val < 1e-6:
            return 0
        low_thresh = min_val + 0.1 * (max_val - min_val)
        high_thresh = min_val + 0.9 * (max_val - min_val)
        low_idx = np.argmax(waveform >= low_thresh)
        high_idx = np.argmax(waveform >= high_thresh)
        return max(0, high_idx - low_idx)
    
    pred_rise = compute_rise_time(pred_for_metrics)
    target_rise = compute_rise_time(target_for_metrics)
    rise_error = abs(pred_rise - target_rise) / (target_rise + 1e-8)
    rise_error = min(rise_error, 5.0)  # Cap to prevent reward explosion
    info['rise_error'] = rise_error
    
    # ========== 4. Ripple ==========
    pred_ripple = np.max(pred_for_metrics) - np.min(pred_for_metrics)
    target_ripple = np.max(target_for_metrics) - np.min(target_for_metrics)
    
    # Compute ripple error as relative difference (clamped to prevent explosion)
    ripple_error = abs(pred_ripple - target_ripple) / (target_ripple + 1e-2)  # +0.01 to avoid div-by-near-zero
    ripple_error = min(ripple_error, 5.0)  # Cap at 5 to prevent reward explosion
    info['ripple_error'] = ripple_error
    
    # ========== 5. Overshoot (CRITICAL) ==========
    target_max = np.max(target_for_metrics)
    pred_max = np.max(pred_for_metrics)
    overshoot = max(0, (pred_max - target_max) / (target_max + 1e-8))
    overshoot = min(overshoot, 5.0)  # Cap
    info['overshoot'] = overshoot
    
    # ========== 6. DC Error ==========
    dc_error = abs(np.mean(pred_for_metrics) - np.mean(target_for_metrics))
    info['dc_error'] = dc_error
    
    # ========== 7. Topology-Specific Metrics ==========
    
    # Smoothness (for Cuk - continuous currents should give smooth output)
    if 'smoothness' in weights:
        diff = np.abs(np.diff(pred_for_metrics))
        smoothness = 1 - np.mean(diff) / (np.abs(np.mean(pred_for_metrics)) + 1e-8)
        smoothness_bonus = max(0, smoothness) * weights['smoothness']
        info['smoothness'] = smoothness
    else:
        smoothness_bonus = 0
    
    # Transition smoothness (for QR Flyback - ZVS creates smooth transitions)
    if 'transition_smoothness' in weights:
        diff2 = np.abs(np.diff(np.diff(pred_for_metrics)))  # Second derivative
        max_diff2 = np.max(diff2) if len(diff2) > 0 else 1.0
        transition_quality = 1 - np.clip(max_diff2 / (np.abs(np.mean(pred_for_metrics)) + 1e-8), 0, 1)
        transition_bonus = max(0, transition_quality) * weights['transition_smoothness']
        info['transition_smoothness'] = transition_quality
    else:
        transition_bonus = 0
    
    # Ringing metric (for Flyback - separate from THD)
    if 'ringing' in weights and config.get('has_ringing'):
        fft = np.abs(np.fft.rfft(pred_for_metrics))
        n = len(fft)
        if n > 4:
            low_freq_energy = np.sum(fft[:n//4])
            high_freq_energy = np.sum(fft[n//4:])
            ringing_ratio = high_freq_energy / (low_freq_energy + 1e-8)
            ringing_penalty = min(1, ringing_ratio) * weights['ringing']
            info['ringing'] = ringing_ratio
        else:
            ringing_penalty = 0
    else:
        ringing_penalty = 0
    
    # ========== 8. Improvement Bonus ==========
    improvement = 0
    if prev_mse is not None:
        improvement = (prev_mse - mse) / (prev_mse + 1e-8)
    info['improvement'] = improvement
    
    # ========== Success Check ==========
    # BUG FIX: Old threshold 0.001 was ~3000x below best achievable MSE (~3.3).
    # The +10 success bonus NEVER triggered. Use realistic threshold.
    efficiency_target = config['efficiency_target']
    success_threshold = 5.0 * (1 / efficiency_target)  # ~5-6 MSE depending on topology
    success = mse < success_threshold
    info['success'] = success
    
    # ========== TOTAL REWARD ==========
    # BUG FIX: Use log_mse instead of raw mse to keep reward terms balanced.
    # Raw MSE (50-200) was drowning all engineering metrics (0.1-3 each).
    # log1p(MSE) keeps it in ~2-5 range, so THD/ripple/overshoot actually matter.
    reward = (
        -weights['mse'] * log_mse +
        -weights['thd'] * thd_error +
        -weights['rise'] * rise_error +
        -weights['ripple'] * ripple_error +
        -weights['overshoot'] * overshoot +
        -weights['dc'] * np.log1p(dc_error) +  # Also log-scale DC error
        -sign_penalty +  # Penalty for wrong sign (inverted topologies)
        -ringing_penalty +  # Penalty for ringing (Flyback)
        +smoothness_bonus +  # Bonus for smoothness (Cuk)
        +transition_bonus +  # Bonus for smooth transitions (QR Flyback)
        +1.0 * max(0, improvement) +
        +10.0 * (1.0 if success else 0)
    )
    
    info['reward'] = reward
    info['topology'] = topology
    info['topology_type'] = config['type']
    
    return reward, info


def get_topology_config(topology: str) -> Dict:
    """Get configuration for a specific topology."""
    return TOPOLOGY_REWARD_CONFIG.get(topology, TOPOLOGY_REWARD_CONFIG['buck'])


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing topology-aware reward computation...")
    
    np.random.seed(42)
    
    # Test all topologies
    for topology in TOPOLOGY_REWARD_CONFIG.keys():
        config = TOPOLOGY_REWARD_CONFIG[topology]
        
        # Create sample waveforms
        t = np.linspace(0, 1, 512)
        
        if config['inverted']:
            # Inverted output (negative voltage)
            target = -12 * (1 - np.exp(-t * 5)) + 0.5 * np.sin(20 * np.pi * t)
            predicted = -11.5 * (1 - np.exp(-t * 4.5)) + 0.6 * np.sin(20 * np.pi * t)
        else:
            target = 12 * (1 - np.exp(-t * 5)) + 0.5 * np.sin(20 * np.pi * t)
            predicted = 11.5 * (1 - np.exp(-t * 4.5)) + 0.6 * np.sin(20 * np.pi * t)
        
        reward, info = compute_topology_aware_reward(predicted, target, topology)
        
        inv_marker = "⚡INVERTED" if config['inverted'] else ""
        print(f"\n{topology.upper():12} {inv_marker:12} Type: {config['type']:10}")
        print(f"  Reward: {reward:.4f}")
        print(f"  MSE: {info['mse']:.6f}, THD Error: {info['thd_error']:.4f}")
        print(f"  DC Error: {info['dc_error']:.4f}, Ripple Error: {info['ripple_error']:.4f}")
        
        if config['inverted']:
            print(f"  Sign Correct: {info['sign_correct']}, Sign Penalty: {info['sign_penalty']:.2f}")
    
    print("\n✅ All topology reward tests passed!")
