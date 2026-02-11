"""
Simple RL Environment for Power Electronics Circuit Design.
No external dependencies - just numpy and torch.

Phase C: The RL agent uses this environment to learn circuit design.
Updated: Topology-aware reward system for proper handling of inverted/resonant topologies.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.forward_surrogate import ForwardSurrogate

# Import topology-aware rewards
try:
    from rl.topology_rewards import compute_topology_aware_reward, get_topology_config
    TOPOLOGY_AWARE_REWARDS = True
except ImportError:
    TOPOLOGY_AWARE_REWARDS = False
    print("⚠️ Topology-aware rewards not available, using standard rewards")


class CircuitDesignEnv:
    """
    Simple RL Environment for circuit design.
    
    The agent's goal: find circuit parameters that produce a target waveform.
    
    State: [target_waveform_features, current_params, error_signals]
    Action: [ΔL, ΔC, ΔR, ΔV_in, Δf_sw, ΔD] (changes to circuit params)
    Reward: Based on THD, Rise Time, Overshoot, etc.
    """
    
    # Parameter bounds (physical constraints)
    PARAM_BOUNDS = {
        'L': (10e-6, 100e-6),       # 10µH to 100µH
        'C': (47e-6, 470e-6),       # 47µF to 470µF
        'R_load': (2, 50),          # 2Ω to 50Ω
        'V_in': (10, 24),           # 10V to 24V
        'f_sw': (50e3, 500e3),      # 50kHz to 500kHz
        'duty': (0.2, 0.8),         # 20% to 80%
    }
    
    PARAM_NAMES = ['L', 'C', 'R_load', 'V_in', 'f_sw', 'duty']
    NUM_PARAMS = 6
    NUM_WAVEFORM_FEATURES = 32
    
    def __init__(
        self,
        surrogate,
        max_steps: int = 50,
        device: str = 'cpu',
        target_waveforms: Optional[np.ndarray] = None,
        topology: str = 'buck',
        use_spice_reward: bool = False,
        spice_validation_freq: int = 10,
    ):
        self.surrogate = surrogate.to(device)
        self.surrogate.eval()
        for p in self.surrogate.parameters():
            p.requires_grad = False
            
        self.device = device
        self.max_steps = max_steps
        self.target_waveforms = target_waveforms
        self._topology = topology  # Use private variable
        
        # SPICE reward integration
        self.use_spice_reward = use_spice_reward
        self.spice_validation_freq = spice_validation_freq
        self.spice_calculator = None
        self._init_spice_calculator(topology)
        
        # Check if surrogate is multi-topology
        self.is_multi_topology = hasattr(surrogate, 'topology_embedding')
        
        # State and action dimensions
        # State = [target_features(32) + current_pred_features(32) + params(6) + error(3)] = 73
        self.state_dim = self.NUM_WAVEFORM_FEATURES * 2 + self.NUM_PARAMS + 3
        self.action_dim = self.NUM_PARAMS
        
        # Episode state
    
    def _init_spice_calculator(self, topology: str):
        """Initialize SPICE calculator for a given topology."""
        if self.use_spice_reward:
            try:
                from rl.spice_reward import SPICERewardCalculator
                self.spice_calculator = SPICERewardCalculator(topology=topology)
                if not self.spice_calculator.ngspice_available:
                    print(f"Warning: ngspice not available, disabling SPICE rewards")
                    self.use_spice_reward = False
            except ImportError:
                print("Warning: Could not import SPICERewardCalculator")
                self.use_spice_reward = False
    
    @property
    def topology(self) -> str:
        return self._topology
    
    @topology.setter
    def topology(self, value: str):
        """Update topology and reinitialize SPICE calculator if needed."""
        if value != self._topology:
            self._topology = value
            if self.use_spice_reward:
                self._init_spice_calculator(value)
        self.current_params = None
        self.target_waveform = None
        self.current_step = 0
        self.prev_mse = None
    
    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range.
        
        BUG FIX: Clamp values to PARAM_BOUNDS before normalizing.
        Previously, if V_in exceeded bounds (e.g., 36 with max=24),
        normalized value would be >1, corrupting policy network inputs.
        """
        normalized = np.zeros(self.NUM_PARAMS)
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            val = np.clip(params[i], low, high)  # Ensure within bounds
            if name in ['L', 'C', 'f_sw']:
                normalized[i] = (np.log(val) - np.log(low)) / (np.log(high) - np.log(low))
            else:
                normalized[i] = (val - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0)  # Final safety clamp
    
    def _extract_waveform_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract compact features from waveform for state representation."""
        features = []
        
        # Statistical features (5)
        features.append(np.mean(waveform))
        features.append(np.std(waveform))
        features.append(np.min(waveform))
        features.append(np.max(waveform))
        features.append(np.max(waveform) - np.min(waveform))  # Peak-to-peak
        
        # FFT features - first 15 frequency components
        fft = np.abs(np.fft.rfft(waveform))
        fft_normalized = fft / (np.sum(fft) + 1e-8)
        features.extend(fft_normalized[:15])
        
        # Time-domain segment means (12)
        num_segments = 12
        segment_size = len(waveform) // num_segments
        for i in range(num_segments):
            segment = waveform[i*segment_size:(i+1)*segment_size]
            features.append(np.mean(segment))
        
        return np.array(features[:self.NUM_WAVEFORM_FEATURES], dtype=np.float32)
    
    def _simulate(self, params: np.ndarray) -> np.ndarray:
        """Run circuit through surrogate model - THE FAST PART!
        
        Returns denormalized voltage waveform using physics-based scaling.
        """
        with torch.no_grad():
            params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            if self.is_multi_topology:
                # Multi-topology model needs topology_ids
                topology_map = {'buck': 0, 'boost': 1, 'buck_boost': 2, 'sepic': 3, 'cuk': 4, 'flyback': 5, 'qr_flyback': 6}
                topology_id = topology_map.get(self.topology, 0)
                topology_ids = torch.tensor([topology_id], dtype=torch.long, device=self.device)
                
                # Use predict_voltage for denormalized output
                if hasattr(self.surrogate, 'predict_voltage'):
                    waveform, _ = self.surrogate.predict_voltage(params_tensor, self.topology)
                else:
                    waveform, _ = self.surrogate(params_tensor, topology_ids, normalize=True)
            else:
                # Legacy single-topology model
                waveform = self.surrogate(params_tensor, normalize=True)
            
            return waveform.cpu().numpy().squeeze()
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector for the agent.
        
        BUG FIX: Now includes BOTH target AND current prediction features,
        so the agent knows WHERE its prediction differs from target (not just 
        a scalar MSE). This gives directional gradient information.
        
        State = [target_features(32), current_pred_features(32), norm_params(6), error_signals(3)] = 73 dims
        """
        target_features = self._extract_waveform_features(self.target_waveform)
        norm_params = self._normalize_params(self.current_params)
        
        current_waveform = self._simulate(self.current_params)
        current_features = self._extract_waveform_features(current_waveform)
        mse = np.mean((current_waveform - self.target_waveform) ** 2)
        
        error_signal = np.array([
            np.log1p(mse),  # Log-scale MSE to prevent huge values dominating
            self.current_step / self.max_steps,
            np.log1p(self.prev_mse) if self.prev_mse else np.log1p(mse),
        ], dtype=np.float32)
        
        return np.concatenate([target_features, current_features, norm_params, error_signal]).astype(np.float32)
    
    def _compute_reward(self, predicted: np.ndarray, target: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute reward based on engineering metrics.
        
        Now uses TOPOLOGY-AWARE reward system:
        - Inverted topologies (Buck-Boost, Cuk): Handle negative voltage correctly
        - Resonant topologies (QR Flyback): Different THD interpretation
        - Different efficiency targets per topology
        """
        
        # Use topology-aware rewards if available
        if TOPOLOGY_AWARE_REWARDS:
            return compute_topology_aware_reward(
                predicted, target, self.topology, self.prev_mse
            )
        
        # Fallback to standard rewards (legacy)
        info = {}
        
        # 1. MSE - basic waveform matching (log-scaled to prevent dominance)
        mse = np.mean((predicted - target) ** 2)
        log_mse = np.log1p(mse)
        info['mse'] = mse
        
        # 2. THD comparison
        def compute_thd(waveform):
            fft = np.abs(np.fft.rfft(waveform))
            if len(fft) > 1 and fft[1] > 0:
                fundamental = fft[1]
                harmonics = np.sum(fft[2:11] ** 2)
                return np.sqrt(harmonics) / fundamental
            return 0
        
        pred_thd = compute_thd(predicted)
        target_thd = compute_thd(target)
        thd_error = abs(pred_thd - target_thd)
        info['thd_error'] = thd_error
        
        # 3. Rise time comparison (10-90%)
        def compute_rise_time(waveform):
            min_val, max_val = np.min(waveform), np.max(waveform)
            if max_val - min_val < 1e-6:
                return 0
            low_thresh = min_val + 0.1 * (max_val - min_val)
            high_thresh = min_val + 0.9 * (max_val - min_val)
            low_idx = np.argmax(waveform >= low_thresh)
            high_idx = np.argmax(waveform >= high_thresh)
            return max(0, high_idx - low_idx)
        
        pred_rise = compute_rise_time(predicted)
        target_rise = compute_rise_time(target)
        rise_error = abs(pred_rise - target_rise) / (target_rise + 1e-8)
        info['rise_error'] = rise_error
        
        # 4. Ripple comparison
        pred_ripple = np.max(predicted) - np.min(predicted)
        target_ripple = np.max(target) - np.min(target)
        ripple_error = abs(pred_ripple - target_ripple) / (target_ripple + 1e-8)
        info['ripple_error'] = ripple_error
        
        # 5. Overshoot penalty (CRITICAL - can fry components!)
        target_max = np.max(target)
        pred_max = np.max(predicted)
        overshoot = max(0, (pred_max - target_max) / (target_max + 1e-8))
        info['overshoot'] = overshoot
        
        # 6. DC error
        dc_error = abs(np.mean(predicted) - np.mean(target))
        info['dc_error'] = dc_error
        
        # 7. Improvement bonus
        improvement = 0
        if self.prev_mse is not None:
            improvement = (self.prev_mse - mse) / (self.prev_mse + 1e-8)
        info['improvement'] = improvement
        
        # Success check (BUG FIX: was 0.001, now realistic)
        success = mse < 5.0
        info['success'] = success
        
        # TOTAL REWARD - weighted combination (BUG FIX: use log_mse)
        reward = (
            -1.0 * log_mse +              # Match waveform shape (log-scaled)
            -0.5 * thd_error +        # Match harmonic content
            -0.3 * rise_error +       # Match rise time
            -0.5 * ripple_error +     # Match ripple
            -2.0 * overshoot +        # Heavy penalty for overshoot
            -0.5 * np.log1p(dc_error) + # Match DC level (log-scaled)
            1.0 * max(0, improvement) + # Bonus for improving
            10.0 * (1.0 if success else 0)  # Big bonus for success
        )
        
        info['reward'] = reward
        return reward, info
    
    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action (parameter changes) to current params."""
        new_params = self.current_params.copy()
        max_change_pct = 0.2  # Maximum 20% change per step
        
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            
            if name in ['L', 'C', 'f_sw']:
                # Log-scale adjustment for values spanning orders of magnitude
                log_val = np.log(self.current_params[i])
                log_range = np.log(high) - np.log(low)
                log_val += action[i] * max_change_pct * log_range
                new_params[i] = np.exp(np.clip(log_val, np.log(low), np.log(high)))
            else:
                # Linear adjustment
                range_val = high - low
                new_params[i] += action[i] * max_change_pct * range_val
                new_params[i] = np.clip(new_params[i], low, high)
        
        return new_params
    
    def _random_params(self) -> np.ndarray:
        """Generate random valid parameters."""
        params = np.zeros(self.NUM_PARAMS)
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                params[i] = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                params[i] = np.random.uniform(low, high)
        return params
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        # Sample target waveform
        if self.target_waveforms is not None:
            if self.target_waveforms.ndim == 1:
                # Single waveform provided
                self.target_waveform = self.target_waveforms.copy()
            else:
                # Multiple waveforms - sample one
                idx = np.random.randint(len(self.target_waveforms))
                self.target_waveform = self.target_waveforms[idx].copy()
        else:
            random_params = self._random_params()
            self.target_waveform = self._simulate(random_params)
        
        # Random starting parameters (agent must find the right ones)
        self.current_params = self._random_params()
        self.current_step = 0
        self.prev_mse = None
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step in the environment.
        
        Args:
            action: [ΔL, ΔC, ΔR, ΔV_in, Δf_sw, ΔDuty] in range [-1, 1]
            
        Returns:
            state: New observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Debug information
        """
        self.current_step += 1
        
        # Apply action to get new circuit parameters
        self.current_params = self._apply_action(action)
        
        # Simulate with new params (FAST - uses surrogate!)
        predicted = self._simulate(self.current_params)
        
        # Compute reward using engineering metrics
        reward, info = self._compute_reward(predicted, self.target_waveform)
        
        # Optionally validate with SPICE for ground-truth reward
        if self.use_spice_reward and self.spice_calculator is not None:
            # Use SPICE every N steps (expensive but accurate)
            if self.current_step % self.spice_validation_freq == 0:
                spice_reward, spice_info = self._compute_spice_reward()
                if spice_info.get('spice_used', False):
                    # Blend surrogate and SPICE rewards (trust SPICE more)
                    blend_factor = 0.7  # 70% SPICE, 30% surrogate
                    reward = blend_factor * spice_reward + (1 - blend_factor) * reward
                    info['spice_reward'] = spice_reward
                    info['spice_vout'] = spice_info.get('spice_vout', 0)
                    info['spice_used'] = True
        
        # Track for improvement bonus
        self.prev_mse = info['mse']
        
        # Check termination
        done = info['success'] or self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, info
    
    def _compute_spice_reward(self) -> Tuple[float, Dict]:
        """Compute reward using actual SPICE simulation (ground truth)."""
        if self.spice_calculator is None:
            return 0.0, {'spice_used': False}
        
        # Get the surrogate's prediction for fallback
        predicted = self._simulate(self.current_params)
        
        # Call SPICE reward calculator with current params
        reward, info = self.spice_calculator.compute_reward(
            params=self.current_params,
            target_waveform=self.target_waveform,
            surrogate_waveform=predicted
        )
        
        # Add spice_used flag
        info['spice_used'] = info.get('source', 'none') == 'spice'
        if 'spice_vout' not in info and 'vout' in info:
            info['spice_vout'] = info['vout']
        
        return reward, info


if __name__ == '__main__':
    print("Testing Circuit Design Environment...")
    
    surrogate = ForwardSurrogate()
    env = CircuitDesignEnv(surrogate)
    
    state = env.reset()
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    
    for i in range(5):
        action = np.random.uniform(-1, 1, env.action_dim)
        state, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, mse={info['mse']:.6f}")
        if done:
            break
    
    print("✓ Environment test passed!")
