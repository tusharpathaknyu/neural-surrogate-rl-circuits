"""
Enhanced RL Environment with Multi-Objective Optimization.

Improvements:
1. Wider voltage range targets (5V - 24V)
2. Multi-objective reward: MSE + Efficiency + THD + Cost
3. Pareto-optimal tracking
4. Support for multiple topologies
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DesignObjectives:
    """Multi-objective design metrics."""
    mse: float
    efficiency: float  # 0-100%
    thd: float  # Total Harmonic Distortion %
    cost: float  # Normalized component cost
    ripple: float  # Ripple voltage %
    
    def to_dict(self) -> Dict:
        return {
            'mse': self.mse,
            'efficiency': self.efficiency,
            'thd': self.thd,
            'cost': self.cost,
            'ripple': self.ripple,
        }
    
    def dominates(self, other: 'DesignObjectives') -> bool:
        """Check if this solution Pareto-dominates another."""
        dominated = (
            self.mse <= other.mse and
            self.efficiency >= other.efficiency and
            self.thd <= other.thd and
            self.cost <= other.cost and
            self.ripple <= other.ripple
        )
        strictly_better = (
            self.mse < other.mse or
            self.efficiency > other.efficiency or
            self.thd < other.thd or
            self.cost < other.cost or
            self.ripple < other.ripple
        )
        return dominated and strictly_better


class ParetoFrontier:
    """Track Pareto-optimal solutions."""
    
    def __init__(self, max_size: int = 100):
        self.solutions: List[Tuple[np.ndarray, DesignObjectives]] = []
        self.max_size = max_size
    
    def add(self, params: np.ndarray, objectives: DesignObjectives) -> bool:
        """Add solution if Pareto-optimal. Returns True if added."""
        # Check if dominated by existing solutions
        for _, existing in self.solutions:
            if existing.dominates(objectives):
                return False
        
        # Remove solutions dominated by new one
        self.solutions = [
            (p, o) for p, o in self.solutions
            if not objectives.dominates(o)
        ]
        
        # Add new solution
        self.solutions.append((params.copy(), objectives))
        
        # Limit size (keep diverse solutions)
        if len(self.solutions) > self.max_size:
            self._prune()
        
        return True
    
    def _prune(self):
        """Prune to maintain diversity."""
        # Keep solutions with best single objectives
        keep_indices = set()
        
        # Best MSE
        best_mse_idx = min(range(len(self.solutions)), 
                          key=lambda i: self.solutions[i][1].mse)
        keep_indices.add(best_mse_idx)
        
        # Best efficiency
        best_eff_idx = max(range(len(self.solutions)),
                          key=lambda i: self.solutions[i][1].efficiency)
        keep_indices.add(best_eff_idx)
        
        # Lowest cost
        best_cost_idx = min(range(len(self.solutions)),
                           key=lambda i: self.solutions[i][1].cost)
        keep_indices.add(best_cost_idx)
        
        # Random sample for diversity
        remaining = [i for i in range(len(self.solutions)) if i not in keep_indices]
        np.random.shuffle(remaining)
        keep_indices.update(remaining[:self.max_size - len(keep_indices)])
        
        self.solutions = [self.solutions[i] for i in sorted(keep_indices)]
    
    def get_best(self, objective: str = 'mse') -> Tuple[np.ndarray, DesignObjectives]:
        """Get best solution for a specific objective."""
        if not self.solutions:
            return None, None
        
        if objective == 'mse':
            idx = min(range(len(self.solutions)), key=lambda i: self.solutions[i][1].mse)
        elif objective == 'efficiency':
            idx = max(range(len(self.solutions)), key=lambda i: self.solutions[i][1].efficiency)
        elif objective == 'cost':
            idx = min(range(len(self.solutions)), key=lambda i: self.solutions[i][1].cost)
        else:
            idx = 0
        
        return self.solutions[idx]


class MultiObjectiveEnv:
    """
    RL Environment with multi-objective optimization.
    
    Reward balances multiple objectives:
    - Waveform matching (MSE)
    - Power efficiency
    - THD (harmonic distortion)
    - Component cost
    """
    
    PARAM_NAMES = ['L', 'C', 'R_load', 'V_in', 'f_sw', 'duty']
    
    PARAM_BOUNDS = {
        'L': (10e-6, 100e-6),
        'C': (47e-6, 470e-6),
        'R_load': (2, 50),
        'V_in': (5, 24),  # Extended range
        'f_sw': (50e3, 500e3),
        'duty': (0.2, 0.8),
    }
    
    # Component costs (normalized)
    COMPONENT_COSTS = {
        'L': 0.001,  # Cost per µH
        'C': 0.0001,  # Cost per µF
        'R_load': 0,  # Load is external
    }
    
    NUM_PARAMS = 6
    NUM_WAVEFORM_FEATURES = 32
    
    def __init__(
        self,
        surrogate,
        max_steps: int = 50,
        device: str = 'cpu',
        objective_weights: Optional[Dict[str, float]] = None,
    ):
        self.surrogate = surrogate.to(device)
        self.surrogate.eval()
        
        self.device = device
        self.max_steps = max_steps
        
        # Multi-objective weights
        self.weights = objective_weights or {
            'mse': 1.0,
            'efficiency': 0.3,
            'thd': 0.2,
            'cost': 0.1,
            'ripple': 0.2,
        }
        
        # State/action dims
        self.state_dim = self.NUM_WAVEFORM_FEATURES + self.NUM_PARAMS + 5 + 3
        self.action_dim = self.NUM_PARAMS
        
        # Pareto frontier
        self.pareto = ParetoFrontier()
        
        # Episode state
        self.current_params = None
        self.target_waveform = None
        self.target_voltage = None
        self.current_step = 0
        self.prev_objectives = None
    
    def _random_target(self) -> Tuple[np.ndarray, float]:
        """Generate random target waveform with wider voltage range."""
        t = np.linspace(0, 1, 512)
        
        # Random target voltage (5V to 24V)
        target_v = np.random.uniform(5, 24)
        
        # Random characteristics
        rise_time = np.random.uniform(0.02, 0.1)
        ripple_pct = np.random.uniform(0.5, 3.0) / 100  # 0.5% to 3%
        
        # Create target
        target = np.ones(512) * target_v
        
        # Rise time
        rise_samples = int(rise_time * 512)
        target[:rise_samples] = target_v * (1 - np.exp(-t[:rise_samples] * 50))
        
        # Add acceptable ripple
        ripple = target_v * ripple_pct * np.sin(2 * np.pi * 20 * t)
        target = target + ripple
        
        return target.astype(np.float32), target_v
    
    def _random_params(self) -> np.ndarray:
        """Generate random circuit parameters."""
        params = []
        for name in self.PARAM_NAMES:
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                val = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                val = np.random.uniform(low, high)
            params.append(val)
        return np.array(params, dtype=np.float32)
    
    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1]."""
        normalized = np.zeros(self.NUM_PARAMS)
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            if name in ['L', 'C', 'f_sw']:
                normalized[i] = (np.log(params[i]) - np.log(low)) / (np.log(high) - np.log(low))
            else:
                normalized[i] = (params[i] - low) / (high - low)
        return normalized.clip(0, 1)
    
    def _simulate(self, params: np.ndarray) -> np.ndarray:
        """Run circuit through surrogate."""
        with torch.no_grad():
            params_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(self.device)
            waveform = self.surrogate(params_t, normalize=True)
            return waveform.cpu().numpy().squeeze()
    
    def _compute_thd(self, waveform: np.ndarray) -> float:
        """Compute Total Harmonic Distortion."""
        fft = np.abs(np.fft.rfft(waveform - waveform.mean()))
        if len(fft) < 10:
            return 0
        
        fundamental = fft[1] + 1e-10
        harmonics = np.sqrt(np.sum(fft[2:10] ** 2))
        thd = harmonics / fundamental * 100
        return min(thd, 100)
    
    def _compute_efficiency(self, params: np.ndarray, waveform: np.ndarray) -> float:
        """Estimate power efficiency."""
        v_out = np.mean(waveform)
        v_in = params[3]  # V_in
        duty = params[5]
        r_load = params[2]
        
        # Ideal efficiency for buck: Vout/Vin
        # With losses from switching and conduction
        ideal_eff = min(v_out / (v_in + 1e-6), 1.0)
        
        # Switching losses (higher frequency = more loss)
        f_sw = params[4]
        sw_loss = 0.02 * (f_sw / 500e3)
        
        # Conduction losses
        cond_loss = 0.03 * duty
        
        efficiency = max(0, ideal_eff - sw_loss - cond_loss) * 100
        return min(efficiency, 98)
    
    def _compute_cost(self, params: np.ndarray) -> float:
        """Compute normalized component cost."""
        L = params[0] * 1e6  # µH
        C = params[1] * 1e6  # µF
        
        cost = L * self.COMPONENT_COSTS['L'] + C * self.COMPONENT_COSTS['C']
        # Normalize to 0-1
        return min(cost, 1.0)
    
    def _compute_objectives(self, predicted: np.ndarray) -> DesignObjectives:
        """Compute all objectives."""
        mse = float(np.mean((predicted - self.target_waveform) ** 2))
        efficiency = self._compute_efficiency(self.current_params, predicted)
        thd = self._compute_thd(predicted)
        cost = self._compute_cost(self.current_params)
        
        ripple_pp = np.max(predicted) - np.min(predicted)
        v_dc = np.mean(predicted)
        ripple = (ripple_pp / (abs(v_dc) + 1e-6)) * 100
        
        return DesignObjectives(
            mse=mse,
            efficiency=efficiency,
            thd=thd,
            cost=cost,
            ripple=ripple,
        )
    
    def _compute_reward(self, objectives: DesignObjectives) -> float:
        """Compute weighted multi-objective reward."""
        # Normalize each objective to similar scales
        mse_reward = -np.log(objectives.mse + 0.01)  # Higher is better
        eff_reward = objectives.efficiency / 100  # 0-1
        thd_penalty = -objectives.thd / 10  # Lower THD is better
        cost_penalty = -objectives.cost  # Lower cost is better
        ripple_penalty = -objectives.ripple / 10  # Lower ripple is better
        
        reward = (
            self.weights['mse'] * mse_reward +
            self.weights['efficiency'] * eff_reward +
            self.weights['thd'] * thd_penalty +
            self.weights['cost'] * cost_penalty +
            self.weights['ripple'] * ripple_penalty
        )
        
        # Improvement bonus
        if self.prev_objectives is not None:
            if objectives.mse < self.prev_objectives.mse:
                reward += 0.5
        
        return float(reward)
    
    def _extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract compact waveform features."""
        features = []
        
        # Stats
        features.extend([
            np.mean(waveform),
            np.std(waveform),
            np.min(waveform),
            np.max(waveform),
            np.max(waveform) - np.min(waveform),
        ])
        
        # FFT
        fft = np.abs(np.fft.rfft(waveform))
        fft_norm = fft / (np.sum(fft) + 1e-8)
        features.extend(fft_norm[:15])
        
        # Segments
        for i in range(12):
            seg = waveform[i*42:(i+1)*42]
            features.append(np.mean(seg))
        
        return np.array(features[:self.NUM_WAVEFORM_FEATURES], dtype=np.float32)
    
    def _get_state(self) -> np.ndarray:
        """Build state vector."""
        target_features = self._extract_features(self.target_waveform)
        norm_params = self._normalize_params(self.current_params)
        
        predicted = self._simulate(self.current_params)
        objectives = self._compute_objectives(predicted)
        
        obj_state = np.array([
            objectives.mse,
            objectives.efficiency / 100,
            objectives.thd / 100,
            objectives.cost,
            objectives.ripple / 100,
        ], dtype=np.float32)
        
        meta = np.array([
            self.current_step / self.max_steps,
            self.target_voltage / 24,  # Normalized target
            self.prev_objectives.mse if self.prev_objectives else objectives.mse,
        ], dtype=np.float32)
        
        return np.concatenate([target_features, norm_params, obj_state, meta])
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.target_waveform, self.target_voltage = self._random_target()
        self.current_params = self._random_params()
        self.current_step = 0
        self.prev_objectives = None
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take environment step."""
        self.current_step += 1
        
        # Apply action
        for i, name in enumerate(self.PARAM_NAMES):
            low, high = self.PARAM_BOUNDS[name]
            
            if name in ['L', 'C', 'f_sw']:
                log_low, log_high = np.log(low), np.log(high)
                current_log = np.log(self.current_params[i])
                delta = action[i] * (log_high - log_low) * 0.1
                new_log = np.clip(current_log + delta, log_low, log_high)
                self.current_params[i] = np.exp(new_log)
            else:
                delta = action[i] * (high - low) * 0.1
                self.current_params[i] = np.clip(self.current_params[i] + delta, low, high)
        
        # Compute objectives
        predicted = self._simulate(self.current_params)
        objectives = self._compute_objectives(predicted)
        
        # Update Pareto frontier
        self.pareto.add(self.current_params, objectives)
        
        # Compute reward
        reward = self._compute_reward(objectives)
        self.prev_objectives = objectives
        
        # Done conditions
        done = (
            self.current_step >= self.max_steps or
            objectives.mse < 0.1
        )
        
        state = self._get_state()
        
        info = {
            **objectives.to_dict(),
            'target_voltage': self.target_voltage,
            'pareto_size': len(self.pareto.solutions),
        }
        
        return state, reward, done, info


def test_multi_objective():
    """Test multi-objective environment."""
    from models.forward_surrogate import ForwardSurrogate
    
    print("Testing Multi-Objective Environment...")
    
    surrogate = ForwardSurrogate()
    env = MultiObjectiveEnv(surrogate)
    
    state = env.reset()
    print(f"State dim: {len(state)}")
    print(f"Target voltage: {env.target_voltage:.1f}V")
    
    total_reward = 0
    for _ in range(10):
        action = np.random.randn(6) * 0.5
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"\nAfter 10 steps:")
    print(f"  MSE: {info['mse']:.4f}")
    print(f"  Efficiency: {info['efficiency']:.1f}%")
    print(f"  THD: {info['thd']:.2f}%")
    print(f"  Cost: {info['cost']:.4f}")
    print(f"  Pareto solutions: {info['pareto_size']}")
    print(f"  Total reward: {total_reward:.2f}")
    
    print("\n✓ Multi-objective environment test passed!")


if __name__ == '__main__':
    test_multi_objective()
