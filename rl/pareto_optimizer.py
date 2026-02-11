"""
Pareto Multi-Objective Optimization for Circuit Design.

Instead of optimizing a single objective, find the Pareto frontier
of designs that trade off between:
1. Output voltage accuracy
2. Efficiency
3. Cost
4. Size (component values)
5. Ripple

A design is Pareto-optimal if no other design is better in ALL objectives.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class CircuitDesign:
    """Represents a circuit design with its parameters and objectives."""
    params: np.ndarray  # [L, C, R_load, V_in, f_sw, duty]
    topology: str
    
    # Objectives (to minimize unless noted)
    v_out_error: float      # |V_out - V_target| / V_target
    efficiency: float       # Negative because we minimize (want high efficiency)
    cost: float             # Estimated component cost
    size: float             # Proxy for physical size (sum of L and C values)
    ripple: float           # Output ripple voltage
    
    def dominates(self, other: 'CircuitDesign') -> bool:
        """Check if this design Pareto-dominates another."""
        objs_self = [self.v_out_error, -self.efficiency, self.cost, self.size, self.ripple]
        objs_other = [other.v_out_error, -other.efficiency, other.cost, other.size, other.ripple]
        
        # Dominates if at least as good in all objectives and strictly better in at least one
        at_least_as_good = all(s <= o for s, o in zip(objs_self, objs_other))
        strictly_better = any(s < o for s, o in zip(objs_self, objs_other))
        
        return at_least_as_good and strictly_better


class ParetoOptimizer:
    """
    Multi-objective optimization using NSGA-II style evolution.
    
    Finds the Pareto frontier of circuit designs.
    """
    
    # Parameter bounds
    PARAM_BOUNDS = {
        'L': (10e-6, 470e-6),
        'C': (47e-6, 1000e-6),
        'R_load': (2, 100),
        'V_in': (5, 48),
        'f_sw': (50e3, 500e3),
        'duty': (0.2, 0.8),
    }
    
    # Cost model (approximate $/unit)
    COST_MODEL = {
        'L_per_uH': 0.02,     # $0.02 per µH
        'C_per_uF': 0.001,    # $0.001 per µF
        'base_cost': 2.0,     # Base cost (switches, diodes, etc.)
    }
    
    def __init__(self, surrogate, device: str = 'cpu'):
        self.surrogate = surrogate.to(device)
        self.device = device
        self.surrogate.eval()
        
        # Topology mapping
        self.topology_map = {
            'buck': 0, 'boost': 1, 'buck_boost': 2,
            'sepic': 3, 'cuk': 4, 'flyback': 5,
            'qr_flyback': 6
        }
    
    def evaluate_design(self, params: np.ndarray, topology: str,
                       v_target: float) -> CircuitDesign:
        """Evaluate a circuit design on all objectives."""
        
        # Run surrogate
        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(self.device)
        topology_id = torch.tensor([self.topology_map[topology]], device=self.device)
        
        with torch.no_grad():
            waveform, metrics = self.surrogate(params_tensor, topology_id, normalize=True)
        
        waveform = waveform.cpu().numpy().squeeze()
        metrics = metrics.cpu().numpy().squeeze()
        
        # Compute objectives
        v_out = np.mean(waveform)
        v_out_error = abs(v_out - v_target) / v_target
        
        efficiency = float(metrics[0])  # Model predicts efficiency
        
        # Cost estimate
        L, C = params[0], params[1]
        cost = (self.COST_MODEL['base_cost'] + 
                self.COST_MODEL['L_per_uH'] * L * 1e6 +
                self.COST_MODEL['C_per_uF'] * C * 1e6)
        
        # Size proxy (normalized L + C)
        L_norm = (L - self.PARAM_BOUNDS['L'][0]) / (self.PARAM_BOUNDS['L'][1] - self.PARAM_BOUNDS['L'][0])
        C_norm = (C - self.PARAM_BOUNDS['C'][0]) / (self.PARAM_BOUNDS['C'][1] - self.PARAM_BOUNDS['C'][0])
        size = L_norm + C_norm
        
        # Ripple
        ripple = np.max(waveform) - np.min(waveform)
        
        return CircuitDesign(
            params=params.copy(),
            topology=topology,
            v_out_error=v_out_error,
            efficiency=efficiency,
            cost=cost,
            size=size,
            ripple=ripple,
        )
    
    def random_design(self, topology: str) -> np.ndarray:
        """Generate random circuit parameters."""
        params = np.array([
            np.random.uniform(*self.PARAM_BOUNDS['L']),
            np.random.uniform(*self.PARAM_BOUNDS['C']),
            np.random.uniform(*self.PARAM_BOUNDS['R_load']),
            np.random.uniform(*self.PARAM_BOUNDS['V_in']),
            np.random.uniform(*self.PARAM_BOUNDS['f_sw']),
            np.random.uniform(*self.PARAM_BOUNDS['duty']),
        ], dtype=np.float32)
        return params
    
    def mutate(self, params: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
        """Mutate circuit parameters."""
        mutated = params.copy()
        bounds = list(self.PARAM_BOUNDS.values())
        
        for i in range(len(params)):
            if np.random.random() < mutation_rate:
                low, high = bounds[i]
                # Gaussian mutation
                if i in [0, 1, 4]:  # L, C, f_sw - log scale
                    log_val = np.log(params[i])
                    log_low, log_high = np.log(low), np.log(high)
                    log_val += np.random.normal(0, 0.1) * (log_high - log_low)
                    mutated[i] = np.exp(np.clip(log_val, log_low, log_high))
                else:
                    delta = 0.1 * (high - low)
                    mutated[i] = np.clip(params[i] + np.random.normal(0, delta), low, high)
        
        return mutated
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child
    
    def get_pareto_front(self, designs: List[CircuitDesign]) -> List[CircuitDesign]:
        """Extract Pareto-optimal designs from population."""
        pareto_front = []
        
        for design in designs:
            is_dominated = False
            for other in designs:
                if other.dominates(design):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Check if already in front
                is_duplicate = False
                for existing in pareto_front:
                    if np.allclose(design.params, existing.params):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    pareto_front.append(design)
        
        return pareto_front
    
    def optimize(self, topology: str, v_target: float,
                 population_size: int = 100,
                 n_generations: int = 50,
                 verbose: bool = True) -> List[CircuitDesign]:
        """
        Run multi-objective optimization.
        
        Args:
            topology: converter type
            v_target: target output voltage
            population_size: number of designs per generation
            n_generations: number of evolution iterations
            verbose: print progress
            
        Returns:
            List of Pareto-optimal designs
        """
        if verbose:
            print(f"\nOptimizing {topology} for Vout={v_target}V")
            print(f"  Population: {population_size}, Generations: {n_generations}")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            params = self.random_design(topology)
            design = self.evaluate_design(params, topology, v_target)
            population.append(design)
        
        # Evolution loop
        for gen in range(n_generations):
            # Get current Pareto front
            pareto_front = self.get_pareto_front(population)
            
            # Create new population
            new_population = pareto_front.copy()  # Elitism
            
            while len(new_population) < population_size:
                # Tournament selection
                tournament = np.random.choice(len(population), size=4, replace=False)
                parent1 = min([population[i] for i in tournament[:2]], 
                             key=lambda d: d.v_out_error)
                parent2 = min([population[i] for i in tournament[2:]], 
                             key=lambda d: -d.efficiency)
                
                # Crossover and mutation
                child_params = self.crossover(parent1.params, parent2.params)
                child_params = self.mutate(child_params)
                
                child = self.evaluate_design(child_params, topology, v_target)
                new_population.append(child)
            
            population = new_population
            
            if verbose and (gen + 1) % 10 == 0:
                front = self.get_pareto_front(population)
                best_error = min(d.v_out_error for d in front)
                best_eff = max(d.efficiency for d in front)
                print(f"  Gen {gen+1:3d}: Pareto size={len(front):3d}, "
                      f"Best error={best_error:.3f}, Best eff={best_eff:.2%}")
        
        # Final Pareto front
        final_front = self.get_pareto_front(population)
        
        if verbose:
            print(f"\n✓ Found {len(final_front)} Pareto-optimal designs")
        
        return final_front
    
    def plot_pareto_front(self, designs: List[CircuitDesign],
                          objectives: Tuple[str, str] = ('v_out_error', 'efficiency'),
                          save_path: str = None):
        """Plot 2D projection of Pareto front."""
        
        obj_map = {
            'v_out_error': ('Voltage Error (%)', lambda d: d.v_out_error * 100),
            'efficiency': ('Efficiency (%)', lambda d: d.efficiency * 100),
            'cost': ('Cost ($)', lambda d: d.cost),
            'size': ('Size (normalized)', lambda d: d.size),
            'ripple': ('Ripple (V)', lambda d: d.ripple),
        }
        
        x_label, x_fn = obj_map[objectives[0]]
        y_label, y_fn = obj_map[objectives[1]]
        
        x_vals = [x_fn(d) for d in designs]
        y_vals = [y_fn(d) for d in designs]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(x_vals, y_vals, c='blue', s=50, alpha=0.7, edgecolors='black')
        
        # Highlight best designs
        best_x = designs[np.argmin(x_vals)]
        best_y = designs[np.argmax(y_vals) if objectives[1] == 'efficiency' else np.argmin(y_vals)]
        
        ax.scatter([x_fn(best_x)], [y_fn(best_x)], c='red', s=100, marker='*', 
                  label=f'Best {objectives[0]}', zorder=5)
        ax.scatter([x_fn(best_y)], [y_fn(best_y)], c='green', s=100, marker='*',
                  label=f'Best {objectives[1]}', zorder=5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Pareto Front: {objectives[0]} vs {objectives[1]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def recommend_design(self, designs: List[CircuitDesign],
                        priority: str = 'balanced') -> CircuitDesign:
        """
        Recommend a single design from the Pareto front.
        
        Args:
            designs: Pareto-optimal designs
            priority: 'balanced', 'efficiency', 'cost', 'accuracy'
            
        Returns:
            Recommended design
        """
        if priority == 'efficiency':
            return max(designs, key=lambda d: d.efficiency)
        elif priority == 'cost':
            return min(designs, key=lambda d: d.cost)
        elif priority == 'accuracy':
            return min(designs, key=lambda d: d.v_out_error)
        else:  # balanced
            # Normalize objectives and find best weighted sum
            errors = np.array([d.v_out_error for d in designs])
            effs = np.array([d.efficiency for d in designs])
            costs = np.array([d.cost for d in designs])
            
            errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)
            effs_norm = 1 - (effs - effs.min()) / (effs.max() - effs.min() + 1e-6)
            costs_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
            
            scores = 0.4 * errors_norm + 0.4 * effs_norm + 0.2 * costs_norm
            
            return designs[np.argmin(scores)]


def demo_pareto_optimization():
    """Demonstrate Pareto optimization."""
    from models.multi_topology_surrogate import load_trained_model
    
    print("\n" + "="*60)
    print("Pareto Multi-Objective Optimization Demo")
    print("="*60)
    
    # Load model
    surrogate = load_trained_model(device='cpu')
    optimizer = ParetoOptimizer(surrogate)
    
    # Optimize for 12V → 5V buck converter
    pareto_front = optimizer.optimize(
        topology='buck',
        v_target=5.0,
        population_size=50,
        n_generations=30,
        verbose=True
    )
    
    # Show recommendations
    print("\n" + "-"*60)
    print("DESIGN RECOMMENDATIONS:")
    print("-"*60)
    
    for priority in ['balanced', 'efficiency', 'cost', 'accuracy']:
        design = optimizer.recommend_design(pareto_front, priority)
        print(f"\n{priority.upper()} priority:")
        print(f"  L={design.params[0]*1e6:.1f}µH, C={design.params[1]*1e6:.0f}µF")
        print(f"  Vout error: {design.v_out_error*100:.1f}%")
        print(f"  Efficiency: {design.efficiency*100:.1f}%")
        print(f"  Cost: ${design.cost:.2f}")
    
    # Plot
    optimizer.plot_pareto_front(
        pareto_front,
        objectives=('v_out_error', 'efficiency'),
        save_path='checkpoints/pareto_front.png'
    )
    print("\n✓ Saved Pareto front plot to checkpoints/pareto_front.png")


if __name__ == '__main__':
    demo_pareto_optimization()
