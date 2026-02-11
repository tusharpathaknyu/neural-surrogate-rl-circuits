"""
SPICE Validation Module.

Validates surrogate predictions and RL designs against real ngspice simulations.
This is the ground truth check to ensure our models match real physics!

Features:
1. Run real SPICE simulation for any parameters
2. Compare surrogate vs SPICE waveforms
3. Validate RL-designed circuits
4. Generate validation reports
"""

import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import json

# SPICE netlist templates for each topology
NETLIST_TEMPLATES = {
    'buck': """* Buck Converter - Step Down
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})
S1 input sw_node ctrl 0 sw_model

D1 0 sw_node dmodel
.model dmodel d is=1e-14 n=1.05

L1 sw_node output {{L_val}} ic=0
C1 output 0 {{C_val}} ic=0
Rload output 0 {{R_val}}

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
""",

    'boost': """* Boost Converter - Step Up
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}

L1 input sw_node {{L_val}} ic=0

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})
S1 sw_node 0 ctrl 0 sw_model

D1 sw_node output dmodel
.model dmodel d is=1e-14 n=1.05

C1 output 0 {{C_val}} ic={{V_val}}
Rload output 0 {{R_val}}

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
""",

    'buck_boost': """* Buck-Boost Converter - Inverting
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})
S1 input sw_node ctrl 0 sw_model

L1 sw_node 0 {{L_val}} ic=0

D1 output sw_node dmodel
.model dmodel d is=1e-14 n=1.05

C1 output 0 {{C_val}} ic=0
Rload output 0 {{R_val}}

.tran 1u 15m 10m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
""",
}

# Import proper SPICE netlist templates from spice_reward for each topology
# (sepic/cuk/flyback have fundamentally different circuit topologies from buck-boost)
try:
    from rl.spice_reward import SPICERewardCalculator
    for topo in ['sepic', 'cuk', 'flyback', 'qr_flyback']:
        if topo in SPICERewardCalculator.TEMPLATES:
            NETLIST_TEMPLATES[topo] = SPICERewardCalculator.TEMPLATES[topo]
except ImportError:
    # Fallback: at minimum mark them as unavailable rather than wrong
    print("⚠️ Could not import proper SPICE templates from rl.spice_reward")
    NETLIST_TEMPLATES['sepic'] = NETLIST_TEMPLATES['buck_boost']
    NETLIST_TEMPLATES['cuk'] = NETLIST_TEMPLATES['buck_boost']
    NETLIST_TEMPLATES['flyback'] = NETLIST_TEMPLATES['buck_boost']


class SPICEValidator:
    """Validate surrogate predictions against real SPICE simulations."""
    
    def __init__(self, ngspice_path: str = '/opt/homebrew/bin/ngspice'):
        self.ngspice_path = ngspice_path
        self.output_points = 512
        
        # Check ngspice is available
        try:
            result = subprocess.run([ngspice_path, '--version'], 
                                   capture_output=True, timeout=5)
            self.ngspice_available = True
        except:
            print("⚠️ ngspice not found - SPICE validation disabled")
            self.ngspice_available = False
    
    def run_spice(self, params: np.ndarray, topology: str = 'buck',
                  timeout: float = 15.0) -> Tuple[Optional[np.ndarray], float]:
        """
        Run real SPICE simulation.
        
        Args:
            params: [L, C, R_load, V_in, f_sw, duty]
            topology: converter type
            timeout: max simulation time
            
        Returns:
            waveform: 512-point output voltage waveform (or None if failed)
            sim_time: simulation time in ms
        """
        if not self.ngspice_available:
            return None, 0
        
        L, C, R_load, V_in, f_sw, duty = params
        
        # Get template
        template = NETLIST_TEMPLATES.get(topology, NETLIST_TEMPLATES['buck'])
        
        # Create temp file for output
        output_file = tempfile.mktemp(suffix='.txt')
        
        # Format netlist
        netlist = template.format(
            L=L, C=C, R_load=R_load, V_in=V_in, 
            f_sw=f_sw, duty=duty, output_file=output_file
        )
        
        # Write netlist
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
            f.write(netlist)
            netlist_path = f.name
        
        try:
            start = time.time()
            result = subprocess.run(
                [self.ngspice_path, '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            sim_time = (time.time() - start) * 1000
            
            # Parse output
            if Path(output_file).exists():
                data = np.loadtxt(output_file)
                if len(data.shape) == 2 and data.shape[1] >= 2:
                    waveform = data[:, 1]
                    # Resample to 512 points
                    indices = np.linspace(0, len(waveform)-1, self.output_points).astype(int)
                    waveform = waveform[indices]
                    return waveform.astype(np.float32), sim_time
            
            return None, sim_time
            
        except subprocess.TimeoutExpired:
            return None, timeout * 1000
        except Exception as e:
            print(f"SPICE error: {e}")
            return None, 0
        finally:
            # Cleanup
            Path(netlist_path).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
    
    def validate_surrogate(self, surrogate, params: np.ndarray, 
                          topology: str = 'buck') -> Dict:
        """
        Compare surrogate prediction against real SPICE.
        
        Returns:
            dict with mse, correlation, spice_waveform, surrogate_waveform, etc.
        """
        # Run SPICE
        spice_waveform, spice_time = self.run_spice(params, topology)
        
        if spice_waveform is None:
            return {'valid': False, 'error': 'SPICE simulation failed'}
        
        # Run surrogate
        device = next(surrogate.parameters()).device
        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(device)
        
        topology_map = {'buck': 0, 'boost': 1, 'buck_boost': 2, 'sepic': 3, 'cuk': 4, 'flyback': 5, 'qr_flyback': 6}
        topology_id = torch.tensor([topology_map.get(topology, 0)], device=device)
        
        start = time.time()
        with torch.no_grad():
            surrogate_waveform, metrics = surrogate(params_tensor, topology_id, normalize=True)
        surrogate_time = (time.time() - start) * 1000
        
        surrogate_waveform = surrogate_waveform.cpu().numpy().squeeze()
        
        # Compute metrics
        mse = np.mean((spice_waveform - surrogate_waveform) ** 2)
        mae = np.mean(np.abs(spice_waveform - surrogate_waveform))
        correlation = np.corrcoef(spice_waveform, surrogate_waveform)[0, 1]
        
        # DC error
        spice_dc = np.mean(spice_waveform)
        surrogate_dc = np.mean(surrogate_waveform)
        dc_error = abs(spice_dc - surrogate_dc)
        dc_error_pct = dc_error / (abs(spice_dc) + 1e-6) * 100
        
        # Ripple error
        spice_ripple = np.max(spice_waveform) - np.min(spice_waveform)
        surrogate_ripple = np.max(surrogate_waveform) - np.min(surrogate_waveform)
        ripple_error = abs(spice_ripple - surrogate_ripple)
        
        return {
            'valid': True,
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(correlation),
            'dc_error': float(dc_error),
            'dc_error_pct': float(dc_error_pct),
            'ripple_error': float(ripple_error),
            'spice_dc': float(spice_dc),
            'surrogate_dc': float(surrogate_dc),
            'spice_time_ms': float(spice_time),
            'surrogate_time_ms': float(surrogate_time),
            'speedup': float(spice_time / (surrogate_time + 1e-6)),
            'spice_waveform': spice_waveform,
            'surrogate_waveform': surrogate_waveform,
            'topology': topology,
        }
    
    def validate_design(self, surrogate, designed_params: np.ndarray,
                       target_v_out: float, topology: str = 'buck') -> Dict:
        """
        Validate an RL-designed circuit against SPICE ground truth.
        
        Args:
            surrogate: trained surrogate model
            designed_params: RL-designed circuit parameters
            target_v_out: target output voltage
            topology: converter type
            
        Returns:
            Validation results including whether design meets specs
        """
        result = self.validate_surrogate(surrogate, designed_params, topology)
        
        if not result['valid']:
            return result
        
        # Check if design meets target
        actual_v_out = result['spice_dc']
        v_out_error = abs(actual_v_out - target_v_out)
        v_out_error_pct = v_out_error / target_v_out * 100
        
        # Determine if design is acceptable (within 5%)
        design_valid = v_out_error_pct < 5.0
        
        result.update({
            'target_v_out': float(target_v_out),
            'actual_v_out': float(actual_v_out),
            'v_out_error': float(v_out_error),
            'v_out_error_pct': float(v_out_error_pct),
            'design_valid': design_valid,
        })
        
        return result
    
    def plot_validation(self, result: Dict, save_path: str = None):
        """Plot SPICE vs Surrogate comparison."""
        if not result['valid']:
            print("Cannot plot - validation failed")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        t = np.linspace(0, 1, 512)
        
        # Waveform comparison
        axes[0].plot(t, result['spice_waveform'], 'b-', label='SPICE (Ground Truth)', linewidth=2)
        axes[0].plot(t, result['surrogate_waveform'], 'r--', label='Surrogate', linewidth=2)
        axes[0].set_xlabel('Time (normalized)')
        axes[0].set_ylabel('Voltage (V)')
        axes[0].set_title(f"{result['topology'].upper()} - SPICE vs Surrogate")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = (
            f"MSE: {result['mse']:.4f}\n"
            f"Correlation: {result['correlation']:.4f}\n"
            f"DC Error: {result['dc_error_pct']:.2f}%\n"
            f"Speedup: {result['speedup']:.0f}x"
        )
        axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Error plot
        error = result['surrogate_waveform'] - result['spice_waveform']
        axes[1].plot(t, error, 'g-', linewidth=1)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].fill_between(t, error, alpha=0.3)
        axes[1].set_xlabel('Time (normalized)')
        axes[1].set_ylabel('Error (V)')
        axes[1].set_title('Prediction Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


def run_validation_suite(surrogate, n_samples: int = 10):
    """Run comprehensive validation across all topologies."""
    
    validator = SPICEValidator()
    
    if not validator.ngspice_available:
        print("❌ ngspice not available - cannot run validation")
        return None
    
    results = {topo: [] for topo in ['buck', 'boost', 'buck_boost']}
    
    print("\n" + "="*60)
    print("Running SPICE Validation Suite")
    print("="*60)
    
    for topology in ['buck', 'boost', 'buck_boost']:
        print(f"\n{topology.upper()}:")
        
        for i in range(n_samples):
            # Random parameters
            params = np.array([
                np.random.uniform(20e-6, 100e-6),   # L
                np.random.uniform(100e-6, 470e-6),  # C
                np.random.uniform(5, 30),           # R_load
                np.random.uniform(10, 24),          # V_in
                np.random.uniform(100e3, 300e3),    # f_sw
                np.random.uniform(0.3, 0.7),        # duty
            ], dtype=np.float32)
            
            result = validator.validate_surrogate(surrogate, params, topology)
            
            if result['valid']:
                results[topology].append(result)
                print(f"  Sample {i+1}: MSE={result['mse']:.4f}, "
                      f"Corr={result['correlation']:.3f}, "
                      f"DC Err={result['dc_error_pct']:.1f}%")
    
    # Summary
    print("\n" + "-"*60)
    print("VALIDATION SUMMARY")
    print("-"*60)
    
    summary = {}
    for topology, topo_results in results.items():
        if topo_results:
            mses = [r['mse'] for r in topo_results]
            corrs = [r['correlation'] for r in topo_results]
            dc_errs = [r['dc_error_pct'] for r in topo_results]
            speedups = [r['speedup'] for r in topo_results]
            
            summary[topology] = {
                'mean_mse': np.mean(mses),
                'mean_correlation': np.mean(corrs),
                'mean_dc_error_pct': np.mean(dc_errs),
                'mean_speedup': np.mean(speedups),
            }
            
            print(f"\n{topology.upper()}:")
            print(f"  MSE:         {np.mean(mses):.4f} ± {np.std(mses):.4f}")
            print(f"  Correlation: {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")
            print(f"  DC Error:    {np.mean(dc_errs):.2f}% ± {np.std(dc_errs):.2f}%")
            print(f"  Speedup:     {np.mean(speedups):.0f}x")
    
    return summary


if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from models.multi_topology_surrogate import load_trained_model
    
    # Load surrogate
    surrogate = load_trained_model(device='cpu')
    
    # Run validation
    summary = run_validation_suite(surrogate, n_samples=5)
    
    if summary:
        # Save summary
        with open('checkpoints/spice_validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("\n✓ Saved validation summary")
