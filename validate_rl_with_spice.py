"""
SPICE Validation for Multi-Topology RL Agents.

Validates trained RL agents by comparing their designed circuits
against real ngspice simulations. This is the ground truth check.

Usage:
    python validate_rl_with_spice.py [--topology buck|boost|all] [--n-tests 10]
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from models.multi_topology_surrogate import load_trained_model
from rl.ppo_agent import PPOAgent
from rl.environment import CircuitDesignEnv


# Device
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Topology enum
class Topology(Enum):
    BUCK = 0
    BOOST = 1
    BUCK_BOOST = 2
    SEPIC = 3
    CUK = 4
    FLYBACK = 5


# SPICE netlist templates for each topology
SPICE_TEMPLATES = {
    Topology.BUCK: """* Buck Converter Validation
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
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

    Topology.BOOST: """* Boost Converter Validation
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

L1 input sw_node {{L_val}} ic=0

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
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

    Topology.BUCK_BOOST: """* Buck-Boost Converter Validation
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 input sw_node ctrl 0 sw_model

L1 sw_node 0 {{L_val}} ic=0

D1 output sw_node dmodel
.model dmodel d is=1e-14 n=1.05

C1 output 0 {{C_val}} ic=0
Rload output 0 {{R_val}}

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} abs(v(output))
.endc
.end
""",

    Topology.SEPIC: """* SEPIC Converter Validation
.param L1_val={L}
.param L2_val={L}
.param Cc_val={C_couple}
.param Co_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

L1 input n1 {{L1_val}} ic=0
Cc n1 n2 {{Cc_val}} ic=0

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 n1 0 ctrl 0 sw_model

L2 n2 0 {{L2_val}} ic=0

D1 n2 output dmodel
.model dmodel d is=1e-14 n=1.05

Co output 0 {{Co_val}} ic={{V_val}}
Rload output 0 {{R_val}}

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
""",

    Topology.CUK: """* Cuk Converter Validation
.param L1_val={L}
.param L2_val={L}
.param Cc_val={C_couple}
.param Co_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

L1 input n1 {{L1_val}} ic=0

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 n1 0 ctrl 0 sw_model

Cc n1 n2 {{Cc_val}} ic={{V_val}}

D1 0 n2 dmodel
.model dmodel d is=1e-14 n=1.05

L2 n2 output {{L2_val}} ic=0
Co output 0 {{Co_val}} ic=0
Rload output 0 {{R_val}}

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} abs(v(output))
.endc
.end
""",

    Topology.FLYBACK: """* Flyback Converter Validation
.param Lp_val={L}
.param n={n_ratio}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 input pri_top ctrl 0 sw_model
Lpri pri_top 0 {{Lp_val}} ic=0

Lsec sec_top 0 {{Lp_val*n*n}} ic=0
K1 Lpri Lsec 0.98

D1 0 sec_top dmodel
.model dmodel d is=1e-14 n=1.05

C1 sec_top output {{C_val}} ic=10
Rload output 0 {{R_val}}
Rref output 0 1e9

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} v(sec_top)
.endc
.end
""",
}


# Parameter ranges for each topology (from training)
PARAM_RANGES = {
    Topology.BUCK: {
        'L': (10e-6, 100e-6),
        'C': (47e-6, 470e-6),
        'R_load': (2, 50),
        'V_in': (8, 48),
        'f_sw': (50e3, 500e3),
        'duty': (0.1, 0.9),
    },
    Topology.BOOST: {
        'L': (22e-6, 220e-6),
        'C': (100e-6, 1000e-6),
        'R_load': (10, 100),
        'V_in': (3.3, 24),
        'f_sw': (50e3, 300e3),
        'duty': (0.2, 0.7),  # Limited to avoid instability
    },
    Topology.BUCK_BOOST: {
        'L': (47e-6, 470e-6),
        'C': (100e-6, 1000e-6),
        'R_load': (5, 50),
        'V_in': (5, 36),
        'f_sw': (50e3, 200e3),
        'duty': (0.2, 0.8),
    },
    Topology.SEPIC: {
        'L': (22e-6, 220e-6),
        'C': (100e-6, 1000e-6),
        'C_couple': (1e-6, 10e-6),
        'R_load': (10, 100),
        'V_in': (5, 24),
        'f_sw': (50e3, 200e3),
        'duty': (0.2, 0.8),
    },
    Topology.CUK: {
        'L': (47e-6, 470e-6),
        'C': (100e-6, 1000e-6),
        'C_couple': (1e-6, 22e-6),
        'R_load': (5, 50),
        'V_in': (5, 24),
        'f_sw': (50e3, 200e3),
        'duty': (0.2, 0.8),
    },
    Topology.FLYBACK: {
        'L': (100e-6, 1000e-6),
        'n_ratio': (0.1, 2.0),
        'C': (100e-6, 1000e-6),
        'R_load': (5, 100),
        'V_in': (12, 48),
        'f_sw': (50e3, 150e3),
        'duty': (0.2, 0.5),
    },
}


# Agent hidden dims (must match training)
TOPOLOGY_AGENT_CONFIG = {
    'buck': {'hidden_dim': 256},
    'boost': {'hidden_dim': 512},
    'buck_boost': {'hidden_dim': 256},
    'sepic': {'hidden_dim': 512},
    'cuk': {'hidden_dim': 256},
    'flyback': {'hidden_dim': 512},
}

# Topology name to ID mapping
TOPOLOGY_NAME_TO_ID = {
    'buck': 0,
    'boost': 1,
    'buck_boost': 2,
    'sepic': 3,
    'cuk': 4,
    'flyback': 5,
}


def check_ngspice():
    """Check if ngspice is installed."""
    try:
        result = subprocess.run(['ngspice', '--version'], 
                                capture_output=True, text=True, timeout=5)
        return True
    except:
        return False


def run_spice_simulation(topology: Topology, params: Dict[str, float], 
                         timeout: float = 15) -> Tuple[Optional[np.ndarray], float]:
    """Run ngspice simulation and return waveform."""
    template = SPICE_TEMPLATES[topology]
    
    # Create temp output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as out_f:
        output_file = out_f.name
    
    # Format netlist
    format_params = {**params, 'output_file': output_file}
    
    # Add coupling capacitor for SEPIC/CUK if not present
    if topology in [Topology.SEPIC, Topology.CUK] and 'C_couple' not in params:
        format_params['C_couple'] = 4.7e-6
    
    # Add turns ratio for Flyback if not present
    if topology == Topology.FLYBACK and 'n_ratio' not in params:
        format_params['n_ratio'] = 1.0
    
    netlist = template.format(**format_params)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
        f.write(netlist)
        netlist_path = f.name
    
    try:
        start = time.time()
        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        spice_time = (time.time() - start) * 1000
        
        if Path(output_file).exists():
            try:
                data = np.loadtxt(output_file)
                if len(data.shape) == 2 and data.shape[1] >= 2:
                    waveform = data[:, 1]
                    # Resample to 512 points
                    indices = np.linspace(0, len(waveform)-1, 512).astype(int)
                    waveform = waveform[indices]
                    
                    # Clean up NaN/Inf
                    waveform = np.nan_to_num(waveform, nan=0.0, posinf=100.0, neginf=-100.0)
                    
                    return waveform, spice_time
            except Exception as e:
                print(f"  Parse error: {e}")
                
    except subprocess.TimeoutExpired:
        print(f"  SPICE timeout ({timeout}s)")
    except Exception as e:
        print(f"  SPICE error: {e}")
    finally:
        # Cleanup
        Path(netlist_path).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)
    
    return None, 0


def calculate_expected_vout(topology: Topology, v_in: float, duty: float, 
                            n_ratio: float = 1.0) -> float:
    """Calculate theoretical output voltage."""
    eps = 0.01  # Avoid division by zero
    
    if topology == Topology.BUCK:
        return v_in * duty
    elif topology == Topology.BOOST:
        return v_in / (1 - duty + eps)
    elif topology in [Topology.BUCK_BOOST, Topology.CUK]:
        return abs(v_in * duty / (1 - duty + eps))
    elif topology == Topology.SEPIC:
        return v_in * duty / (1 - duty + eps)
    elif topology == Topology.FLYBACK:
        return v_in * n_ratio * duty / (1 - duty + eps)
    return v_in * duty


def load_rl_agent(topology_name: str, surrogate, device: str) -> Optional[PPOAgent]:
    """Load trained RL agent for a topology."""
    checkpoint_path = Path(f"checkpoints/rl_agent_{topology_name}.pt")
    
    if not checkpoint_path.exists():
        print(f"  Agent not found: {checkpoint_path}")
        return None
    
    config = TOPOLOGY_AGENT_CONFIG.get(topology_name, {'hidden_dim': 256})
    topology_id = TOPOLOGY_NAME_TO_ID[topology_name]
    
    # Create environment for this topology
    env = CircuitDesignEnv(surrogate, device=device, topology=topology_id)
    
    # Create agent with correct architecture
    agent = PPOAgent(
        env=env,
        hidden_dim=config['hidden_dim'],
        lr=1e-4,
        device=device
    )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'policy' in checkpoint:
            agent.policy.load_state_dict(checkpoint['policy'])
        elif 'actor_state_dict' in checkpoint:
            agent.policy.load_state_dict(checkpoint['actor_state_dict'])
        print(f"  Loaded agent from {checkpoint_path}")
        return agent
    except Exception as e:
        print(f"  Error loading agent: {e}")
        return None


def create_target_for_topology(topology: Topology, target_voltage: float) -> np.ndarray:
    """Create a target waveform for the topology."""
    t = np.linspace(0, 1, 512)
    rise_time = 0.05
    
    # Exponential rise to steady state
    waveform = target_voltage * (1 - np.exp(-t / rise_time))
    
    # Add small ripple (realistic)
    ripple = 0.02 * target_voltage * np.sin(2 * np.pi * 20 * t)
    waveform[int(0.1*512):] += ripple[int(0.1*512):]
    
    return waveform


def run_rl_optimization(agent: PPOAgent, surrogate, topology: Topology, 
                        topology_id: int, target_waveform: np.ndarray,
                        n_steps: int = 50) -> np.ndarray:
    """Run RL agent to optimize circuit parameters."""
    ranges = PARAM_RANGES[topology]
    
    # Reset environment with target
    agent.env.target_waveform = target_waveform
    state = agent.env.reset()
    
    best_params = agent.env.current_params.copy()
    best_mse = float('inf')
    
    for step in range(n_steps):
        # Get action from agent
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _, _ = agent.policy.get_action(state_tensor, deterministic=True)
        action = action.cpu().numpy().squeeze()
        
        # Take step in environment
        next_state, reward, done, info = agent.env.step(action)
        
        # Track best
        mse = info.get('mse', float('inf'))
        if mse < best_mse:
            best_mse = mse
            best_params = agent.env.current_params.copy()
        
        state = next_state
        if done:
            break
    
    return best_params


@dataclass
class ValidationResult:
    topology: str
    target_voltage: float
    rl_params: Dict[str, float]
    surrogate_vout: float
    spice_vout: float
    surrogate_mse: float
    spice_mse: float
    voltage_error_percent: float
    spice_time_ms: float


# NOTE: The surrogate outputs normalized waveforms (trained on waveforms/max).
# At inference, we don't have the scale factor, so surrogate predictions 
# are in [-1, 1] range. SPICE validation is the ground truth check.
# A future improvement would be to train the surrogate to predict actual voltages.


def validate_topology(topology_name: str, surrogate, n_tests: int = 5) -> List[ValidationResult]:
    """Validate a single topology's RL agent against SPICE."""
    topology = Topology[topology_name.upper()]
    topology_id = topology.value
    ranges = PARAM_RANGES[topology]
    
    print(f"\n{'='*60}")
    print(f"Validating {topology_name.upper()} Topology")
    print(f"{'='*60}")
    
    # Load agent
    agent = load_rl_agent(topology_name, surrogate, DEVICE)
    if agent is None:
        print(f"  Skipping - no trained agent found")
        return []
    
    results = []
    
    for test_idx in range(n_tests):
        # Random target voltage (appropriate for topology)
        v_in = np.random.uniform(*ranges['V_in'])
        duty = np.random.uniform(*ranges['duty'])
        n_ratio = ranges.get('n_ratio', (1.0, 1.0))
        n_ratio = np.random.uniform(*n_ratio) if isinstance(n_ratio, tuple) else n_ratio
        
        target_voltage = calculate_expected_vout(topology, v_in, duty, n_ratio)
        target_voltage = np.clip(target_voltage, 1.0, 100.0)  # Reasonable limits
        
        target_waveform = create_target_for_topology(topology, target_voltage)
        
        print(f"\n  Test {test_idx + 1}/{n_tests}: Target Vout = {target_voltage:.2f}V")
        
        # Run RL optimization
        rl_params = run_rl_optimization(agent, surrogate, topology, topology_id, 
                                        target_waveform, n_steps=30)
        
        # Get surrogate prediction
        with torch.no_grad():
            params_tensor = torch.tensor(rl_params, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            topology_tensor = torch.tensor([topology_id]).to(DEVICE)
            surrogate_wave, _ = surrogate(params_tensor, topology_tensor)
            surrogate_wave = surrogate_wave.cpu().numpy().squeeze()
        
        surrogate_vout = np.mean(surrogate_wave[-100:])  # Steady-state
        surrogate_mse = np.mean((surrogate_wave - target_waveform) ** 2)
        
        # Run SPICE simulation
        spice_params = {
            'L': rl_params[0],
            'C': rl_params[1],
            'R_load': rl_params[2],
            'V_in': rl_params[3],
            'f_sw': rl_params[4],
            'duty': rl_params[5],
        }
        
        # Add extra params for complex topologies
        if topology in [Topology.SEPIC, Topology.CUK]:
            spice_params['C_couple'] = 4.7e-6
        if topology == Topology.FLYBACK:
            spice_params['n_ratio'] = n_ratio
        
        spice_wave, spice_time = run_spice_simulation(topology, spice_params)
        
        if spice_wave is not None:
            spice_vout = np.mean(spice_wave[-100:])
            spice_mse = np.mean((spice_wave - target_waveform) ** 2)
            voltage_error = abs(spice_vout - target_voltage) / target_voltage * 100
            
            result = ValidationResult(
                topology=topology_name,
                target_voltage=target_voltage,
                rl_params=spice_params,
                surrogate_vout=surrogate_vout,
                spice_vout=spice_vout,
                surrogate_mse=surrogate_mse,
                spice_mse=spice_mse,
                voltage_error_percent=voltage_error,
                spice_time_ms=spice_time,
            )
            results.append(result)
            
            print(f"    Surrogate Vout: {surrogate_vout:.2f}V (MSE: {surrogate_mse:.2f})")
            print(f"    SPICE Vout:     {spice_vout:.2f}V (MSE: {spice_mse:.2f})")
            print(f"    Voltage Error:  {voltage_error:.1f}%")
            print(f"    SPICE Time:     {spice_time:.0f}ms")
        else:
            print(f"    SPICE simulation failed")
    
    return results


def plot_validation_results(all_results: Dict[str, List[ValidationResult]]):
    """Create visualization of validation results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    topologies = list(all_results.keys())
    
    for idx, topo in enumerate(topologies):
        ax = axes[idx]
        results = all_results[topo]
        
        if not results:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(f'{topo.upper()}')
            continue
        
        targets = [r.target_voltage for r in results]
        spice_vouts = [r.spice_vout for r in results]
        surrogate_vouts = [r.surrogate_vout for r in results]
        
        # Scatter plot
        ax.scatter(targets, spice_vouts, c='blue', label='SPICE', s=100, marker='o')
        ax.scatter(targets, surrogate_vouts, c='red', label='Surrogate', s=100, marker='x')
        
        # Perfect line
        min_v, max_v = min(targets), max(targets)
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5, label='Perfect')
        
        ax.set_xlabel('Target Voltage (V)')
        ax.set_ylabel('Output Voltage (V)')
        ax.set_title(f'{topo.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra axes if < 6 topologies
    for idx in range(len(topologies), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle('RL Agent Validation: Surrogate vs SPICE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=150)
    print(f"\nPlot saved to validation_results.png")
    plt.close()


def print_summary(all_results: Dict[str, List[ValidationResult]]):
    """Print validation summary."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"{'Topology':<12} {'Tests':>6} {'Avg Err%':>10} {'Avg SPICE MSE':>15} {'Avg Time':>10}")
    print("-"*70)
    
    total_tests = 0
    total_err = 0
    
    for topo, results in all_results.items():
        if results:
            avg_err = np.mean([r.voltage_error_percent for r in results])
            avg_mse = np.mean([r.spice_mse for r in results])
            avg_time = np.mean([r.spice_time_ms for r in results])
            print(f"{topo.upper():<12} {len(results):>6} {avg_err:>9.1f}% {avg_mse:>15.2f} {avg_time:>9.0f}ms")
            total_tests += len(results)
            total_err += avg_err * len(results)
        else:
            print(f"{topo.upper():<12} {'N/A':>6} {'N/A':>10} {'N/A':>15} {'N/A':>10}")
    
    if total_tests > 0:
        print("-"*70)
        print(f"{'OVERALL':<12} {total_tests:>6} {total_err/total_tests:>9.1f}%")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Validate RL agents against SPICE')
    parser.add_argument('--topology', type=str, default='all',
                        help='Topology to validate (buck, boost, all, etc.)')
    parser.add_argument('--n-tests', type=int, default=5,
                        help='Number of tests per topology')
    args = parser.parse_args()
    
    # Check ngspice
    if not check_ngspice():
        print("ERROR: ngspice not found. Please install:")
        print("  macOS: brew install ngspice")
        print("  Ubuntu: sudo apt install ngspice")
        return
    
    print("="*60)
    print("SPICE Validation for Multi-Topology RL Agents")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load surrogate model
    print("\nLoading surrogate model...")
    try:
        surrogate = load_trained_model(device=DEVICE)
        surrogate.eval()
        print("  Surrogate loaded successfully")
    except Exception as e:
        print(f"  Error loading surrogate: {e}")
        return
    
    # Determine topologies to test
    if args.topology.lower() == 'all':
        topologies = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback']
    else:
        topologies = [args.topology.lower()]
    
    # Run validation
    all_results = {}
    for topo in topologies:
        results = validate_topology(topo, surrogate, n_tests=args.n_tests)
        all_results[topo] = results
    
    # Summary
    print_summary(all_results)
    
    # Plot
    if any(all_results.values()):
        plot_validation_results(all_results)


if __name__ == '__main__':
    main()
