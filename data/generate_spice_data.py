"""
Generate training dataset for buck converter surrogate model.
Runs ngspice simulations with randomized component values.
"""

import numpy as np
import subprocess
import os
import tempfile
from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Simulation parameters
NUM_SAMPLES = 5000  # Full training dataset
WAVEFORM_POINTS = 512  # Number of points in output waveform

# Component ranges (realistic values)
PARAM_RANGES = {
    'L': (10e-6, 100e-6),      # 10µH to 100µH
    'C': (47e-6, 470e-6),      # 47µF to 470µF  
    'R_load': (2, 50),          # 2Ω to 50Ω
    'V_in': (10, 24),           # 10V to 24V
    'f_sw': (50e3, 500e3),      # 50kHz to 500kHz
    'duty': (0.2, 0.8),         # 20% to 80% duty cycle
}

# SPICE netlist template
NETLIST_TEMPLATE = """* Buck Converter - Auto Generated
.param L_val = {L}
.param C_val = {C}
.param R_load = {R_load}
.param V_in = {V_in}
.param f_sw = {f_sw}
.param duty = {duty}

Vin input 0 DC {{V_in}}

* PWM switch model
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {{duty/f_sw}} {{1/f_sw}})
S1 input sw_out ctrl 0 SWITCH
.model SWITCH SW(Ron=0.01 Roff=1Meg Vt=2.5 Vh=0.1)

* Freewheeling diode
D1 0 sw_out DIODE
.model DIODE D(Is=1e-14 Rs=0.01 N=1.05)

* LC filter with parasitics
L1 sw_out lx {{L_val}} IC=0
R_L lx output 0.02
C1 output 0 {{C_val}} IC={{V_in*duty}}
R_C output cap_node 0.01

* Load
Rload output 0 {{R_load}}

* Transient analysis - simulate for 10ms, save last 1ms
.tran 1u 10m 9m 1u UIC

.control
run
set filetype=ascii
wrdata {output_file} V(output) I(L1)
.endc

.end
"""


def generate_random_params():
    """Generate random component values within specified ranges."""
    params = {}
    for name, (min_val, max_val) in PARAM_RANGES.items():
        if name in ['L', 'C', 'f_sw']:
            # Log-uniform for values spanning orders of magnitude
            params[name] = np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
        else:
            params[name] = np.random.uniform(min_val, max_val)
    return params


def run_simulation(params, sim_id):
    """Run ngspice simulation with given parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        netlist_path = os.path.join(tmpdir, 'circuit.cir')
        output_path = os.path.join(tmpdir, 'output.txt')
        
        # Generate netlist
        netlist = NETLIST_TEMPLATE.format(
            L=params['L'],
            C=params['C'],
            R_load=params['R_load'],
            V_in=params['V_in'],
            f_sw=params['f_sw'],
            duty=params['duty'],
            output_file=output_path
        )
        
        with open(netlist_path, 'w') as f:
            f.write(netlist)
        
        # Run ngspice
        try:
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return None
            
            # Parse output
            waveform = parse_ngspice_output(output_path)
            if waveform is None:
                return None
                
            return {
                'params': params,
                'waveform': waveform,
                'sim_id': sim_id
            }
            
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            print(f"Simulation {sim_id} failed: {e}")
            return None


def parse_ngspice_output(output_path):
    """Parse ngspice output file to extract waveform."""
    try:
        data = np.loadtxt(output_path, skiprows=1)
        if len(data) < 10:
            return None
        
        # Extract time, voltage, current
        time = data[:, 0]
        voltage = data[:, 1]
        current = data[:, 2] if data.shape[1] > 2 else np.zeros_like(voltage)
        
        # Resample to fixed number of points
        t_new = np.linspace(time[0], time[-1], WAVEFORM_POINTS)
        v_resampled = np.interp(t_new, time, voltage)
        i_resampled = np.interp(t_new, time, current)
        
        return {
            'time': t_new.tolist(),
            'voltage': v_resampled.tolist(),
            'current': i_resampled.tolist()
        }
        
    except Exception as e:
        return None


def compute_waveform_metrics(waveform, params):
    """Compute engineering metrics from waveform."""
    voltage = np.array(waveform['voltage'])
    current = np.array(waveform['current'])
    
    V_target = params['V_in'] * params['duty']
    
    metrics = {
        'V_avg': np.mean(voltage),
        'V_ripple': np.max(voltage) - np.min(voltage),
        'V_ripple_percent': (np.max(voltage) - np.min(voltage)) / np.mean(voltage) * 100,
        'I_avg': np.mean(np.abs(current)),
        'overshoot': (np.max(voltage) - V_target) / V_target * 100 if V_target > 0 else 0,
        'P_out': np.mean(voltage) ** 2 / params['R_load'],
    }
    
    return metrics


def generate_dataset(num_samples=NUM_SAMPLES, num_workers=4):
    """Generate full dataset with parallel simulation."""
    dataset = []
    
    print(f"Generating {num_samples} samples...")
    
    # Generate all parameter sets first
    param_sets = [generate_random_params() for _ in range(num_samples)]
    
    # Run simulations (can be parallelized)
    successful = 0
    failed = 0
    
    for i, params in enumerate(tqdm(param_sets)):
        result = run_simulation(params, i)
        
        if result is not None:
            # Add metrics
            result['metrics'] = compute_waveform_metrics(
                result['waveform'], 
                result['params']
            )
            dataset.append(result)
            successful += 1
        else:
            failed += 1
    
    print(f"Completed: {successful} successful, {failed} failed")
    
    return dataset


def save_dataset(dataset, output_dir='./'):
    """Save dataset to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON for easy loading
    with open(output_path / 'buck_dataset.json', 'w') as f:
        json.dump(dataset, f)
    
    # Also save as numpy arrays for faster training
    params_array = np.array([
        [d['params']['L'], d['params']['C'], d['params']['R_load'],
         d['params']['V_in'], d['params']['f_sw'], d['params']['duty']]
        for d in dataset
    ])
    
    waveforms_array = np.array([d['waveform']['voltage'] for d in dataset])
    
    np.save(output_path / 'params.npy', params_array)
    np.save(output_path / 'waveforms.npy', waveforms_array)
    
    print(f"Saved {len(dataset)} samples to {output_path}")
    print(f"  - buck_dataset.json (full data)")
    print(f"  - params.npy shape: {params_array.shape}")
    print(f"  - waveforms.npy shape: {waveforms_array.shape}")


if __name__ == '__main__':
    # Check if ngspice is available
    try:
        result = subprocess.run(['ngspice', '--version'], capture_output=True)
        print("ngspice found!")
    except FileNotFoundError:
        print("ERROR: ngspice not found. Please install it:")
        print("  macOS: brew install ngspice")
        print("  Ubuntu: sudo apt install ngspice")
        exit(1)
    
    # Generate dataset
    dataset = generate_dataset(num_samples=NUM_SAMPLES)
    
    # Save
    save_dataset(dataset, output_dir='./data')
