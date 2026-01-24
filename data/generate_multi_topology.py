"""
Multi-topology SPICE data generation.
Supports: Buck, Boost, Buck-Boost converters.

Each topology has different characteristics:
- Buck: Step-down (Vout < Vin)
- Boost: Step-up (Vout > Vin)  
- Buck-Boost: Inverted output, can step up or down
"""

import numpy as np
import subprocess
import tempfile
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Topology(Enum):
    BUCK = "buck"
    BOOST = "boost"
    BUCK_BOOST = "buck_boost"


# Simulation parameters
NUM_SAMPLES_PER_TOPOLOGY = 2000
WAVEFORM_POINTS = 512

# Component ranges for each topology
PARAM_RANGES = {
    Topology.BUCK: {
        'L': (10e-6, 100e-6),
        'C': (47e-6, 470e-6),
        'R_load': (2, 50),
        'V_in': (10, 24),
        'f_sw': (50e3, 500e3),
        'duty': (0.2, 0.8),
    },
    Topology.BOOST: {
        'L': (22e-6, 220e-6),  # Larger inductors for boost
        'C': (100e-6, 1000e-6),  # Larger caps for boost
        'R_load': (10, 100),
        'V_in': (5, 15),  # Lower input for boost
        'f_sw': (50e3, 300e3),
        'duty': (0.3, 0.7),  # More restricted duty for stability
    },
    Topology.BUCK_BOOST: {
        'L': (47e-6, 470e-6),
        'C': (100e-6, 1000e-6),
        'R_load': (5, 50),
        'V_in': (8, 20),
        'f_sw': (50e3, 200e3),
        'duty': (0.3, 0.7),
    },
}


# SPICE netlist templates
BUCK_TEMPLATE = """* Buck Converter
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
"""

BOOST_TEMPLATE = """* Boost Converter
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

* Inductor from input
L1 input sw_node {{L_val}} ic=0

* Switch to ground
.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 sw_node 0 ctrl 0 sw_model

* Diode to output
D1 sw_node output dmodel
.model dmodel d is=1e-14 n=1.05

* Output cap and load
C1 output 0 {{C_val}} ic={{V_val}}
Rload output 0 {{R_val}}

.tran 1u 15m 10m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
"""

BUCK_BOOST_TEMPLATE = """* Buck-Boost Converter (Inverting)
.param L_val={L}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

* Switch from input to inductor
.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 input sw_node ctrl 0 sw_model

* Inductor
L1 sw_node 0 {{L_val}} ic=0

* Diode (inverted output)
D1 output sw_node dmodel
.model dmodel d is=1e-14 n=1.05

* Output (negative voltage referenced to ground)
C1 output 0 {{C_val}} ic=0
Rload output 0 {{R_val}}

.tran 1u 15m 10m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
"""

TEMPLATES = {
    Topology.BUCK: BUCK_TEMPLATE,
    Topology.BOOST: BOOST_TEMPLATE,
    Topology.BUCK_BOOST: BUCK_BOOST_TEMPLATE,
}


def random_params(topology: Topology) -> Dict[str, float]:
    """Generate random component values for a topology."""
    ranges = PARAM_RANGES[topology]
    params = {}
    for name, (low, high) in ranges.items():
        if name in ['L', 'C', 'f_sw']:
            params[name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            params[name] = np.random.uniform(low, high)
    return params


def run_simulation(topology: Topology, params: Dict[str, float], timeout: int = 10) -> Optional[np.ndarray]:
    """Run SPICE simulation for given topology and parameters."""
    output_file = tempfile.mktemp(suffix='.txt')
    template = TEMPLATES[topology]
    
    netlist = template.format(output_file=output_file, **params)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
        f.write(netlist)
        netlist_path = f.name
    
    try:
        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if Path(output_file).exists():
            data = np.loadtxt(output_file)
            if len(data.shape) == 2 and data.shape[1] >= 2:
                waveform = data[:, 1]
                # Handle inverted output for buck-boost
                if topology == Topology.BUCK_BOOST:
                    waveform = -waveform  # Invert to positive
                # Resample
                indices = np.linspace(0, len(waveform)-1, WAVEFORM_POINTS).astype(int)
                waveform = waveform[indices]
                # Validate
                if not np.isnan(waveform).any() and np.abs(waveform).max() < 100:
                    return waveform
    except Exception:
        pass
    finally:
        for f in [netlist_path, output_file]:
            try:
                Path(f).unlink()
            except:
                pass
    
    return None


def generate_dataset(output_dir: str = 'data'):
    """Generate multi-topology dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Check ngspice
    try:
        subprocess.run(['ngspice', '--version'], capture_output=True, check=True)
        print("✓ ngspice found")
    except:
        print("✗ ngspice not found!")
        return
    
    all_params = []
    all_waveforms = []
    all_topologies = []
    
    for topology in Topology:
        print(f"\nGenerating {topology.value} samples...")
        
        params_list = []
        waveforms_list = []
        
        pbar = tqdm(total=NUM_SAMPLES_PER_TOPOLOGY, desc=topology.value)
        attempts = 0
        max_attempts = NUM_SAMPLES_PER_TOPOLOGY * 3
        
        while len(waveforms_list) < NUM_SAMPLES_PER_TOPOLOGY and attempts < max_attempts:
            attempts += 1
            params = random_params(topology)
            waveform = run_simulation(topology, params)
            
            if waveform is not None:
                params_list.append([params[k] for k in ['L', 'C', 'R_load', 'V_in', 'f_sw', 'duty']])
                waveforms_list.append(waveform)
                pbar.update(1)
        
        pbar.close()
        print(f"  Generated {len(waveforms_list)} samples")
        
        all_params.extend(params_list)
        all_waveforms.extend(waveforms_list)
        all_topologies.extend([topology.value] * len(waveforms_list))
    
    # Save
    params_array = np.array(all_params, dtype=np.float32)
    waveforms_array = np.array(all_waveforms, dtype=np.float32)
    topologies_array = np.array(all_topologies)
    
    np.save(output_path / 'multi_params.npy', params_array)
    np.save(output_path / 'multi_waveforms.npy', waveforms_array)
    np.save(output_path / 'multi_topologies.npy', topologies_array)
    
    print(f"\n✓ Saved multi-topology dataset:")
    print(f"  Params: {params_array.shape}")
    print(f"  Waveforms: {waveforms_array.shape}")
    print(f"  Topologies: {len(set(all_topologies))} types")
    
    # Summary stats
    for topology in Topology:
        mask = topologies_array == topology.value
        waves = waveforms_array[mask]
        print(f"\n{topology.value}:")
        print(f"  Count: {mask.sum()}")
        print(f"  Vout range: {waves.mean(axis=1).min():.2f}V - {waves.mean(axis=1).max():.2f}V")


if __name__ == '__main__':
    generate_dataset()
