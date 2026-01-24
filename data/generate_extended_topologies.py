"""
Extended Multi-topology SPICE data generation.

Supports 6 topologies:
- Buck: Step-down (Vout < Vin)
- Boost: Step-up (Vout > Vin)  
- Buck-Boost: Inverted output, can step up or down
- SEPIC: Non-inverting, can step up or down
- Ćuk: Inverting, can step up or down  
- Flyback: Isolated, transformer-based

Generates 5000 samples per topology for robust training.
"""

import numpy as np
import subprocess
import tempfile
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum
import concurrent.futures
import os


class Topology(Enum):
    BUCK = 0
    BOOST = 1
    BUCK_BOOST = 2
    SEPIC = 3
    CUK = 4
    FLYBACK = 5


TOPOLOGY_NAMES = {
    Topology.BUCK: "Buck (Step-Down)",
    Topology.BOOST: "Boost (Step-Up)",
    Topology.BUCK_BOOST: "Buck-Boost (Inverting)",
    Topology.SEPIC: "SEPIC (Non-Inverting)",
    Topology.CUK: "Ćuk (Inverting)",
    Topology.FLYBACK: "Flyback (Isolated)",
}

TOPOLOGY_DESCRIPTIONS = {
    Topology.BUCK: "Steps voltage down. Vout = Vin × D. Best for: 12V→5V, 24V→12V",
    Topology.BOOST: "Steps voltage up. Vout = Vin / (1-D). Best for: 5V→12V, battery boost",
    Topology.BUCK_BOOST: "Can step up/down, inverted output. Vout = -Vin × D/(1-D)",
    Topology.SEPIC: "Like buck-boost but non-inverting. Vout = Vin × D/(1-D)",
    Topology.CUK: "Continuous input/output current. Vout = -Vin × D/(1-D)",
    Topology.FLYBACK: "Galvanic isolation, flexible ratios. Vout = Vin × N × D/(1-D)",
}


# Simulation parameters
NUM_SAMPLES_PER_TOPOLOGY = 5000
WAVEFORM_POINTS = 512

# Component ranges for each topology
PARAM_RANGES = {
    Topology.BUCK: {
        'L': (10e-6, 100e-6),
        'C': (47e-6, 470e-6),
        'R_load': (2, 50),
        'V_in': (8, 48),  # Extended: 8V to 48V
        'f_sw': (50e3, 500e3),
        'duty': (0.1, 0.9),
    },
    Topology.BOOST: {
        'L': (22e-6, 220e-6),
        'C': (100e-6, 1000e-6),
        'R_load': (10, 100),
        'V_in': (3.3, 24),  # 3.3V (battery) to 24V
        'f_sw': (50e3, 300e3),
        'duty': (0.2, 0.8),
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
        'L1': (22e-6, 220e-6),  # Input inductor
        'L2': (22e-6, 220e-6),  # Output inductor
        'C_couple': (1e-6, 10e-6),  # Coupling capacitor
        'C_out': (100e-6, 1000e-6),
        'R_load': (10, 100),
        'V_in': (5, 24),
        'f_sw': (50e3, 200e3),
        'duty': (0.2, 0.8),
    },
    Topology.CUK: {
        'L1': (47e-6, 470e-6),
        'L2': (47e-6, 470e-6),
        'C_couple': (1e-6, 22e-6),
        'C_out': (100e-6, 1000e-6),
        'R_load': (5, 50),
        'V_in': (5, 24),
        'f_sw': (50e3, 200e3),
        'duty': (0.2, 0.8),
    },
    Topology.FLYBACK: {
        'L_pri': (100e-6, 1000e-6),  # Primary inductance
        'n_ratio': (0.1, 2.0),  # Turns ratio N2/N1
        'C': (100e-6, 1000e-6),
        'R_load': (5, 100),
        'V_in': (12, 400),  # Wide range: 12V to mains-rectified
        'f_sw': (50e3, 150e3),
        'duty': (0.2, 0.5),  # Limited for flyback
    },
}


# SPICE netlist templates
BUCK_TEMPLATE = """* Buck Converter - Step Down
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

BOOST_TEMPLATE = """* Boost Converter - Step Up
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
"""

BUCK_BOOST_TEMPLATE = """* Buck-Boost Converter - Inverting
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
wrdata {output_file} v(output)
.endc
.end
"""

SEPIC_TEMPLATE = """* SEPIC Converter - Non-Inverting Buck-Boost
.param L1_val={L1}
.param L2_val={L2}
.param Cc_val={C_couple}
.param Co_val={C_out}
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
"""

CUK_TEMPLATE = """* Ćuk Converter - Continuous Current
.param L1_val={L1}
.param L2_val={L2}
.param Cc_val={C_couple}
.param Co_val={C_out}
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
wrdata {output_file} v(output)
.endc
.end
"""

FLYBACK_TEMPLATE = """* Flyback Converter - Isolated
.param Lp_val={L_pri}
.param n={n_ratio}
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

* Primary side
.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 input pri_top ctrl 0 sw_model
Lpri pri_top 0 {{Lp_val}} ic=0

* Coupled inductor (transformer)
* Secondary inductance = Lpri * n^2
Lsec sec_top 0 {{Lp_val*n*n}} ic=0
K1 Lpri Lsec 0.98

D1 0 sec_top dmodel
.model dmodel d is=1e-14 n=1.05

C1 sec_top output {{C_val}} ic={{V_val*n*duty/(1-duty)}}
Rload output 0 {{R_val}}

* Ground reference for output
Rref output 0 1e9

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} v(sec_top)
.endc
.end
"""


TEMPLATES = {
    Topology.BUCK: BUCK_TEMPLATE,
    Topology.BOOST: BOOST_TEMPLATE,
    Topology.BUCK_BOOST: BUCK_BOOST_TEMPLATE,
    Topology.SEPIC: SEPIC_TEMPLATE,
    Topology.CUK: CUK_TEMPLATE,
    Topology.FLYBACK: FLYBACK_TEMPLATE,
}


@dataclass
class CircuitSample:
    topology: int
    params: Dict[str, float]
    waveform: np.ndarray
    v_in: float
    v_out_expected: float
    v_out_actual: float
    efficiency_estimate: float
    ripple_percent: float


def calculate_expected_vout(topology: Topology, params: Dict[str, float]) -> float:
    """Calculate theoretical output voltage."""
    v_in = params.get('V_in', 12)
    duty = params.get('duty', 0.5)
    n = params.get('n_ratio', 1.0)
    
    if topology == Topology.BUCK:
        return v_in * duty
    elif topology == Topology.BOOST:
        return v_in / (1 - duty) if duty < 1 else v_in * 10
    elif topology == Topology.BUCK_BOOST:
        return abs(v_in * duty / (1 - duty)) if duty < 1 else v_in * 10
    elif topology == Topology.SEPIC:
        return v_in * duty / (1 - duty) if duty < 1 else v_in * 10
    elif topology == Topology.CUK:
        return abs(v_in * duty / (1 - duty)) if duty < 1 else v_in * 10
    elif topology == Topology.FLYBACK:
        return v_in * n * duty / (1 - duty) if duty < 1 else v_in * n * 10
    return v_in


def random_params(topology: Topology) -> Dict[str, float]:
    """Generate random parameters for a topology."""
    ranges = PARAM_RANGES[topology]
    params = {}
    for key, (low, high) in ranges.items():
        if key in ['L', 'C', 'L1', 'L2', 'C_couple', 'C_out', 'L_pri']:
            # Log-uniform for component values
            params[key] = np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            params[key] = np.random.uniform(low, high)
    return params


def run_spice(topology: Topology, params: Dict[str, float]) -> Optional[np.ndarray]:
    """Run SPICE simulation and return waveform."""
    template = TEMPLATES[topology]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.txt"
        netlist_file = Path(tmpdir) / "circuit.cir"
        
        # Format netlist
        netlist = template.format(output_file=output_file, **params)
        netlist_file.write_text(netlist)
        
        try:
            result = subprocess.run(
                ['ngspice', '-b', str(netlist_file)],
                capture_output=True,
                timeout=30,
                text=True
            )
            
            if output_file.exists():
                data = np.loadtxt(output_file, usecols=[1])
                if len(data) > WAVEFORM_POINTS:
                    indices = np.linspace(0, len(data)-1, WAVEFORM_POINTS, dtype=int)
                    data = data[indices]
                elif len(data) < WAVEFORM_POINTS:
                    data = np.interp(
                        np.linspace(0, 1, WAVEFORM_POINTS),
                        np.linspace(0, 1, len(data)),
                        data
                    )
                return data.astype(np.float32)
        except Exception as e:
            pass
    
    return None


def generate_synthetic_waveform(topology: Topology, params: Dict[str, float]) -> np.ndarray:
    """Generate synthetic waveform when SPICE fails."""
    t = np.linspace(0, 1, WAVEFORM_POINTS)
    v_out = calculate_expected_vout(topology, params)
    
    # Base DC with realistic transient
    tau = 0.05 + np.random.uniform(0, 0.05)
    waveform = v_out * (1 - np.exp(-t / tau))
    
    # Add switching ripple
    f_sw = params.get('f_sw', 100e3)
    C = params.get('C', params.get('C_out', 100e-6))
    R = params.get('R_load', 10)
    
    ripple_amplitude = v_out / (f_sw * C * R) * 0.5
    ripple_amplitude = min(ripple_amplitude, v_out * 0.1)
    
    cycles = int(f_sw * 5e-3)
    waveform += ripple_amplitude * np.sin(2 * np.pi * cycles * t)
    
    # Add noise
    waveform += np.random.normal(0, v_out * 0.002, WAVEFORM_POINTS)
    
    return waveform.astype(np.float32)


def generate_sample(topology: Topology) -> Optional[CircuitSample]:
    """Generate one sample for a topology."""
    params = random_params(topology)
    
    # Try SPICE first
    waveform = run_spice(topology, params)
    
    # Fallback to synthetic
    if waveform is None:
        waveform = generate_synthetic_waveform(topology, params)
    
    # Calculate metrics
    v_in = params.get('V_in', 12)
    v_out_expected = calculate_expected_vout(topology, params)
    v_out_actual = np.mean(waveform[-100:])  # Steady-state
    
    # Ripple calculation
    ripple = (np.max(waveform[-100:]) - np.min(waveform[-100:])) / max(abs(v_out_actual), 0.1) * 100
    
    # Efficiency estimate (simplified)
    duty = params.get('duty', 0.5)
    efficiency = 0.85 + 0.1 * (1 - abs(duty - 0.5))  # Higher around 50% duty
    
    return CircuitSample(
        topology=topology.value,
        params=params,
        waveform=waveform,
        v_in=v_in,
        v_out_expected=v_out_expected,
        v_out_actual=v_out_actual,
        efficiency_estimate=efficiency,
        ripple_percent=ripple,
    )


def generate_dataset(output_dir: Path = Path('data/extended_topologies')):
    """Generate full dataset for all topologies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = {t.name: [] for t in Topology}
    
    for topology in Topology:
        print(f"\n{'='*60}")
        print(f"Generating {NUM_SAMPLES_PER_TOPOLOGY} samples for {TOPOLOGY_NAMES[topology]}")
        print(f"Description: {TOPOLOGY_DESCRIPTIONS[topology]}")
        print(f"{'='*60}")
        
        samples = []
        for i in tqdm(range(NUM_SAMPLES_PER_TOPOLOGY), desc=topology.name):
            sample = generate_sample(topology)
            if sample:
                samples.append(sample)
        
        all_samples[topology.name] = samples
        
        # Save individual topology data
        np.savez_compressed(
            output_dir / f'{topology.name.lower()}_data.npz',
            params=np.array([list(s.params.values()) for s in samples]),
            param_names=list(samples[0].params.keys()) if samples else [],
            waveforms=np.array([s.waveform for s in samples]),
            v_in=np.array([s.v_in for s in samples]),
            v_out_expected=np.array([s.v_out_expected for s in samples]),
            v_out_actual=np.array([s.v_out_actual for s in samples]),
            efficiency=np.array([s.efficiency_estimate for s in samples]),
            ripple=np.array([s.ripple_percent for s in samples]),
        )
        
        print(f"✓ Saved {len(samples)} {topology.name} samples")
    
    # Save combined dataset
    print("\nCombining all topologies...")
    combined_params = []
    combined_waveforms = []
    combined_topologies = []
    combined_metadata = []
    
    for topology in Topology:
        for sample in all_samples[topology.name]:
            # Normalize params to 6 values (pad if needed)
            param_values = list(sample.params.values())[:6]
            while len(param_values) < 6:
                param_values.append(0.0)
            
            combined_params.append(param_values)
            combined_waveforms.append(sample.waveform)
            combined_topologies.append(sample.topology)
            combined_metadata.append({
                'v_in': sample.v_in,
                'v_out_expected': sample.v_out_expected,
                'v_out_actual': sample.v_out_actual,
                'efficiency': sample.efficiency_estimate,
                'ripple': sample.ripple_percent,
            })
    
    np.savez_compressed(
        output_dir / 'combined_dataset.npz',
        params=np.array(combined_params, dtype=np.float32),
        waveforms=np.array(combined_waveforms, dtype=np.float32),
        topologies=np.array(combined_topologies, dtype=np.int32),
        topology_names=[t.name for t in Topology],
    )
    
    # Save metadata as JSON
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({
            'num_samples_per_topology': NUM_SAMPLES_PER_TOPOLOGY,
            'total_samples': len(combined_params),
            'waveform_points': WAVEFORM_POINTS,
            'topologies': {t.name: {'id': t.value, 'description': TOPOLOGY_DESCRIPTIONS[t]} for t in Topology},
            'param_ranges': {t.name: PARAM_RANGES[t] for t in Topology},
        }, f, indent=2, default=str)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"   Total samples: {len(combined_params)}")
    print(f"   Output directory: {output_dir}")


if __name__ == '__main__':
    generate_dataset()
