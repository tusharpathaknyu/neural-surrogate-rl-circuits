"""
SPICE-based reward for RL training.

Uses ngspice for ground-truth circuit simulation to provide
accurate rewards during training. Uses caching to avoid
redundant simulations.
"""

import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
import os


class SPICERewardCalculator:
    """
    Calculate rewards using actual SPICE simulations.
    
    Features:
    - Caches simulation results to avoid redundant runs
    - Batched evaluation for efficiency
    - Falls back to surrogate if SPICE fails
    """
    
    # SPICE netlist templates
    TEMPLATES = {
        'buck': """* Buck Converter
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
        'boost': """* Boost Converter
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
        'buck_boost': """* Buck-Boost Converter
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
wrdata {output_file} -v(output)
.endc
.end
""",
        'sepic': """* SEPIC Converter
.param L_val={L}
.param C_val={C}
.param Cc_val=4.7e-6
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

L1 input sw_node {{L_val}} ic=0

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 sw_node 0 ctrl 0 sw_model

Cc sw_node l2_in {{Cc_val}} ic=0
L2 l2_in diode_in {{L_val}} ic=0

D1 diode_in output dmodel
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
        'cuk': """* Cuk Converter
.param L_val={L}
.param C_val={C}
.param Cc_val=4.7e-6
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/freq}} {{1/freq}})

L1 input sw_node {{L_val}} ic=0

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 sw_node 0 ctrl 0 sw_model

Cc sw_node diode_in {{Cc_val}} ic={{V_val}}

D1 0 diode_in dmodel
.model dmodel d is=1e-14 n=1.05

L2 diode_in output {{L_val}} ic=0
C1 output 0 {{C_val}} ic=0
Rload output 0 {{R_val}}

.tran 1u 10m 5m uic
.control
run
set filetype=ascii
wrdata {output_file} -v(output)
.endc
.end
""",
        'flyback': """* Flyback Converter
.param Lp_val={L}
.param n=1.0
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}
Vctrl ctrl 0 PULSE(1 0 0 1n 1n {{duty/freq}} {{1/freq}})

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 input sw_node ctrl 0 sw_model

Lpri sw_node 0 {{Lp_val}} ic=0
Lsec 0 sec_node {{Lp_val*n*n}} ic=0
K1 Lpri Lsec 0.95

D1 sec_node output dmodel
.model dmodel d is=1e-14 n=1.05 rs=0.01

C1 output 0 {{C_val}} ic={{V_val*n*duty/(1-duty+0.01)}}
Rload output 0 {{R_val}}

.tran 1u 15m 10m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
""",
        'qr_flyback': """* Quasi-Resonant (QR) Flyback Converter
* Soft-switching with ZVS/ZCS for reduced EMI and losses
.param Lp_val={L}
.param Lr_val={Lr}
.param Cr_val={Cr}
.param n=1.0
.param C_val={C}
.param R_val={R_load}
.param V_val={V_in}
.param freq={f_sw}
.param duty={duty}

Vin input 0 DC {{V_val}}

* Variable frequency control (valley switching)
* In real QR, frequency varies - here we approximate
Vctrl ctrl 0 PULSE(1 0 0 1n 1n {{duty/freq}} {{1/freq}})

.model sw_model sw vt=0.5 vh=0.1 ron=0.01 roff=1e6
S1 input sw_node ctrl 0 sw_model

* Resonant inductor (leakage or discrete)
Lr sw_node pri_node {{Lr_val}}

* Magnetizing inductance
Lpri pri_node 0 {{Lp_val}} ic=0
Lsec 0 sec_node {{Lp_val*n*n}} ic=0
K1 Lpri Lsec 0.95

* Resonant capacitor across switch for ZVS
Cr sw_node 0 {{Cr_val}}

D1 sec_node output dmodel
.model dmodel d is=1e-14 n=1.05 rs=0.01

C1 output 0 {{C_val}} ic={{V_val*n*duty/(1-duty+0.01)}}
Rload output 0 {{R_val}}

.tran 0.5u 15m 10m uic
.control
run
set filetype=ascii
wrdata {output_file} v(output)
.endc
.end
""",
    }
    
    def __init__(self, topology: str = 'buck', cache_size: int = 1000, 
                 timeout: float = 5.0, waveform_points: int = 512):
        self.topology = topology.lower()
        self.cache = {}
        self.cache_size = cache_size
        self.timeout = timeout
        self.waveform_points = waveform_points
        self.hit_count = 0
        self.miss_count = 0
        
        # Check ngspice availability
        self._check_ngspice()
    
    def _check_ngspice(self):
        """Check if ngspice is available."""
        try:
            result = subprocess.run(['ngspice', '--version'], 
                                   capture_output=True, timeout=2)
            self.ngspice_available = result.returncode == 0
        except:
            self.ngspice_available = False
            print("Warning: ngspice not found. SPICE rewards disabled.")
    
    def _params_to_hash(self, params: np.ndarray) -> str:
        """Create hash from parameters for caching."""
        # Round to reduce cache misses from floating point noise
        rounded = np.round(params, 6)
        return hashlib.md5(rounded.tobytes()).hexdigest()[:16]
    
    def simulate(self, params: np.ndarray) -> Optional[np.ndarray]:
        """
        Run SPICE simulation for given parameters.
        
        Args:
            params: [L, C, R_load, V_in, f_sw, duty]
            
        Returns:
            Waveform array or None if simulation fails
        """
        if not self.ngspice_available:
            return None
        
        # Check cache
        cache_key = self._params_to_hash(params)
        if cache_key in self.cache:
            self.hit_count += 1
            return self.cache[cache_key]
        
        self.miss_count += 1
        
        # Get template
        if self.topology not in self.TEMPLATES:
            return None
        
        template = self.TEMPLATES[self.topology]
        
        # Create parameter dict
        param_dict = {
            'L': params[0],
            'C': params[1],
            'R_load': params[2],
            'V_in': params[3],
            'f_sw': params[4],
            'duty': params[5],
        }
        
        # Pre-compute derived parameters for QR flyback
        if self.topology == 'qr_flyback':
            param_dict['Lr'] = params[0] * 0.05   # Resonant inductor = 5% of Lp
            param_dict['Cr'] = params[1] * 0.01   # Resonant capacitor = 1% of C
        
        # Create temp files
        output_file = tempfile.mktemp(suffix='.txt')
        netlist = template.format(output_file=output_file, **param_dict)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
            f.write(netlist)
            netlist_path = f.name
        
        try:
            # Run ngspice
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                timeout=self.timeout
            )
            
            # Parse output
            if Path(output_file).exists():
                data = np.loadtxt(output_file)
                if len(data.shape) == 2 and data.shape[1] >= 2:
                    waveform = data[:, 1]
                    
                    # Handle inverted outputs
                    if self.topology in ['buck_boost', 'cuk']:
                        waveform = np.abs(waveform)
                    
                    # Resample to fixed length
                    if len(waveform) != self.waveform_points:
                        indices = np.linspace(0, len(waveform)-1, 
                                            self.waveform_points).astype(int)
                        waveform = waveform[indices]
                    
                    # Validate
                    if not np.isnan(waveform).any() and np.abs(waveform).max() < 1000:
                        # Cache result
                        if len(self.cache) >= self.cache_size:
                            # Remove oldest entry
                            self.cache.pop(next(iter(self.cache)))
                        self.cache[cache_key] = waveform
                        return waveform
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            pass
        finally:
            # Cleanup
            for f in [netlist_path, output_file]:
                try:
                    Path(f).unlink()
                except:
                    pass
        
        return None
    
    def compute_reward(self, params: np.ndarray, target_waveform: np.ndarray,
                       surrogate_waveform: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        """
        Compute reward using SPICE simulation.
        
        Args:
            params: Circuit parameters
            target_waveform: Target output waveform
            surrogate_waveform: Fallback if SPICE fails
            
        Returns:
            reward: Computed reward
            info: Debug information
        """
        info = {}
        
        # Try SPICE simulation
        spice_waveform = self.simulate(params)
        
        if spice_waveform is not None:
            # Use SPICE ground truth
            mse = np.mean((spice_waveform - target_waveform) ** 2)
            vout = np.mean(spice_waveform[-100:])
            info['source'] = 'spice'
            info['spice_vout'] = vout
        elif surrogate_waveform is not None:
            # Fall back to surrogate
            mse = np.mean((surrogate_waveform - target_waveform) ** 2)
            vout = np.mean(surrogate_waveform[-100:])
            info['source'] = 'surrogate'
        else:
            # No simulation available
            return -10.0, {'source': 'none', 'error': 'No simulation'}
        
        info['mse'] = mse
        info['vout'] = vout
        
        # Target voltage
        target_vout = np.mean(target_waveform[-100:])
        voltage_error = abs(vout - target_vout) / (abs(target_vout) + 1e-6)
        info['voltage_error'] = voltage_error
        info['target_vout'] = target_vout
        
        # Compute reward (higher is better)
        # Base reward from MSE
        mse_reward = -np.log10(mse + 1e-6)  # Higher for lower MSE
        
        # Voltage accuracy bonus
        if voltage_error < 0.05:
            voltage_bonus = 2.0
        elif voltage_error < 0.10:
            voltage_bonus = 1.0
        elif voltage_error < 0.20:
            voltage_bonus = 0.5
        else:
            voltage_bonus = 0.0
        
        reward = mse_reward + voltage_bonus
        
        # Success check
        info['success'] = voltage_error < 0.05 and mse < 10.0
        
        return reward, info
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
        }


if __name__ == '__main__':
    print("Testing SPICE Reward Calculator...")
    
    calc = SPICERewardCalculator(topology='buck')
    print(f"ngspice available: {calc.ngspice_available}")
    
    # Test with sample parameters
    params = np.array([50e-6, 100e-6, 10.0, 12.0, 100e3, 0.5])
    waveform = calc.simulate(params)
    
    if waveform is not None:
        print(f"✓ Simulation successful")
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Vout: {np.mean(waveform[-100:]):.2f}V")
        print(f"  Expected: {12.0 * 0.5:.2f}V")
    else:
        print("✗ Simulation failed")
    
    print(f"Cache stats: {calc.get_stats()}")
