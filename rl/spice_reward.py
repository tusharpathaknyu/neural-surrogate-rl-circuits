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
                 timeout: float = 5.0):
        self.topology = topology.lower()
        self.cache = {}  # caches full-res waveforms
        self.cache_size = cache_size
        self.timeout = timeout
        self.hit_count = 0
        self.miss_count = 0
        self.sim_count = 0  # total successful simulations
        self.fail_count = 0  # total failed simulations
        
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
        
        Returns the FULL-RESOLUTION waveform from ngspice (typically ~5000 points).
        This preserves ripple, switching transients, and ringing — the real
        waveform characteristics that are the whole point of SPICE validation.
        
        Args:
            params: [L, C, R_load, V_in, f_sw, duty]
            
        Returns:
            Full-resolution waveform array or None if simulation fails
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
                    
                    # DO NOT resample — keep full resolution!
                    # Full-res waveform preserves ripple, transients, ringing.
                    
                    # Validate
                    if not np.isnan(waveform).any() and np.abs(waveform).max() < 1000:
                        # Cache result
                        if len(self.cache) >= self.cache_size:
                            self.cache.pop(next(iter(self.cache)))
                        self.cache[cache_key] = waveform
                        self.sim_count += 1
                        return waveform
                    else:
                        self.fail_count += 1
        except subprocess.TimeoutExpired:
            self.fail_count += 1
        except Exception as e:
            self.fail_count += 1
        finally:
            # Cleanup
            for f in [netlist_path, output_file]:
                try:
                    Path(f).unlink()
                except:
                    pass
        
        return None
    
    def analyze_waveform(self, waveform: np.ndarray, params: np.ndarray) -> Dict:
        """
        Extract waveform quality metrics from a FULL-RESOLUTION SPICE waveform.
        
        This is where the real value of SPICE lives — these metrics cannot be
        computed from a 32-point surrogate output. The full-res waveform has
        thousands of points that capture switching ripple, ringing, overshoot,
        and settling behavior.
        
        Args:
            waveform: Full-resolution SPICE output (typically ~5000 points)
            params: Circuit parameters [L, C, R_load, V_in, f_sw, duty]
            
        Returns:
            Dict with waveform quality metrics
        """
        metrics = {}
        n = len(waveform)
        
        # Use last 60% of waveform for steady-state analysis
        # (first 40% may have startup transients)
        ss_start = int(n * 0.4)
        steady_state = waveform[ss_start:]
        
        # ---- DC output ----
        v_out_mean = np.mean(steady_state)
        metrics['v_out_mean'] = v_out_mean
        
        # ---- Ripple voltage (peak-to-peak in steady state) ----
        v_ripple_pp = np.max(steady_state) - np.min(steady_state)
        v_ripple_pct = v_ripple_pp / (abs(v_out_mean) + 1e-6) * 100  # as percentage
        metrics['ripple_pp'] = v_ripple_pp
        metrics['ripple_pct'] = v_ripple_pct
        
        # ---- Overshoot ----
        full_max = np.max(waveform)
        overshoot_pct = max(0, (full_max - abs(v_out_mean)) / (abs(v_out_mean) + 1e-6)) * 100
        metrics['overshoot_pct'] = overshoot_pct
        
        # ---- Settling time (time to reach ±2% of final value) ----
        band = 0.02 * abs(v_out_mean) + 0.01  # 2% band
        within_band = np.abs(waveform - v_out_mean) < band
        # Find first index where all subsequent points are within band
        settling_idx = n  # default: never settled
        for i in range(n - 1, -1, -1):
            if not within_band[i]:
                settling_idx = i + 1
                break
        settling_fraction = settling_idx / n  # 0=instant, 1=never
        metrics['settling_fraction'] = settling_fraction
        
        # ---- THD from full-resolution data ----
        fft = np.abs(np.fft.rfft(steady_state))
        if len(fft) > 10:
            # Find fundamental (largest non-DC component)
            fundamental_idx = np.argmax(fft[1:]) + 1
            fundamental = fft[fundamental_idx]
            if fundamental > 1e-6:
                # Sum harmonics (2nd through 10th)
                harmonic_energy = 0
                for h in range(2, 11):
                    idx = fundamental_idx * h
                    if idx < len(fft):
                        harmonic_energy += fft[idx] ** 2
                thd = np.sqrt(harmonic_energy) / fundamental
            else:
                thd = 0
        else:
            thd = 0
        metrics['thd'] = thd
        
        # ---- Smoothness (RMS of derivative — lower = smoother) ----
        dv = np.diff(steady_state)
        rms_dv = np.sqrt(np.mean(dv ** 2))
        smoothness = rms_dv / (abs(v_out_mean) + 1e-6)
        metrics['smoothness'] = smoothness
        
        # ---- Ringing detection (high-frequency energy ratio) ----
        n_fft = len(fft)
        low_energy = np.sum(fft[:n_fft // 4] ** 2)
        high_energy = np.sum(fft[n_fft // 4:] ** 2)
        ringing_ratio = high_energy / (low_energy + 1e-6)
        metrics['ringing_ratio'] = ringing_ratio
        
        return metrics
    
    def compute_spice_quality_bonus(self, spice_waveform: np.ndarray,
                                     params: np.ndarray,
                                     topology: str) -> Tuple[float, Dict]:
        """
        Compute a SPICE-based waveform quality bonus/penalty.
        
        This is added ON TOP of the surrogate-based reward. It captures
        waveform characteristics only visible in the full SPICE simulation:
        ripple, overshoot, ringing, settling — the real engineering metrics.
        
        Args:
            spice_waveform: Full-res SPICE output
            params: Circuit parameters
            topology: Topology name
            
        Returns:
            bonus: Float bonus/penalty to add to surrogate reward
            metrics: Dict of SPICE waveform metrics
        """
        from rl.topology_rewards import TOPOLOGY_REWARD_CONFIG
        config = TOPOLOGY_REWARD_CONFIG.get(topology, TOPOLOGY_REWARD_CONFIG['buck'])
        
        metrics = self.analyze_waveform(spice_waveform, params)
        
        bonus = 0.0
        
        # ---- Ripple quality (compare to topology target) ----
        target_ripple_pct = config.get('ripple_target', 0.03) * 100  # convert to %
        actual_ripple_pct = metrics['ripple_pct']
        if actual_ripple_pct <= target_ripple_pct:
            bonus += 1.5  # Under target = good
        elif actual_ripple_pct <= target_ripple_pct * 2:
            bonus += 0.5  # Up to 2x target = acceptable
        else:
            bonus -= min(2.0, (actual_ripple_pct / target_ripple_pct - 1) * 0.5)  # Penalize
        
        # ---- Overshoot penalty ----
        if metrics['overshoot_pct'] < 5:
            bonus += 1.0  # Low overshoot = good
        elif metrics['overshoot_pct'] < 15:
            bonus += 0.0  # Moderate = neutral
        else:
            bonus -= min(2.0, metrics['overshoot_pct'] / 15)  # High = bad
        
        # ---- Settling quality ----
        if metrics['settling_fraction'] < 0.3:
            bonus += 0.5  # Settles quickly
        elif metrics['settling_fraction'] > 0.8:
            bonus -= 1.0  # Still settling = bad
        
        # ---- THD from SPICE (topology-aware) ----
        if config['type'] == 'resonant':
            # QR topologies inherently have harmonic content — don't penalize
            pass
        else:
            if metrics['thd'] < 0.05:
                bonus += 0.5  # Low distortion
            elif metrics['thd'] > 0.3:
                bonus -= min(1.5, metrics['thd'] * 2)
        
        # ---- Ringing penalty (especially for isolated topologies) ----
        if config.get('has_ringing'):
            # Flyback: some ringing expected, only penalize excessive
            if metrics['ringing_ratio'] > 0.5:
                bonus -= min(1.0, metrics['ringing_ratio'])
        else:
            # Non-isolated: ringing is always bad
            if metrics['ringing_ratio'] > 0.2:
                bonus -= min(1.0, metrics['ringing_ratio'] * 2)
        
        # Clip total bonus to reasonable range
        bonus = np.clip(bonus, -5.0, 5.0)
        
        metrics['spice_bonus'] = bonus
        return bonus, metrics
    
    def resample_to_target(self, waveform: np.ndarray, n_points: int) -> np.ndarray:
        """
        Resample full-res waveform to match target/surrogate length.
        Uses proper interpolation instead of naive index selection.
        """
        x_old = np.linspace(0, 1, len(waveform))
        x_new = np.linspace(0, 1, n_points)
        return np.interp(x_new, x_old, waveform)
    
    def get_stats(self) -> Dict:
        """Get simulation statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        total_attempts = self.sim_count + self.fail_count
        success_rate = self.sim_count / total_attempts if total_attempts > 0 else 0
        return {
            'cache_size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'sim_success': self.sim_count,
            'sim_fail': self.fail_count,
            'success_rate': success_rate,
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
