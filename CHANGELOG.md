# Changelog — Neural Surrogate + RL for Power Electronics

## [2026-02-11] Root Cause Fix: 3 Critical Bugs (commit `2d84f67`)

### Summary

After extensive debugging of the RL training pipeline, we identified that **the RL agent was working correctly** — the root cause of poor performance was the **surrogate model itself**. Three interconnected bugs caused the surrogate to produce meaningless outputs, which meant the RL agent was optimizing against garbage targets.

**Result of these bugs:** The surrogate's `predict_voltage()` always returned the *exact theoretical formula* (e.g., `V_in * D` for buck), ignoring the model's actual learned prediction. The RL agent appeared to "converge" quickly (because matching theory is trivial), but SPICE validation showed the real circuit behaved differently — and the reward system couldn't detect this because it used waveform MSE (a meaningless metric across domains).

---

### Bug 1: Parameter Truncation in Data Generation

**File:** `data/generate_extended_topologies.py`  
**Severity:** Critical — 4 of 7 topologies had broken training data

**The Problem:**  
When combining per-topology SPICE simulation results into the training dataset, the code used:
```python
param_values = list(params.values())[:6]
```
This blindly took the first 6 values from the parameter dictionary. But different topologies have different parameter dictionaries:

| Topology | Dict keys (in order) | What `[:6]` captured | What was **dropped** |
|----------|---------------------|---------------------|---------------------|
| Buck/Boost/Buck-Boost | L, C, R_load, V_in, f_sw, duty | All 6 ✓ | Nothing |
| SEPIC/Ćuk | L1, L2, C_couple, C_out, R_load, V_in, f_sw, duty | L1, L2, C_couple, C_out, R_load, V_in | **f_sw AND duty** |
| Flyback/QR_Flyback | L_pri, C, R_load, V_in, n_ratio, f_sw, duty | L_pri, C, R_load, V_in, n_ratio, f_sw | **duty** |

The surrogate model's canonical input is `[L, C, R_load, V_in, f_sw, duty]`. Without the duty cycle, the model literally **cannot learn** the input-output relationship for these topologies, since output voltage is fundamentally a function of duty cycle.

**The Fix:**  
Added `map_to_canonical_params()` function that explicitly maps each topology's parameter dictionary to the canonical `[L, C, R_load, V_in, f_sw, duty]` order:

```python
def map_to_canonical_params(sample_params, topology):
    if topology in [Topology.BUCK, Topology.BOOST, Topology.BUCK_BOOST]:
        return [params['L'], params['C'], params['R_load'], params['V_in'], params['f_sw'], params['duty']]
    elif topology in [Topology.SEPIC, Topology.CUK]:
        return [params['L1'], params['C_out'], params['R_load'], params['V_in'], params['f_sw'], params['duty']]
    elif topology in [Topology.FLYBACK, Topology.QR_FLYBACK]:
        return [params['L_pri'], params['C'], params['R_load'], params['V_in'], params['f_sw'], params['duty']]
```

**Why this matters:**  
- SEPIC mapped `L1→L` (primary inductor) and `C_out→C` (output capacitor), which are the physically analogous components
- Flyback mapped `L_pri→L` (primary winding inductance)
- All topologies now correctly include `f_sw` (switching frequency) and `duty` (duty cycle)

---

### Bug 2: Denormalization Override in Surrogate Model

**File:** `models/multi_topology_surrogate.py`  
**Severity:** Critical — model output was completely discarded

**The Problem:**  
The `denormalize_waveform()` method was supposed to convert the model's normalized output back to actual voltage values. Instead, it **replaced the entire model output** with theoretical DC voltage + a synthetic 5% ripple:

```python
# OLD (broken) — this was the actual code:
def denormalize_waveform(self, waveform, params, topology):
    v_out = compute_theoretical_vout(v_in, duty, topology)  # e.g., V_in * D for buck
    waveform_centered = waveform - waveform.mean()
    ripple_scale = 0.05 * v_out   # Synthetic 5% ripple
    return v_out + waveform_centered * ripple_scale  # MODEL OUTPUT THROWN AWAY
```

What this meant:
- For buck with V_in=12V, D=0.5: output was ALWAYS 6.0V ± 0.3V ripple
- The model's actual prediction (which might predict 5.57V based on real SPICE data) was discarded
- This made the surrogate look "perfect" (always matches theory) while actually being useless

**Diagnostic evidence:**
```
Raw model output:  buck=7.0V, boost=28.4V, buck_boost=-14.6V
After denorm:      buck=6.0V, boost=24.0V, buck_boost=-6.0V  ← ALL theory!
SPICE ground truth: buck=5.57V                                ← 0.43V gap hidden!
```

**The Fix:**  
`denormalize_waveform()` now reverses the per-topology `(mean, std)` normalization that was applied during training:

```python
# NEW (fixed):
def denormalize_waveform(self, waveform, params, topology):
    if hasattr(self, '_waveform_stats') and self._waveform_stats is not None:
        stats = self._waveform_stats.get(topo_id)
        if stats is not None:
            return waveform * stats['std'] + stats['mean']  # Proper inverse transform
    
    # Fallback for old checkpoints (backward compat — still uses theory)
    ...
```

The `_waveform_stats` dictionary is loaded from the checkpoint (saved during surrogate training). The `load_trained_model()` function was also updated to load these stats:

```python
if 'waveform_stats' in checkpoint:
    model._waveform_stats = checkpoint['waveform_stats']
```

**Same fix applied to:** `huggingface_deploy/models/multi_topology_surrogate.py`

---

### Bug 3: Waveform-MSE-Based SPICE Reward (Domain Mismatch)

**File:** `train_intensive_spice.py`  
**Severity:** Critical — reward signal was meaningless

**The Problem:**  
The SPICE-in-the-loop reward used waveform MSE to compute "agreement" between SPICE and the surrogate:

```python
# OLD (broken):
spice_mse = np.mean((spice_resampled - predicted) ** 2)
agreement = 1.0 / (1.0 + spice_mse / 10.0)
```

This metric was **fundamentally meaningless** because:
1. SPICE produces raw transient waveforms with startup artifacts, ringing, and full switching behavior
2. The surrogate produces smooth, normalized waveforms in a completely different domain
3. Even a **perfect** surrogate would have MSE of 50-2000+ vs SPICE due to domain differences
4. The `agreement` was therefore always near 0, making SPICE validation useless — the reward was dominated by the surrogate-only term

**Evidence:**
```
SPICE MSE: 1606.7  →  agreement = 1/(1+160.67) = 0.006  ← functionally zero
```

**The Fix:**  
Replaced waveform MSE with **DC voltage agreement** — a domain-invariant metric:

```python
# NEW (fixed) — Step function:
spice_dc = float(np.mean(spice_resampled))
surrogate_dc = float(np.mean(predicted))
dc_agreement_err = abs(spice_dc - surrogate_dc) / (abs(surrogate_dc) + 1e-6)
agreement = 1.0 / (1.0 + dc_agreement_err * 5.0)

# DC accuracy vs target:
spice_dc_error_pct = abs(spice_dc - target_dc) / (abs(target_dc) + 1e-6)
if spice_dc_error_pct < 0.3:  dc_term = +3.0 * (1 - err/0.3)   # Bonus
elif spice_dc_error_pct < 0.5: dc_term = 0.0                    # Neutral
else:                          dc_term = -penalty                # Penalty

reward = reward * agreement + dc_term + spice_bonus
```

**Why DC voltage works:**
- Mean voltage is invariant to waveform representation (time-domain, frequency-domain, different sampling rates)
- A buck converter at 12V/0.5D should output ~6V DC whether measured by SPICE or the surrogate
- 20% DC gap → agreement ≈ 0.5 (reasonable modulation), not the 0.006 we saw with MSE

**Same fix applied to:** the validation block (iterations 1, 5, 10, then every 10th) which reports `dc_err%` as the primary metric instead of the old `spice_mse`.

---

### Bug 3b: Per-Sample Waveform Normalization During Training

**File:** `models/train_multi_topology.py`  
**Severity:** High — magnitude information destroyed

**The Problem:**  
The `load_extended_dataset()` function normalized each waveform sample independently:

```python
# OLD:
waveform_max = np.abs(waveforms).max(axis=1, keepdims=True)
waveforms_norm = waveforms / waveform_max
```

This per-sample normalization mapped every waveform to the [-1, 1] range, which meant:
- A 5V buck output and a 200V flyback output looked **identical** to the model
- The model could only learn waveform *shape* (ripple pattern), not *magnitude* (voltage level)
- This is why the old `denormalize_waveform()` had to use theoretical formulas — the model's output contained no magnitude information

**The Fix:**  
Per-topology global normalization using `(mean, std)`:

```python
# NEW:
for topo_id in range(num_topologies):
    mask = topologies == topo_id
    topo_mean = waveforms[mask].mean()
    topo_std = waveforms[mask].std()
    waveforms_norm[mask] = (waveforms[mask] - topo_mean) / topo_std
    waveform_stats[topo_id] = {'mean': topo_mean, 'std': topo_std}
```

This preserves the information that buck outputs ~6V (mean ~6, std ~0.5) while boost outputs ~24V (mean ~24, std ~2), etc. The stats are saved in the checkpoint and used by `denormalize_waveform()` to reverse the transform.

The `param_stats` (min/max per parameter column) are also saved for potential use in normalization consistency checks.

---

## Pipeline Changes

### New: `retrain_and_run.sh`

Automated pipeline script that:
1. Verifies the new dataset exists and has the right shape
2. Backs up the old surrogate checkpoint
3. Retrains the surrogate with the fixed normalization
4. Validates the new checkpoint loads correctly
5. Starts RL training with the new surrogate

### Data Regeneration

All 35,000 samples (5,000 per topology × 7 topologies) are being regenerated from scratch using ngspice, with the `map_to_canonical_params()` fix ensuring correct parameter ordering. This takes approximately 2-2.5 hours on the M3 MacBook Air.

---

## How the Bugs Interacted (Cascading Failure)

```
Bug 1 (Param Truncation)
   → Training data for 4/7 topologies is missing duty cycle
   → Model can't learn duty-cycle dependence for SEPIC, Ćuk, Flyback, QR_Flyback
   
Bug 3b (Per-Sample Normalization)  
   → All magnitude info destroyed during training
   → Model can only learn waveform shapes, not voltage levels
   → Even Buck/Boost/Buck-Boost (which had correct params) lost magnitude

Bug 2 (Denormalization Override)
   → Model output replaced with theoretical formula
   → "Hides" Bugs 1 and 3b — surrogate *appears* to work perfectly
   → Surrogate always outputs V_in*D for buck, V_in/(1-D) for boost, etc.
   → Masked the fact that the model learned nothing useful

Bug 3 (Waveform MSE Reward)
   → SPICE validation can't detect the deception
   → Agreement always ~0 due to domain mismatch (not due to actual disagreement)
   → RL agent optimizes purely against the fake surrogate, never gets corrected by SPICE

Result: RL agent "converges" quickly (matching theory is trivial),
        but real-world SPICE performance is uncontrolled.
```

---

## Current Status (Feb 11, 2026)

| Component | Status |
|-----------|--------|
| Data generation | Running (PID 15812), ~2h remaining |
| Surrogate retraining | Waiting for data gen |
| RL training | Waiting for surrogate |
| Code pushed | ✓ commit `2d84f67` on `main` |

## Expected Impact

After retraining with the fixed data and normalization:
- Surrogate should learn **actual voltage magnitudes** (not just shapes)
- `denormalize_waveform()` will return the model's **real predictions** (not theoretical overrides)
- SPICE DC agreement should be **meaningful** (high when truly matching, low when diverging)
- RL agent will get **corrective feedback** from SPICE, preventing reward hacking
- Complex topologies (SEPIC, Ćuk, Flyback) should actually improve over training (they had no duty cycle before!)
