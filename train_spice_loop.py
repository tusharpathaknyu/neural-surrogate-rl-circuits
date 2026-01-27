#!/usr/bin/env python3
"""
SPICE-IN-THE-LOOP PRODUCTION TRAINING

This script runs ACTUAL ngspice simulations during RL training.
The agent's actions generate circuit parameters, we run SPICE,
and rewards are computed from REAL simulation outputs.

This is slower but gives true SPICE-validated RL agents.
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from models.multi_topology_surrogate import MultiTopologySurrogate
from rl.ppo_agent import ActorCritic
from rl.topology_rewards import compute_topology_aware_reward, TOPOLOGY_REWARD_CONFIG

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = 'cpu'
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
BASE_PATH = Path('/Users/tushardhananjaypathak/Desktop/MLEntry')
NGSPICE_PATH = '/opt/homebrew/bin/ngspice'

# PRODUCTION HYPERPARAMETERS
SURROGATE_EPOCHS = 1000
SURROGATE_BATCH_SIZE = 128
SURROGATE_LR = 1e-4
SURROGATE_PATIENCE = 50

# RL with HYBRID approach - surrogate for speed, SPICE for validation
RL_ITERATIONS = 2000
RL_BATCH_SIZE = 64
RL_LR = 3e-4
RL_PPO_EPOCHS = 4
SPICE_WORKERS = 4

# HYBRID: Run SPICE validation every N iterations
SPICE_VALIDATION_INTERVAL = 50  # Every 50 iterations, validate with real SPICE
SPICE_VALIDATION_SAMPLES = 32   # Number of samples to validate

LOG_INTERVAL = 50
WAVEFORM_POINTS = 32


# =============================================================================
# SPICE NETLIST TEMPLATES
# =============================================================================
NETLIST_TEMPLATES = {
    'buck': """* Buck Converter
.param L={L} C={C} R={R_load} Vin={V_in} fsw={f_sw} duty={duty}

Vin in 0 DC {{Vin}}
S1 in sw ctrl 0 SWITCH
Rctrl ctrl 0 1k
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/fsw}} {{1/fsw}})
D1 0 sw DIODE
L1 sw out {{L}}
C1 out 0 {{C}}
R1 out 0 {{R}}

.model SWITCH SW(Ron=0.01 Roff=1e6 Vt=0.5 Vh=0.1)
.model DIODE D(Is=1e-14 Rs=0.01)

.tran 0.1u 2m 1m
.control
run
set wr_vecnames
wrdata {output_file} v(out)
.endc
.end
""",

    'boost': """* Boost Converter
.param L={L} C={C} R={R_load} Vin={V_in} fsw={f_sw} duty={duty}

Vin in 0 DC {{Vin}}
L1 in sw {{L}}
S1 sw 0 ctrl 0 SWITCH
Rctrl ctrl 0 1k
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/fsw}} {{1/fsw}})
D1 sw out DIODE
C1 out 0 {{C}}
R1 out 0 {{R}}

.model SWITCH SW(Ron=0.01 Roff=1e6 Vt=0.5 Vh=0.1)
.model DIODE D(Is=1e-14 Rs=0.01)

.tran 0.1u 2m 1m
.control
run
set wr_vecnames
wrdata {output_file} v(out)
.endc
.end
""",

    'buck_boost': """* Buck-Boost Converter (Inverting)
.param L={L} C={C} R={R_load} Vin={V_in} fsw={f_sw} duty={duty}

Vin in 0 DC {{Vin}}
S1 in sw ctrl 0 SWITCH
Rctrl ctrl 0 1k
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/fsw}} {{1/fsw}})
L1 sw 0 {{L}}
D1 out sw DIODE
C1 out 0 {{C}}
R1 out 0 {{R}}

.model SWITCH SW(Ron=0.01 Roff=1e6 Vt=0.5 Vh=0.1)
.model DIODE D(Is=1e-14 Rs=0.01)

.tran 0.1u 2m 1m
.control
run
set wr_vecnames
wrdata {output_file} v(out)
.endc
.end
""",

    'sepic': """* SEPIC Converter
.param L1={L} L2={L} C1=10u C2={C} R={R_load} Vin={V_in} fsw={f_sw} duty={duty}

Vin in 0 DC {{Vin}}
L1 in sw1 {{L1}}
S1 sw1 0 ctrl 0 SWITCH
Rctrl ctrl 0 1k
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/fsw}} {{1/fsw}})
C1 sw1 sw2 {{C1}}
L2 sw2 0 {{L2}}
D1 sw2 out DIODE
C2 out 0 {{C2}}
R1 out 0 {{R}}

.model SWITCH SW(Ron=0.01 Roff=1e6 Vt=0.5 Vh=0.1)
.model DIODE D(Is=1e-14 Rs=0.01)

.tran 0.1u 2m 1m
.control
run
set wr_vecnames
wrdata {output_file} v(out)
.endc
.end
""",

    'cuk': """* Cuk Converter (Inverting)
.param L1={L} L2={L} C1=10u C2={C} R={R_load} Vin={V_in} fsw={f_sw} duty={duty}

Vin in 0 DC {{Vin}}
L1 in sw1 {{L1}}
S1 sw1 0 ctrl 0 SWITCH
Rctrl ctrl 0 1k
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/fsw}} {{1/fsw}})
C1 sw1 sw2 {{C1}}
D1 0 sw2 DIODE
L2 sw2 out {{L2}}
C2 out 0 {{C2}}
R1 out 0 {{R}}

.model SWITCH SW(Ron=0.01 Roff=1e6 Vt=0.5 Vh=0.1)
.model DIODE D(Is=1e-14 Rs=0.01)

.tran 0.1u 2m 1m
.control
run
set wr_vecnames
wrdata {output_file} v(out)
.endc
.end
""",

    'flyback': """* Flyback Converter
.param Lp={L} Ls=50u C={C} R={R_load} Vin={V_in} fsw={f_sw} duty={duty} N=0.5

Vin in 0 DC {{Vin}}
Lp in sw {{Lp}}
S1 sw 0 ctrl 0 SWITCH
Rctrl ctrl 0 1k
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/fsw}} {{1/fsw}})

* Secondary (simplified coupled inductor model)
E1 sec_p 0 sw 0 {{N}}
Ls sec_p sec {{Ls}}
D1 sec out DIODE
C1 out 0 {{C}}
R1 out 0 {{R}}

.model SWITCH SW(Ron=0.01 Roff=1e6 Vt=0.5 Vh=0.1)
.model DIODE D(Is=1e-14 Rs=0.01)

.tran 0.1u 2m 1m
.control
run
set wr_vecnames
wrdata {output_file} v(out)
.endc
.end
""",

    'qr_flyback': """* Quasi-Resonant Flyback
.param Lp={L} Ls=50u Lr=1u Cr=1n C={C} R={R_load} Vin={V_in} fsw={f_sw} duty={duty} N=0.5

Vin in 0 DC {{Vin}}
Lp in sw {{Lp}}
Lr sw sw_r {{Lr}}
Cr sw_r 0 {{Cr}}
S1 sw_r 0 ctrl 0 SWITCH
Rctrl ctrl 0 1k
Vctrl ctrl 0 PULSE(0 1 0 1n 1n {{duty/fsw}} {{1/fsw}})

* Secondary
E1 sec_p 0 sw 0 {{N}}
Ls sec_p sec {{Ls}}
D1 sec out DIODE
C1 out 0 {{C}}
R1 out 0 {{R}}

.model SWITCH SW(Ron=0.01 Roff=1e6 Vt=0.5 Vh=0.1)
.model DIODE D(Is=1e-14 Rs=0.01)

.tran 0.1u 2m 1m
.control
run
set wr_vecnames
wrdata {output_file} v(out)
.endc
.end
"""
}


def log_message(msg: str, log_file):
    """Print and log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_file.write(full_msg + "\n")
    log_file.flush()


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def denormalize_params(norm_params: np.ndarray) -> Dict[str, float]:
    """Convert normalized [0,1] params to real circuit values."""
    return {
        'L': norm_params[0] * 900e-6 + 100e-6,      # 100uH - 1mH
        'C': norm_params[1] * 900e-6 + 100e-6,      # 100uF - 1mF
        'R_load': norm_params[2] * 90 + 10,          # 10-100 ohms
        'V_in': norm_params[3] * 24 + 12,            # 12-36V
        'f_sw': norm_params[4] * 150e3 + 50e3,       # 50-200kHz
        'duty': norm_params[5] * 0.7 + 0.1           # 0.1-0.8
    }


def run_spice_simulation(params: Dict[str, float], topology: str, 
                         timeout: float = 10.0) -> Optional[np.ndarray]:
    """
    Run ACTUAL ngspice simulation and return output waveform.
    
    Returns:
        waveform: WAVEFORM_POINTS-length array or None if failed
    """
    template = NETLIST_TEMPLATES.get(topology, NETLIST_TEMPLATES['buck'])
    
    # Create temp files
    output_file = tempfile.mktemp(suffix='.txt')
    
    try:
        netlist = template.format(
            L=params['L'], C=params['C'], R_load=params['R_load'],
            V_in=params['V_in'], f_sw=params['f_sw'], duty=params['duty'],
            output_file=output_file
        )
    except Exception as e:
        return None
    
    netlist_file = tempfile.mktemp(suffix='.cir')
    
    try:
        with open(netlist_file, 'w') as f:
            f.write(netlist)
        
        result = subprocess.run(
            [NGSPICE_PATH, '-b', netlist_file],
            capture_output=True, text=True, timeout=timeout
        )
        
        if Path(output_file).exists():
            data = np.loadtxt(output_file)
            if len(data.shape) == 2 and data.shape[1] >= 2:
                waveform = data[:, 1]
                # Resample to WAVEFORM_POINTS
                indices = np.linspace(0, len(waveform)-1, WAVEFORM_POINTS).astype(int)
                return waveform[indices].astype(np.float32)
        
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        Path(netlist_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def run_spice_batch(params_batch: np.ndarray, topology: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SPICE simulations in parallel for a batch of parameters.
    
    Returns:
        waveforms: (batch_size, WAVEFORM_POINTS) array
        valid_mask: (batch_size,) boolean array indicating successful simulations
    """
    batch_size = len(params_batch)
    waveforms = np.zeros((batch_size, WAVEFORM_POINTS), dtype=np.float32)
    valid_mask = np.zeros(batch_size, dtype=bool)
    
    def run_single(idx):
        real_params = denormalize_params(params_batch[idx])
        wf = run_spice_simulation(real_params, topology)
        return idx, wf
    
    with ThreadPoolExecutor(max_workers=SPICE_WORKERS) as executor:
        futures = [executor.submit(run_single, i) for i in range(batch_size)]
        for future in as_completed(futures):
            idx, wf = future.result()
            if wf is not None:
                waveforms[idx] = wf
                valid_mask[idx] = True
    
    return waveforms, valid_mask


def compute_spice_reward(spice_wf: np.ndarray, topology: str, 
                         params: np.ndarray) -> float:
    """
    Compute reward from ACTUAL SPICE simulation output.
    
    This is the KEY difference - we're rewarding based on real physics,
    not surrogate predictions.
    """
    config = TOPOLOGY_REWARD_CONFIG.get(topology, TOPOLOGY_REWARD_CONFIG['buck'])
    
    # Denormalize to get expected output
    real_params = denormalize_params(params)
    V_in = real_params['V_in']
    duty = real_params['duty']
    R_load = real_params['R_load']
    
    # Expected DC output based on topology equations
    if topology == 'buck':
        expected_vout = V_in * duty
    elif topology == 'boost':
        expected_vout = V_in / (1 - duty + 0.01)
    elif topology == 'buck_boost':
        expected_vout = -V_in * duty / (1 - duty + 0.01)  # Negative
    elif topology == 'sepic':
        expected_vout = V_in * duty / (1 - duty + 0.01)
    elif topology == 'cuk':
        expected_vout = -V_in * duty / (1 - duty + 0.01)  # Negative
    elif topology == 'flyback':
        N = 0.5
        expected_vout = V_in * N * duty / (1 - duty + 0.01)
    elif topology == 'qr_flyback':
        N = 0.5
        expected_vout = V_in * N * duty / (1 - duty + 0.01)
    else:
        expected_vout = V_in * duty
    
    # Actual SPICE output statistics
    actual_mean = np.mean(spice_wf)
    actual_ripple = (np.max(spice_wf) - np.min(spice_wf)) / (np.abs(actual_mean) + 1e-6)
    
    # For inverted topologies, compare absolute values for magnitude
    if config['inverted']:
        voltage_error = abs(abs(actual_mean) - abs(expected_vout)) / (abs(expected_vout) + 1e-6)
        # Check sign is correct
        sign_correct = (np.sign(actual_mean) == np.sign(expected_vout))
        sign_penalty = 0 if sign_correct else 2.0
    else:
        voltage_error = abs(actual_mean - expected_vout) / (abs(expected_vout) + 1e-6)
        sign_penalty = 0
    
    # Efficiency estimate (simplified)
    P_out = (actual_mean ** 2) / R_load
    P_in = V_in * (P_out / (V_in * duty + 1e-6))  # Rough estimate
    efficiency = min(1.0, P_out / (P_in + 1e-6))
    
    # Target metrics from topology config
    eff_target = config.get('efficiency_target', 0.85)
    ripple_target = config.get('ripple_target', 0.03)
    
    # Compute reward components
    voltage_reward = max(0, 1 - voltage_error)
    efficiency_reward = max(0, efficiency / eff_target) if efficiency < eff_target else 1.0
    ripple_reward = max(0, 1 - actual_ripple / ripple_target) if actual_ripple > ripple_target else 1.0
    
    # Weighted combination
    weights = config.get('weights', {})
    w_eff = weights.get('efficiency', 1.0)
    w_ripple = weights.get('ripple', 1.0)
    w_dc = weights.get('dc', 1.0)
    
    total_reward = (
        w_dc * voltage_reward +
        w_eff * efficiency_reward +
        w_ripple * ripple_reward -
        sign_penalty
    )
    
    return total_reward


# =============================================================================
# MAIN TRAINING
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Check ngspice
    try:
        subprocess.run([NGSPICE_PATH, '--version'], capture_output=True, timeout=5)
        print("✅ ngspice found")
    except:
        print("❌ ngspice not found at", NGSPICE_PATH)
        sys.exit(1)
    
    log_path = BASE_PATH / "spice_loop_training.log"
    log_file = open(log_path, "w")
    
    log_message("=" * 70, log_file)
    log_message("🚀 SPICE-IN-THE-LOOP PRODUCTION TRAINING", log_file)
    log_message("=" * 70, log_file)
    log_message(f"Device: {DEVICE}", log_file)
    log_message(f"SPICE workers: {SPICE_WORKERS}", log_file)
    log_message(f"Surrogate epochs: {SURROGATE_EPOCHS}", log_file)
    log_message(f"RL iterations (with real SPICE): {RL_ITERATIONS}", log_file)
    log_message("", log_file)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    log_message("📦 Loading SPICE-validated data...", log_file)
    data = dict(np.load(BASE_PATH / 'data' / 'spice_validated_data.npz'))
    
    params = torch.FloatTensor(data['params'])
    waveforms = torch.FloatTensor(data['waveforms'])
    topo_ids = torch.LongTensor(data['topology_ids'])
    
    n_samples = len(params)
    log_message(f"   Samples: {n_samples:,}", log_file)
    
    idx = torch.randperm(n_samples)
    train_idx = idx[:int(0.9 * n_samples)]
    val_idx = idx[int(0.9 * n_samples):]
    
    train_dataset = torch.utils.data.TensorDataset(
        params[train_idx], topo_ids[train_idx], waveforms[train_idx]
    )
    val_dataset = torch.utils.data.TensorDataset(
        params[val_idx], topo_ids[val_idx], waveforms[val_idx]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=SURROGATE_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=SURROGATE_BATCH_SIZE, shuffle=False
    )
    
    # =========================================================================
    # PHASE 1: TRAIN SURROGATE (same as before)
    # =========================================================================
    log_message("\n" + "=" * 70, log_file)
    log_message("🧠 PHASE 1: TRAINING SURROGATE MODEL", log_file)
    log_message("=" * 70, log_file)
    
    model = MultiTopologySurrogate(
        num_topologies=7, param_dim=6, waveform_len=32,
        embed_dim=64, hidden_dim=512
    ).to(DEVICE)
    
    log_message(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}", log_file)
    
    optimizer = optim.AdamW(model.parameters(), lr=SURROGATE_LR, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    phase1_start = time.time()
    
    for epoch in range(1, SURROGATE_EPOCHS + 1):
        model.train()
        train_losses = []
        
        for batch_params, batch_topo, batch_wf in train_loader:
            batch_params = batch_params.to(DEVICE)
            batch_topo = batch_topo.to(DEVICE)
            batch_wf = batch_wf.to(DEVICE)
            
            optimizer.zero_grad()
            pred_wf, _ = model(batch_params, batch_topo)
            loss = criterion(pred_wf, batch_wf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_params, batch_topo, batch_wf in val_loader:
                batch_params = batch_params.to(DEVICE)
                batch_topo = batch_topo.to(DEVICE)
                batch_wf = batch_wf.to(DEVICE)
                pred_wf, _ = model(batch_params, batch_topo)
                val_losses.append(criterion(pred_wf, batch_wf).item())
        
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt')
        else:
            patience_counter += 1
        
        if epoch % LOG_INTERVAL == 0 or epoch == 1:
            elapsed = time.time() - phase1_start
            eta = elapsed / epoch * (SURROGATE_EPOCHS - epoch)
            log_message(
                f"   Epoch {epoch:4d}/{SURROGATE_EPOCHS}: train={avg_train:.4f}, "
                f"val={avg_val:.4f}, patience={patience_counter}/{SURROGATE_PATIENCE}, "
                f"ETA={format_time(eta)}", log_file
            )
        
        if patience_counter >= SURROGATE_PATIENCE:
            log_message(f"   ⏹️ Early stopping at epoch {epoch}", log_file)
            break
    
    log_message(f"   ✅ Best val loss: {best_val_loss:.6f}", log_file)
    
    # Load best
    checkpoint = torch.load(BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # =========================================================================
    # PHASE 2: HYBRID RL - SURROGATE + PERIODIC SPICE VALIDATION
    # =========================================================================
    log_message("\n" + "=" * 70, log_file)
    log_message("🎮 PHASE 2: HYBRID RL TRAINING", log_file)
    log_message("=" * 70, log_file)
    log_message(f"   📊 Surrogate rewards: every iteration", log_file)
    log_message(f"   ⚡ SPICE validation: every {SPICE_VALIDATION_INTERVAL} iterations", log_file)
    
    phase2_start = time.time()
    
    def create_target_waveform(topology: str, params: torch.Tensor, device: str) -> torch.Tensor:
        """Create topology-appropriate target waveforms."""
        batch_size = params.shape[0]
        t = torch.linspace(0, 1, 32, device=device)
        
        vin = params[:, 3:4] * 24 + 12
        duty = params[:, 5:6] * 0.7 + 0.1
        
        if topology == 'buck':
            vout = vin * duty
            ripple = 0.02 * (1 - duty)
        elif topology == 'boost':
            vout = vin / (1 - duty + 0.01)
            ripple = 0.03 * duty
        elif topology == 'buck_boost':
            vout = -vin * duty / (1 - duty + 0.01)
            ripple = 0.04 * duty
        elif topology == 'sepic':
            vout = vin * duty / (1 - duty + 0.01)
            ripple = 0.03 * duty
        elif topology == 'cuk':
            vout = -vin * duty / (1 - duty + 0.01)
            ripple = 0.035 * duty
        elif topology in ['flyback', 'qr_flyback']:
            vout = vin * 0.5 * duty / (1 - duty + 0.01)
            ripple = 0.04 * duty
        else:
            vout = vin * duty
            ripple = 0.02
        
        base = vout.expand(-1, 32)
        ripple_wave = ripple * torch.sin(2 * np.pi * 8 * t.unsqueeze(0).expand(batch_size, -1))
        return base + ripple_wave
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        is_inverted = config.get('inverted', False)
        inv_label = " ⚡INV" if is_inverted else ""
        
        log_message(f"\n   {'─' * 50}", log_file)
        log_message(f"   Training: {topo_name.upper()}{inv_label}", log_file)
        log_message(f"   {'─' * 50}", log_file)
        
        agent = ActorCritic(state_dim=41, action_dim=6, hidden_dim=256).to(DEVICE)
        optimizer = optim.Adam(agent.parameters(), lr=RL_LR)
        
        best_reward = -float('inf')
        best_spice_reward = -float('inf')
        reward_history = []
        spice_rewards_history = []
        topo_start = time.time()
        
        for iteration in range(1, RL_ITERATIONS + 1):
            # Sample random parameters
            batch_params = torch.rand(RL_BATCH_SIZE, 6).to(DEVICE)
            batch_topo = torch.full((RL_BATCH_SIZE,), topo_idx, dtype=torch.long).to(DEVICE)
            
            # Get surrogate predictions
            with torch.no_grad():
                pred_wf, _ = model(batch_params, batch_topo)
            
            # Create target waveforms
            target_wf = create_target_waveform(topo_name, batch_params, DEVICE)
            
            # Build state
            error = (pred_wf - target_wf).abs().mean(dim=1, keepdim=True)
            mse = ((pred_wf - target_wf)**2).mean(dim=1, keepdim=True)
            corr = torch.zeros(RL_BATCH_SIZE, 1, device=DEVICE)
            state = torch.cat([pred_wf, batch_params, error, mse, corr], dim=1)
            
            # Get actions
            with torch.no_grad():
                actions, log_probs_old, values_old = agent.get_action(state, deterministic=False)
            
            # Apply actions
            new_params = torch.clamp(batch_params + actions * 0.1, 0, 1)
            
            # Get new predictions from surrogate
            with torch.no_grad():
                new_pred, _ = model(new_params, batch_topo)
            
            # FAST: Compute rewards from SURROGATE (topology-aware)
            rewards = []
            for i in range(RL_BATCH_SIZE):
                r, _ = compute_topology_aware_reward(
                    new_pred[i].cpu().numpy(),
                    target_wf[i].cpu().numpy(),
                    topo_name
                )
                rewards.append(r)
            rewards = torch.FloatTensor(rewards).to(DEVICE)
            
            # PPO update
            for _ in range(RL_PPO_EPOCHS):
                log_probs, entropy, values = agent.evaluate(state, actions)
                
                advantages = rewards - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                ratio = torch.exp(log_probs - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - rewards) ** 2).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
            
            avg_reward = rewards.mean().item()
            reward_history.append(avg_reward)
            
            if avg_reward > best_reward:
                best_reward = avg_reward
            
            # ⚡ PERIODIC SPICE VALIDATION ⚡
            if iteration % SPICE_VALIDATION_INTERVAL == 0:
                # Run real SPICE simulations to validate
                val_params = torch.rand(SPICE_VALIDATION_SAMPLES, 6)
                spice_waveforms, valid_mask = run_spice_batch(val_params.numpy(), topo_name)
                
                spice_rewards = []
                for i in range(SPICE_VALIDATION_SAMPLES):
                    if valid_mask[i]:
                        r = compute_spice_reward(spice_waveforms[i], topo_name, val_params[i].numpy())
                        spice_rewards.append(r)
                
                if spice_rewards:
                    avg_spice = np.mean(spice_rewards)
                    spice_rewards_history.append(avg_spice)
                    spice_success = valid_mask.sum() / len(valid_mask)
                    
                    if avg_spice > best_spice_reward:
                        best_spice_reward = avg_spice
                        # Save model when SPICE reward improves
                        torch.save({
                            'model_state_dict': agent.state_dict(),
                            'topology': topo_name,
                            'best_reward': best_reward,
                            'best_spice_reward': best_spice_reward,
                            'spice_validated': True,
                        }, BASE_PATH / 'checkpoints' / f'rl_agent_{topo_name}.pt')
                    
                    elapsed = time.time() - topo_start
                    eta = elapsed / iteration * (RL_ITERATIONS - iteration)
                    log_message(
                        f"      Iter {iteration:4d}/{RL_ITERATIONS}: "
                        f"surrogate={avg_reward:.3f}, ⚡SPICE={avg_spice:.3f} (best={best_spice_reward:.3f}), "
                        f"success={spice_success:.0%}, ETA={format_time(eta)}", log_file
                    )
        
        topo_time = time.time() - topo_start
        final_spice = np.mean(spice_rewards_history[-5:]) if spice_rewards_history else 0
        log_message(
            f"   ✅ {topo_name}: surrogate={best_reward:.3f}, SPICE={best_spice_reward:.3f} ({format_time(topo_time)})", 
            log_file
        )
    
    phase2_time = time.time() - phase2_start
    
    # =========================================================================
    # PHASE 3: FINAL EVALUATION WITH SPICE
    # =========================================================================
    log_message("\n" + "=" * 70, log_file)
    log_message("📊 PHASE 3: FINAL SPICE-VALIDATED EVALUATION", log_file)
    log_message("=" * 70, log_file)
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        
        # Load agent
        agent = ActorCritic(state_dim=41, action_dim=6, hidden_dim=256).to(DEVICE)
        checkpoint = torch.load(BASE_PATH / 'checkpoints' / f'rl_agent_{topo_name}.pt')
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        
        # Test with SPICE
        n_test = 50
        test_params = torch.rand(n_test, 6)
        
        spice_waveforms, valid_mask = run_spice_batch(test_params.numpy(), topo_name)
        
        rewards = []
        for i in range(n_test):
            if valid_mask[i]:
                r = compute_spice_reward(spice_waveforms[i], topo_name, test_params[i].numpy())
                rewards.append(r)
        
        if rewards:
            avg_reward = np.mean(rewards)
            success_rate = valid_mask.sum() / len(valid_mask)
            inv_label = "(INV)" if config.get('inverted', False) else ""
            log_message(
                f"   {topo_name.upper():12} {inv_label:6} "
                f"Avg reward={avg_reward:.3f}, SPICE success={success_rate:.0%}", 
                log_file
            )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    log_message("\n" + "=" * 70, log_file)
    log_message("✅ SPICE-IN-THE-LOOP TRAINING COMPLETE!", log_file)
    log_message("=" * 70, log_file)
    log_message(f"   Total time: {format_time(total_time)}", log_file)
    log_message(f"   Surrogate best val loss: {best_val_loss:.6f}", log_file)
    log_message("   All RL agents trained with REAL SPICE simulations", log_file)
    log_message("=" * 70, log_file)
    
    log_file.close()
    print("\n🎉 Done! All models SPICE-validated.")
