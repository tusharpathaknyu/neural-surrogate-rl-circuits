# Training Guide: QR Flyback on Windows PC

## Overview

This project trains RL agents (PPO) to design DC-DC power converter circuits.
Each topology gets its own agent trained with SPICE-in-the-loop validation.

**Your task:** Train `qr_flyback` on the Windows PC while `flyback` finishes on the Mac.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                 Training Pipeline                    │
│                                                      │
│  1. Surrogate Model (pre-trained neural network)     │
│     └─ Predicts circuit waveforms from parameters    │
│     └─ File: checkpoints/multi_topology_surrogate.pt │
│                                                      │
│  2. RL Environment (TopologySpecificEnv)             │
│     └─ Agent adjusts circuit params (L, C, R, etc.)  │
│     └─ Gets reward based on waveform quality         │
│     └─ Every 5 steps: validates with real SPICE sim  │
│                                                      │
│  3. PPO Agent                                        │
│     └─ Collects rollouts → Updates policy            │
│     └─ 700 iterations × 4096 steps each              │
│     └─ Saves best checkpoint                         │
│                                                      │
│  4. SPICE (ngspice)                                  │
│     └─ Ground-truth circuit simulation               │
│     └─ Runs as subprocess, CPU-only                  │
│     └─ ~0.5s per call                                │
└─────────────────────────────────────────────────────┘
```

## File Structure (What Matters)

```
MLEntry/
├── train_intensive_spice.py          ← MAIN TRAINING SCRIPT (modify TOPOLOGIES list)
├── models/
│   ├── multi_topology_surrogate.py   ← Surrogate model definition + loader
│   └── forward_surrogate.py          ← Base surrogate class
├── rl/
│   ├── environment.py                ← Base RL environment (CircuitDesignEnv)
│   ├── ppo_agent.py                  ← PPO agent (ActorCritic network)
│   ├── spice_reward.py               ← SPICE simulation runner (ngspice integration)
│   └── topology_rewards.py           ← Per-topology reward functions
├── checkpoints/
│   ├── multi_topology_surrogate.pt   ← Pre-trained surrogate (REQUIRED, don't retrain)
│   └── rl_agent_<topology>.pt        ← Saved RL agents (output)
└── spice_templates/                  ← SPICE netlist templates
```

## Training Flow (What train_intensive_spice.py Does)

```python
# Step 1: Load pre-trained surrogate model
surrogate = load_trained_model(device='cpu')  # or 'cuda'

# Step 2: For each topology in TOPOLOGIES list:
for topology in TOPOLOGIES:
    # Create environment with SPICE validation
    env = TopologySpecificEnv(surrogate, topology, use_spice=True, spice_freq=5)
    
    # Create PPO agent with topology-specific hyperparameters
    agent = PPOAgent(env, hidden_dim=512, lr=5e-5, ...)
    
    # Training loop
    for iteration in range(700):  # qr_flyback has 700 iterations
        rollout = agent.collect_rollouts(4096)   # Collect 4096 steps
        agent.update(rollout, n_epochs=10, batch_size=64)  # PPO update
        
        # Every 5 iterations: run SPICE validation
        if (iteration + 1) % 5 == 0:
            # Run 5 episodes with real ngspice
            # Compare surrogate predictions vs SPICE ground truth
        
        # Every 35 iterations: save checkpoint if reward improved
        if improved:
            agent.save('checkpoints/rl_agent_qr_flyback.pt')
```

## QR Flyback Config (Hardest Topology)

```python
'qr_flyback': {
    'hidden_dim': 512,        # Larger network (complex resonant behavior)
    'lr': 5e-5,               # Very low LR (sensitive to ZVS/ZCS dynamics)
    'n_iterations': 700,      # Most iterations (hardest to learn)
    'steps_per_iter': 4096,   # Large batch for stability
    'gamma': 0.998,           # Long-horizon (resonant transients)
    'gae_lambda': 0.98,       # High GAE lambda
    'clip_epsilon': 0.10,     # Tight clipping (careful updates)
    'entropy_coef': 0.03,     # More exploration (complex landscape)
    'spice_freq': 5,          # SPICE validation every 5 iterations
    'description': 'QR soft-switching, ZVS/ZCS, resonant tank',
}
```

---

## Setup Steps for Windows

### 1. Install Python Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy tqdm matplotlib
```

If your GPU is NVIDIA with CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### 2. Install ngspice

**Option A - Download binary:**
- Go to https://ngspice.sourceforge.io/download.html
- Download Windows 64-bit installer
- Install and add to PATH
- Verify: `ngspice --version`

**Option B - Chocolatey:**
```bash
choco install ngspice
```

**Option C - Winget:**
```bash
winget install ngspice
```

Verify it works:
```bash
ngspice --version
```

### 3. Clone the repo (already done)

```bash
git clone https://github.com/tusharpathaknyu/neural-surrogate-rl-circuits.git
cd neural-surrogate-rl-circuits
```

### 4. Modify train_intensive_spice.py

Change ONLY these two things:

**Line ~36 - Set TOPOLOGIES to just qr_flyback:**
```python
# CHANGE THIS:
TOPOLOGIES = ['qr_flyback']
```

**Line ~33 - Set DEVICE to CUDA if you have NVIDIA GPU:**
```python
# CHANGE THIS:
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 5. Verify surrogate loads

```bash
python -c "
import sys; sys.path.append('.')
from models.multi_topology_surrogate import load_trained_model
m = load_trained_model(device='cpu')
print(f'Surrogate loaded: {sum(p.numel() for p in m.parameters()):,} params')
"
```

Should print: `Surrogate loaded: 1,157,731 params`

### 6. Test SPICE integration

```bash
python -c "
from rl.spice_reward import SPICERewardCalculator
import numpy as np
calc = SPICERewardCalculator('qr_flyback')
params = np.array([100e-6, 10e-6, 10.0, 12.0, 100e3, 0.5])
result = calc.simulate(params)
print(f'SPICE result: {result is not None}, shape: {result.shape if result is not None else None}')
"
```

### 7. Run training

```bash
python -u train_intensive_spice.py > training_qr_flyback.log 2>&1
```

Or to run in background (PowerShell):
```powershell
Start-Process -NoNewWindow python -ArgumentList "-u", "train_intensive_spice.py" -RedirectStandardOutput "training_qr_flyback.log" -RedirectStandardError "training_qr_flyback_err.log"
```

Or CMD:
```cmd
start /B python -u train_intensive_spice.py > training_qr_flyback.log 2>&1
```

### 8. Monitor progress

```bash
# Filter out noise, see latest progress:
findstr /V "MallocStackLogging" training_qr_flyback.log | more

# Or just check the checkpoint:
dir checkpoints\rl_agent_qr_flyback.pt
```

### 9. Set Windows to never sleep

```
Settings → System → Power & Sleep → Sleep: Never
```

Or PowerShell:
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

---

## Expected Training Time

| Setup | QR Flyback (700 iters) |
|-------|------------------------|
| CPU only | ~100-120 hours |
| NVIDIA GPU + CPU SPICE | ~70-80 hours |
| Multi-core CPU parallelism | Potential further speedup |

The GPU speeds up the neural network (PPO), but SPICE always runs on CPU.

---

## After Training Completes

The output is: `checkpoints/rl_agent_qr_flyback.pt`

Copy it back to the Mac repo:
```bash
git add checkpoints/rl_agent_qr_flyback.pt
git commit -m "QR flyback agent trained on Windows PC"
git push
```

Then on Mac: `git pull`

---

## Troubleshooting

### "No module named 'models'" or import errors
```bash
# Run from the project root directory:
cd path/to/neural-surrogate-rl-circuits
python -u train_intensive_spice.py
```

### ngspice not found
- Make sure ngspice is in your PATH
- Windows: Check `C:\Program Files\ngspice\bin` is in PATH
- Test: `ngspice --version` in terminal

### CUDA out of memory
- Reduce `steps_per_iter` from 4096 to 2048
- Or set DEVICE = 'cpu'

### Training seems stuck
- Each iteration takes ~8-10 minutes (normal!)
- Check log file is growing: `dir training_qr_flyback.log`
- SPICE calls dominate the time, not the GPU

### Path issues on Windows
- The SPICE templates use `/tmp/` paths which don't exist on Windows
- If SPICE fails, check `rl/spice_reward.py` and change `tempfile` usage
- Python's `tempfile.mkdtemp()` should work cross-platform
