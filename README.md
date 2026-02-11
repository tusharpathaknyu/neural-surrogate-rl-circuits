# Neural Surrogate + RL for Power Electronics Design

A machine learning system that designs DC-DC power converter circuits using neural surrogates and reinforcement learning with SPICE-in-the-loop validation. Covers 7 converter topologies end-to-end: data generation, surrogate training, and per-topology PPO agents grounded by real ngspice simulations.

## Problem

Designing power converters requires iterating through component values (inductors, capacitors, switching frequency, duty cycle) and simulating the output waveform each time. A single ngspice simulation takes ~100ms. Optimizing a design can take hours of manual tuning.

This project replaces that loop:

1. A **neural surrogate** (1.16M parameters) predicts circuit waveforms from 6 component parameters in ~0.001ms -- a 100,000x speedup over SPICE.
2. **PPO agents** use the surrogate to explore millions of parameter combinations, learning to produce target waveforms for each topology.
3. Periodically, real **ngspice simulations** validate the surrogate's predictions (SPICE-in-the-loop), keeping the agent grounded in physical reality.

## Supported Topologies

| Topology | Type | Transfer Function | Agent Status |
|----------|------|-------------------|--------------|
| Buck | Step-down | V_out = V_in * D | Trained |
| Boost | Step-up | V_out = V_in / (1 - D) | Trained |
| Buck-Boost | Inverting | V_out = -V_in * D / (1 - D) | Trained |
| SEPIC | Non-inverting | V_out = V_in * D / (1 - D) | Trained |
| Cuk | Inverting, low ripple | V_out = -V_in * D / (1 - D) | Trained |
| Flyback | Isolated | V_out = V_in * D * n / (1 - D) | Trained (SPICE-validated, 80 hrs) |
| QR Flyback | Resonant, soft-switching | Modified duty with resonant period | Training (SPICE-validated) |

## Architecture

```
                    +-------------------+
                    |   ngspice (SPICE) |  Ground-truth circuit simulator
                    |   ~100ms / call   |  Runs as subprocess, CPU-only
                    +--------+----------+
                             |
                             | Validates every 5 iterations
                             v
+----------------+    +------+-------+    +---------------------+
| Target         |    | PPO Agent    |    | Surrogate Model     |
| Waveform       +--->| (per topo)   +--->| MultiTopologySurr.  |
| (32 points)    |    | ActorCritic  |    | 1.16M params        |
+----------------+    | 6-dim action |    | ~0.001ms / call     |
                      +--------------+    +---------------------+

State:  [32 waveform features | 6 normalized params | 3 error signals] = 41 dims
Action: [delta_L, delta_C, delta_R, delta_Vin, delta_fsw, delta_duty] in [-1, 1]
```

### RL Policy (PPO)

- **Network**: Shared backbone with separate actor and critic heads
  - Shared: Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU
  - Actor: Linear -> ReLU -> Linear -> Tanh (outputs action mean) + learnable log-std
  - Critic: Linear -> ReLU -> Linear (outputs scalar V(s))
- **Action distribution**: Diagonal Gaussian -- samples from N(mu(s), diag(sigma^2))
- **Advantages**: Generalized Advantage Estimation (GAE)
- **Loss**: Clipped surrogate objective + value loss + entropy bonus
- **No external RL libraries** -- pure PyTorch implementation

Topology-specific hyperparameters scale network capacity and exploration for harder topologies (e.g., QR Flyback uses hidden_dim=512, lr=5e-5, clip=0.10, entropy=0.03).

### Reward Function

Topology-aware, multi-objective reward combining engineering metrics:

- **MSE**: Waveform shape matching (primary signal)
- **THD**: Total harmonic distortion error
- **Rise time**: 10-90% rise time matching
- **Ripple**: Output voltage ripple error
- **Overshoot**: Heavy penalty (can damage components in real circuits)
- **DC error**: Mean voltage accuracy
- **Topology-specific terms**: Sign penalty (inverted topologies), ringing penalty (flyback), transition smoothness bonus (QR flyback, rewards ZVS behavior), smoothness bonus (Cuk)

### Surrogate Model

- **Architecture**: MultiTopologySurrogate -- topology embedding + shared encoder + waveform decoder with conv refinement
- **Parameters**: 1,157,731
- **Input**: 6 circuit parameters + topology ID
- **Output**: 512-point voltage waveform (compressed to 32 for RL state)
- **Training data**: 5000+ ngspice simulations per topology with randomized component values

## Circuit Parameters

| Parameter | Range | Scale | Description |
|-----------|-------|-------|-------------|
| L | 10-100 uH | Log | Inductor |
| C | 47-470 uF | Log | Capacitor |
| R_load | 2-50 Ohm | Linear | Load resistance |
| V_in | 10-24 V | Linear | Input voltage |
| f_sw | 50-500 kHz | Log | Switching frequency |
| duty | 20-80% | Linear | Duty cycle |

The agent adjusts these 6 parameters each step (max 20% relative change), with log-scale adjustments for L, C, and f_sw to handle their wide dynamic range.

## Project Structure

```
MLEntry/
|-- train_intensive_spice.py           Main training script (SPICE-in-the-loop)
|-- models/
|   |-- multi_topology_surrogate.py    Multi-topology surrogate (1.16M params)
|   +-- forward_surrogate.py           Base surrogate class
|-- rl/
|   |-- environment.py                 RL environment (CircuitDesignEnv)
|   |-- ppo_agent.py                   PPO agent (ActorCritic, pure PyTorch)
|   |-- spice_reward.py                ngspice integration + SPICE reward
|   +-- topology_rewards.py            Per-topology reward shaping
|-- checkpoints/
|   |-- multi_topology_surrogate.pt    Pre-trained surrogate (required)
|   +-- rl_agent_<topology>.pt         Trained RL agents (one per topology)
|-- data/
|   +-- generate_spice_data.py         SPICE data generation
+-- training/
    +-- train_surrogate.py             Surrogate model training
```

## Quick Start

```bash
# Prerequisites
brew install ngspice          # macOS
# or: sudo apt install ngspice  # Linux

# Install Python dependencies
pip install torch numpy scipy matplotlib tqdm

# Verify surrogate loads
python -c "
from models.multi_topology_surrogate import load_trained_model
m = load_trained_model(device='cpu')
print(f'Surrogate loaded: {sum(p.numel() for p in m.parameters()):,} params')
"

# Train a specific topology (edit TOPOLOGIES list in script)
python -u train_intensive_spice.py

# Or run the full pipeline from scratch
python data/generate_spice_data.py     # Generate SPICE training data
python training/train_surrogate.py     # Train surrogate model
python -u train_intensive_spice.py     # Train RL agents with SPICE validation
```

## Training Details

Each topology trains for hundreds of iterations. Each iteration collects thousands of environment steps using the surrogate, then validates against real ngspice every 5 iterations.

| Topology | Iterations | Steps/Iter | SPICE Calls | Training Time |
|----------|-----------|------------|-------------|---------------|
| Buck | 300 | 1024 | ~30K | ~4 hrs |
| Boost | 400 | 2048 | ~80K | ~12 hrs |
| Buck-Boost | 350 | 1024 | ~35K | ~6 hrs |
| SEPIC | 400 | 2048 | ~80K | ~12 hrs |
| Cuk | 350 | 1024 | ~35K | ~6 hrs |
| Flyback | 600 | 2048 | ~493K | ~80 hrs |
| QR Flyback | 700 | 4096 | ~576K+ | ~70+ hrs |

Total compute: 200+ hours of training, 1M+ SPICE simulations across all topologies.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, tqdm, Matplotlib
- ngspice (installed and on PATH)

## License

MIT License
