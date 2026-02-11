# Neural Surrogate + RL for Power Electronics Circuit Design

An ML system that learns to design DC-DC power converter circuits automatically. Given a desired output voltage waveform, a trained RL agent selects component values (inductor, capacitor, switching frequency, duty cycle, etc.) that produce that waveform -- in seconds instead of hours.

---

## The Problem

Power electronics engineers design circuits by trial and error: pick component values, run a SPICE simulation (~100ms), look at the output waveform, adjust, repeat. For complex topologies this takes hours of manual iteration.

This project automates that entire loop using three components working together:

## How It Works

### Step 1: Build a Fast Simulator (Neural Surrogate)

Real circuit simulation (ngspice) is accurate but slow -- about 100ms per run. To let an RL agent explore millions of designs, we first train a neural network to approximate the simulator.

- **Input**: 6 circuit parameters + which topology (buck, boost, flyback, etc.)
- **Output**: The predicted output voltage waveform (512 time-points)
- **Speed**: ~0.001ms per prediction (100,000x faster than SPICE)
- **Training data**: 5,000+ real ngspice simulations per topology with randomized component values

The surrogate is a shared encoder with topology embeddings and a convolutional waveform decoder (1.16M parameters total). It learns a unified representation across all 7 topologies.

### Step 2: Train an RL Agent to Design Circuits

With the fast surrogate in place, we train a reinforcement learning agent that learns to pick the right component values for any target waveform.

**The setup works like a game:**

1. The agent is shown a target waveform (the voltage shape we want the circuit to produce)
2. It starts with random component values
3. Each "turn", it adjusts the 6 circuit parameters by small amounts
4. After each adjustment, the surrogate instantly predicts what the circuit would output
5. The agent gets a reward based on how close its waveform is to the target
6. Over millions of turns, it learns which parameter changes improve the waveform

**What the agent sees (state, 41 dimensions):**
- Features extracted from the target waveform (what we want) -- 32 values capturing shape, frequency content, and segment-level detail
- The current circuit parameter values (normalized) -- 6 values
- Error signals (current MSE, progress through episode, previous MSE) -- 3 values

**What the agent does (action, 6 dimensions):**
- Outputs a small adjustment to each circuit parameter: inductor (L), capacitor (C), load resistance (R), input voltage (V_in), switching frequency (f_sw), and duty cycle (D)
- Each adjustment is bounded to +/-20% of the parameter's range per step
- Parameters that span large ranges (L, C, f_sw) are adjusted on a log scale so the agent can make fine adjustments at any order of magnitude

**How it learns (PPO -- Proximal Policy Optimization):**
- The agent's brain is an actor-critic neural network written from scratch in PyTorch (no external RL libraries)
- The "actor" decides what adjustments to make (outputs a Gaussian distribution over actions)
- The "critic" estimates how good the current state is (used to compute advantages)
- Both share a backbone network, then split into separate heads
- PPO's clipped objective prevents the policy from changing too drastically in a single update, which stabilizes training
- Each topology gets its own hyperparameters -- harder topologies (flyback, QR flyback) use larger networks, lower learning rates, and more exploration

### Step 3: Ground It in Reality (SPICE-in-the-Loop)

The surrogate is fast but imperfect. To prevent the agent from exploiting surrogate errors and learning "tricks" that wouldn't work on a real circuit, we periodically validate with real ngspice simulations:

- Every 5 training iterations, we run the agent's current design through actual ngspice
- The reward is blended: 70% real SPICE result + 30% surrogate prediction
- This keeps the agent honest -- it can't learn shortcuts that only fool the neural network
- Over a full training run, 500K-600K real SPICE simulations are used alongside millions of surrogate calls

### The Reward Function

The agent doesn't just minimize prediction error. The reward captures what a power electronics engineer actually cares about:

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| MSE | Overall waveform shape match | Primary learning signal |
| DC Error | Is the average output voltage correct? | A 12V supply must output 12V, not 11.5V |
| Overshoot | Does the voltage spike above target? | Can physically damage components in real circuits |
| Ripple | How much does the voltage wobble? | Too much ripple means noisy power delivery |
| Rise Time | How fast does the voltage reach its target? | Slow rise = sluggish response to load changes |
| THD | Total harmonic distortion | Measures unwanted frequency content in the output |

Each topology gets different weights on these metrics. For example:
- **Inverted topologies** (Buck-Boost, Cuk) get a penalty if the output voltage has the wrong sign
- **Flyback** gets a separate ringing penalty (high-frequency oscillation from the transformer)
- **QR Flyback** gets THD penalty reduced by 80% (harmonics are expected in resonant converters) and a bonus for smooth switching transitions

## Supported Topologies

Each topology is a different circuit architecture for converting DC voltage. They differ in complexity, component count, and behavior:

| Topology | What it does | Transfer Function | Difficulty |
|----------|-------------|-------------------|------------|
| Buck | Steps voltage down | V_out = V_in x D | Simplest |
| Boost | Steps voltage up | V_out = V_in / (1 - D) | Simple |
| Buck-Boost | Steps up or down, inverts polarity | V_out = -V_in x D / (1 - D) | Medium |
| SEPIC | Steps up or down, same polarity | V_out = V_in x D / (1 - D) | Medium |
| Cuk | Steps up or down, inverts, low ripple | V_out = -V_in x D / (1 - D) | Medium |
| Flyback | Isolated output via transformer | V_out = V_in x D x n / (1 - D) | Hard |
| QR Flyback | Flyback with resonant soft-switching | Modified duty with resonant tank | Hardest |

D = duty cycle (fraction of time the switch is ON). n = transformer turns ratio.

## Circuit Parameters the Agent Controls

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Inductance | L | 10-100 uH | Energy storage element |
| Capacitance | C | 47-470 uF | Output voltage smoothing |
| Load Resistance | R_load | 2-50 Ohm | How much current the load draws |
| Input Voltage | V_in | 10-24 V | DC source voltage |
| Switching Frequency | f_sw | 50-500 kHz | How fast the transistor switches |
| Duty Cycle | D | 20-80% | Fraction of each cycle the switch is ON |

## Training Compute

Each topology trains for hundreds of iterations. Each iteration, the agent collects thousands of environment steps using the fast surrogate, then validates against real ngspice.

| Topology | Iterations | Env Steps/Iter | SPICE Calls | Wall Time |
|----------|-----------|----------------|-------------|-----------|
| Buck | 300 | 1,024 | ~30K | ~4 hrs |
| Boost | 400 | 2,048 | ~80K | ~12 hrs |
| Buck-Boost | 350 | 1,024 | ~35K | ~6 hrs |
| SEPIC | 400 | 2,048 | ~80K | ~12 hrs |
| Cuk | 350 | 1,024 | ~35K | ~6 hrs |
| Flyback | 600 | 2,048 | ~493K | ~80 hrs |
| QR Flyback | 700 | 4,096 | ~576K+ | ~70+ hrs |

**Total: 200+ hours of training, 1M+ real SPICE simulations across all topologies.** All training ran on a MacBook Air (M3, CPU-only). SPICE simulation is CPU-bound and dominates the wall time.

## Project Structure

```
MLEntry/
|-- train_intensive_spice.py           Main training script (SPICE-in-the-loop)
|-- models/
|   |-- multi_topology_surrogate.py    Surrogate model (1.16M params)
|   +-- forward_surrogate.py           Base surrogate class
|-- rl/
|   |-- environment.py                 RL environment (state, action, step logic)
|   |-- ppo_agent.py                   PPO agent (ActorCritic, pure PyTorch)
|   |-- spice_reward.py                ngspice subprocess integration
|   +-- topology_rewards.py            Per-topology reward shaping
|-- checkpoints/
|   |-- multi_topology_surrogate.pt    Pre-trained surrogate (required)
|   +-- rl_agent_<topology>.pt         Trained RL agents (one per topology)
|-- data/
|   +-- generate_spice_data.py         SPICE data generation scripts
+-- training/
    +-- train_surrogate.py             Surrogate model training
```

## Quick Start

```bash
# Prerequisites
brew install ngspice            # macOS
# or: sudo apt install ngspice  # Linux

pip install torch numpy scipy matplotlib tqdm

# Train RL agents (edit TOPOLOGIES list in script to pick topologies)
python -u train_intensive_spice.py

# Or run the full pipeline from scratch:
python data/generate_spice_data.py     # 1. Generate SPICE training data
python training/train_surrogate.py     # 2. Train surrogate model
python -u train_intensive_spice.py     # 3. Train RL agents with SPICE validation
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, tqdm, Matplotlib
- ngspice (circuit simulator, installed and on PATH)

## License

MIT License
