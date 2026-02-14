# Neural Surrogate + RL for Power Electronics Circuit Design

An ML system that designs DC-DC power converter circuits automatically. You give it a desired output voltage waveform, and a trained RL agent figures out the right component values (inductor, capacitor, switching frequency, duty cycle, etc.) to produce that waveform — in seconds instead of hours of manual tuning.

**All 7 converter topologies achieve SPICE-validated MSE < 5.0 (Excellent quality).**

---

## Results

Every topology is validated against real ngspice circuit simulation — not just the neural network. These are ground-truth results:

| Topology | SPICE MSE | Quality | Mode |
|----------|-----------|---------|------|
| Buck | 3.4 | Excellent | Surrogate |
| Boost | 3.5 | Excellent | Surrogate |
| Buck-Boost | 2.8 | Excellent | Formula |
| SEPIC | 3.9 | Excellent | Surrogate |
| Cuk | 3.2 | Excellent | Surrogate |
| Flyback | 0.2 | Excellent | Formula |
| QR Flyback | 2.8 | Excellent | Formula |


## The Problem

Power electronics engineers design circuits by trial and error: pick component values, run a SPICE simulation (~100ms), look at the output waveform, adjust, repeat. For complex topologies this takes hours of manual iteration.

This project automates that entire loop using three components working together:

## How It Works

### Step 1: Build a Fast Simulator (Neural Surrogate)

Real circuit simulation (ngspice) is accurate but slow -- about 100ms per run. To let an RL agent explore millions of designs, we first train a neural network to approximate the simulator.

- **Input**: 6 circuit parameters + which topology (buck, boost, flyback, etc.)
- **Output**: The predicted output voltage waveform (32 time-points, compressed from ~5000 SPICE points)
- **Speed**: ~0.001ms per prediction (100,000x faster than SPICE)
- **Training data**: 5,000+ real ngspice simulations per topology with randomized component values

The surrogate is a shared encoder with topology embeddings and a convolutional waveform decoder (555,619 parameters). It learns a unified representation across all 7 topologies.

> **Why 32 points?** Each SPICE simulation produces ~5000 data points over 5ms of circuit operation. The surrogate compresses this into 32 normalized features that capture the essential waveform shape — DC level, ripple, rise/fall, switching transients.

### Step 2: Train an RL Agent to Design Circuits

With the fast surrogate in place, we train a reinforcement learning agent that learns to pick the right component values for any target waveform.

**The setup works like a game:**

1. The agent is shown a target waveform (the voltage shape we want the circuit to produce)
2. It starts with random component values
3. Each "turn", it adjusts the 6 circuit parameters by small amounts
4. After each adjustment, the surrogate instantly predicts what the circuit would output
5. The agent gets a reward based on how close its waveform is to the target
6. Over millions of turns, it learns which parameter changes improve the waveform

### RL Policy Details

#### State Space (109 dimensions)

The state is a concatenation of enriched signal groups with explicit difference features:

| Group | Dims | Content |
|-------|------|---------|
| Target waveform features | 32 | Compressed waveform from surrogate or formula |
| Current waveform features | 32 | Agent's current predicted waveform |
| Pointwise difference | 32 | `target - current` at each of 32 time-points |
| Normalized parameters | 6 | Current [L, C, R_load, V_in, f_sw, duty] mapped to [0, 1] — log-scale for L, C, f_sw; linear for the rest |
| Error signals | 7 | Current MSE, previous MSE, MSE improvement, step fraction, best MSE so far, mean absolute error, max absolute error |

#### Action Space (6 continuous dimensions)

$$a = [\Delta L, \; \Delta C, \; \Delta R_{load}, \; \Delta V_{in}, \; \Delta f_{sw}, \; \Delta D] \in [-1, 1]^6$$

Actions are **relative adjustments** (not absolute values). Each is applied with a **max 20% change per step**:

- **L, C, f_sw** -- adjusted in **log-space** (these span orders of magnitude, e.g. 10uH to 100uH)
- **R_load, V_in, duty** -- adjusted **linearly**, then clipped to physical bounds

#### Network Architecture (ActorCritic)

```
State (109-dim)
    |
    v
[Shared Backbone]
  Linear(109, H) -> LayerNorm -> ReLU -> Linear(H, H) -> LayerNorm -> ReLU
    |                                           |
    v                                           v
[Actor Head]                              [Critic Head]
  Linear(H, H/2) -> ReLU                   Linear(H, H/2) -> ReLU
  Linear(H/2, 6) -> Tanh                   Linear(H/2, 1)
  -> action mean (mu)                       -> state value V(s)
  + learnable log(sigma) per dim
```

Hidden dim H varies per topology (e.g. 256 for simple topologies, **512** for qr_flyback).

#### Action Sampling

At each step, the policy samples from a diagonal Gaussian:

$$a \sim \mathcal{N}(\mu(s), \; \text{diag}(\sigma^2))$$

where $\mu(s)$ is the actor output and $\sigma = e^{\log \sigma}$ are the learnable per-dimension standard deviations. At inference time, $a = \mu(s)$ deterministically.

#### PPO Training Objective

The policy is updated using PPO's clipped surrogate objective:

$$\mathcal{L} = \underbrace{-\min\!\left(r_t \hat{A}_t, \; \text{clip}(r_t, \, 1 \pm \epsilon) \, \hat{A}_t\right)}_{\text{policy loss}} + \underbrace{0.5 \cdot \text{MSE}(V(s), \, G_t)}_{\text{value loss}} + \underbrace{c_{\text{ent}} \cdot (-H[\pi])}_{\text{entropy regularization}}$$

where $r_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the importance sampling ratio, $\hat{A}_t$ are GAE advantages (normalized to zero mean, unit variance), and gradients are clipped at max norm 0.5.

No external RL libraries -- pure PyTorch implementation.

#### Topology-Specific Hyperparameters

Harder topologies get larger networks, lower learning rates, and more exploration:

| Hyperparameter | Typical (Buck) | QR Flyback |
|----------------|---------------|------------|
| Hidden dim H | 256 | **512** |
| Learning rate | 3e-4 | **5e-5** (very low -- sensitive dynamics) |
| Clip epsilon | 0.20 | **0.10** (tighter -- careful updates) |
| Entropy coeff | 0.01 | **0.03** (more exploration) |
| Gamma | 0.99 | **0.998** (long-horizon for resonant transients) |
| GAE lambda | 0.95 | **0.98** |
| Steps/iter | 2,048 | **4,096** (larger batches for stability) |
| Mini-batch | 64 | 64 |
| Epochs/update | 10 | 10 |
| Grad clip | 0.5 | 0.5 |

### Step 3: Ground It in Reality (SPICE-in-the-Loop)

The surrogate is fast but imperfect. To prevent the agent from exploiting surrogate errors and learning "tricks" that wouldn't work on a real circuit, we periodically validate with real ngspice simulations:

- Every 5 training iterations, we run the agent's current design through actual ngspice
- The reward is blended: 70% real SPICE result + 30% surrogate prediction
- This keeps the agent honest — it can't learn shortcuts that only fool the neural network
- Over a full training run, 268K+ real SPICE simulations are used alongside millions of surrogate calls

### Formula Mode (Dual-Mode Architecture)

The neural surrogate works well for 4 topologies, but for 3 others (buck-boost, flyback, QR flyback) the surrogate's predictions aren't consistent enough for the RL agent to learn from. Instead of forcing more data through the neural network, we bypass it entirely for those topologies using known physics equations:

| Mode | Topologies | Fast Prediction Source |
|------|-----------|----------------------|
| **Surrogate** | Buck, Boost, SEPIC, Cuk | Neural network (32-point waveform) |
| **Formula** | Buck-Boost, Flyback, QR Flyback | Analytical $V_{out} = f(V_{in}, D, \eta)$ (512-point waveform) |

Both modes still validate against real SPICE simulation — the only difference is what provides the fast prediction between SPICE calls. Formula mode uses the known DC transfer function (e.g., $V_{out} = -V_{in} \cdot D / (1-D) \cdot \eta$ for buck-boost) with an efficiency factor $\eta \sim 0.85$ and synthesized ripple.


### The Reward Function

The agent doesn't just minimize prediction error. The reward captures what a power electronics engineer actually cares about:

$$R = -w_1 \cdot \text{MSE} - w_2 \cdot \text{THD} - w_3 \cdot \text{RiseErr} - w_4 \cdot \text{RippleErr} - w_5 \cdot \text{Overshoot} - w_6 \cdot \text{DC\_Err} + \text{bonuses}$$

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

| Topology | Iterations | Steps/Iter | SPICE Freq | Wall Time |
|----------|-----------|------------|------------|----------|
| Buck | 300 | 1,024 | 20% episodes | ~4 hrs |
| Boost | 400 | 2,048 | 20% episodes | ~4 hrs |
| Buck-Boost | 350 | 1,024 | 50% episodes | ~3 hrs |
| SEPIC | 400 | 2,048 | 20% episodes | ~4 hrs |
| Cuk | 350 | 1,024 | 20% episodes | ~3 hrs |
| Flyback | 600 | 2,048 | 50% episodes | ~6 hrs |
| QR Flyback | 700 | 4,096 | 50% episodes | ~6 hrs |

**Total: ~29.5 hours of training, 268K+ real SPICE simulations across all topologies.** All training ran on a MacBook Air (M3, 8GB, CPU-only). Formula mode topologies use 50% SPICE episode rate for more ground-truth feedback.

## Project Structure

```
MLEntry/
|-- train_intensive_spice.py           Main training script (dual-mode, SPICE-in-the-loop)
|-- generate_linkedin_visuals.py       Demo visualization generator
|-- models/
|   |-- multi_topology_surrogate.py    Surrogate model (555K params)
|   +-- forward_surrogate.py           Base surrogate class
|-- rl/
|   |-- environment.py                 RL environment (109-dim state, 6-dim action)
|   |-- ppo_agent.py                   PPO agent (ActorCritic, pure PyTorch)
|   |-- spice_reward.py                ngspice subprocess integration + SPICE netlists
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

MIT License — see [LICENSE](LICENSE) for details.
