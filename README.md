# Neural Surrogate + RL for Power Electronics Design

A machine learning system that accelerates power electronics circuit design by **100,000x** using neural surrogates and reinforcement learning.

## What This Does

Instead of running slow SPICE simulations (100ms each), we:
1. **Train a neural surrogate** that predicts circuit behavior in 0.001ms
2. **Use RL to explore** millions of designs using the fast surrogate
3. **Design circuits instantly** that would take hours with traditional methods

## Results

| Metric | Traditional SPICE | Our Approach | Speedup |
|--------|------------------|--------------|---------|
| Single simulation | 100ms | 0.001ms | **100,000x** |
| Design optimization | Hours | Seconds | **10,000x** |
| Surrogate MSE | - | 0.007 | - |
| RL Design MSE | - | 1.28 | - |

## Quick Start

```bash
# Install dependencies
brew install ngspice
pip install torch numpy scipy matplotlib tqdm

# Run complete pipeline
python run_all_phases.py

# Or run phases individually:
python data/generate_spice_data.py   # Phase A: SPICE data
python training/train_surrogate.py   # Phase B: Train surrogate
python rl/train_agent.py             # Phase C: Train RL agent
```

## The Three Phases

### Phase A: SPICE Data Harvesting
- Runs 5000 ngspice simulations with randomized component values
- Buck converter: L, C, R_load, V_in, f_sw, duty cycle
- Captures output waveforms (512 points each)

### Phase B: Train Neural Surrogate
- 1D-CNN maps 6 parameters to 512-point waveform
- Multi-component loss: MSE + Spectral + THD + Ripple
- Achieves 0.007 MSE on validation

### Phase C: Train RL Agent
- PPO with Actor-Critic (pure PyTorch, no gymnasium)
- Uses surrogate for microsecond simulations
- 500K steps in ~25 minutes

## Circuit Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| L | 10-100 uH | Inductor |
| C | 47-470 uF | Capacitor |
| R_load | 2-50 Ohm | Load resistance |
| V_in | 10-24 V | Input voltage |
| f_sw | 50-500 kHz | Switching frequency |
| duty | 20-80% | Duty cycle |

## Development Log

### Session 1 (Jan 23, 2026)
- Set up project structure
- Implemented SPICE data generation (buck converter)
- Built forward surrogate model (1D-CNN)
- Created power electronics loss functions
- Implemented PPO agent (pure PyTorch)
- Trained on 5000 samples + 500K RL steps
- Achieved 0.007 surrogate MSE, 1.28 design MSE

## License

MIT License
