---
title: Neural Circuit Designer
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Neural Circuit Designer

AI-powered DC-DC converter design system using reinforcement learning and neural surrogate models.

## Features
- **7 Converter Topologies**: Buck, Boost, Buck-Boost, SEPIC, Ćuk, Flyback, QR-Flyback
- **555K Parameter Neural Surrogate**: Trained on 35,000 SPICE simulations
- **RL-Optimized Design**: PPO agents with physics-informed hyperparameters
- **Real-time Waveform Prediction**: 10,000x faster than SPICE simulation

## Usage
1. Select a converter topology
2. Set input/output voltage requirements
3. Click "Design Circuit" to get optimized parameters
4. View predicted waveforms and efficiency metrics

## Tech Stack
- PyTorch for neural networks
- PPO (Proximal Policy Optimization) for RL
- Gradio for web interface
- SPICE validation during training
