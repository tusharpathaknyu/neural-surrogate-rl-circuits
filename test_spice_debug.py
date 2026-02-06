#!/usr/bin/env python3
"""Test SPICE integration"""
from train_intensive_spice import TopologySpecificEnv
from models.multi_topology_surrogate import load_trained_model
import numpy as np

surrogate = load_trained_model(device='cpu')
env = TopologySpecificEnv(surrogate, 'buck', device='cpu', use_spice=True, spice_freq=5)

print('Debug:')
print(f'  use_spice_reward: {env.use_spice_reward}')
print(f'  spice_calculator: {env.spice_calculator}')
print(f'  spice_validation_freq: {env.spice_validation_freq}')

# Initialize params first
env.current_params = np.array([50e-6, 100e-6, 10, 12, 100e3, 0.5])
env.target_waveform = np.ones(32) * 6  # Simple target

# Test SPICE directly
print('\nTesting direct SPICE call...')
result = env._run_spice_simulation()
print(f'  Direct SPICE result: {result}')
if result is not None:
    print(f'  Result shape: {result.shape}')

# Test via step
print('\nTesting via step():')
state = env.reset()
for i in range(12):
    action = np.random.randn(6) * 0.1
    state, reward, done, info = env.step(action)
    spice_flag = info.get('spice_validated', False)
    print(f'  Step {i+1}: spice_validated={spice_flag}, step_count={env.step_count}')
    if done:
        break

print(f'\nTotal SPICE calls: {env.spice_call_count}')
