from train_intensive_spice import TopologySpecificEnv
from models.multi_topology_surrogate import load_trained_model
import numpy as np

surrogate = load_trained_model(device='cpu')
env = TopologySpecificEnv(surrogate, 'buck', device='cpu', use_spice=True, spice_freq=5)

print('Debug:')
print(f'  use_spice_reward: {env.use_spice_reward}')
print(f'  spice_calculator: {env.spice_calculator}')
print(f'  spice_validation_freq: {env.spice_validation_freq}')

# Test SPICE directly
result = env._run_spice_simulation()
print(f'  Direct SPICE test: {result is not None}')
if result is not None:
    print(f'  Result shape: {result.shape}')

state = env.reset()
for i in range(12):
    action = np.random.randn(6) * 0.1
    state, reward, done, info = env.step(action)
    if info.get('spice_validated'):
        print(f'Step {i+1}: SPICE validated!')
    if done:
        break

print(f'Total SPICE calls: {env.spice_call_count}')
