#!/usr/bin/env python
"""Test the trained multi-topology RL agent."""

import torch
import numpy as np
from models.multi_topology_surrogate import load_trained_model
from rl.ppo_agent import PPOAgent
from rl.train_multi_topology_agent import MultiTopologyEnv, TOPOLOGIES

def main():
    # Load surrogate
    surrogate = load_trained_model(device='cpu')
    
    # Create environment
    env = MultiTopologyEnv(surrogate, device='cpu')
    
    # Create agent with correct hidden_dim (512 for extended training)
    agent = PPOAgent(env, hidden_dim=512, device='cpu')
    agent.load('checkpoints/multi_topo_rl_agent.pt')
    print('✓ Loaded trained agent (hidden=512)')
    
    # Test on all topologies
    print()
    print('Testing on All Topologies:')
    print('-' * 50)
    
    results = {}
    for topo in TOPOLOGIES:
        mses = []
        for _ in range(10):
            state = env.reset(topology=topo)
            for step in range(50):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = agent.policy.get_action(state_tensor, deterministic=True)
                action_np = action.numpy().squeeze()
                state, _, done, info = env.step(action_np)
                if done:
                    break
            mses.append(info['mse'])
        
        mean_mse = np.mean(mses)
        std_mse = np.std(mses)
        rmse = np.sqrt(mean_mse)
        results[topo] = {'mse': mean_mse, 'std': std_mse, 'rmse': rmse}
        print(f'{topo:12s}: MSE = {mean_mse:7.1f} ± {std_mse:6.1f}  (RMSE = {rmse:.1f}V)')
    
    # Summary
    print()
    print('=' * 50)
    avg_mse = np.mean([r['mse'] for r in results.values()])
    print(f'Average MSE across all topologies: {avg_mse:.1f}')
    print(f'Average RMSE: {np.sqrt(avg_mse):.1f}V')

if __name__ == '__main__':
    main()
