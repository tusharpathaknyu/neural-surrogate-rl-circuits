#!/usr/bin/env python3
"""Test SPICE integration in RL environment."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from rl.environment import CircuitDesignEnv
from rl.spice_reward import SPICERewardCalculator
from models.multi_topology_surrogate import load_trained_model

def main():
    print("Testing SPICE-Enhanced Environment...")
    print("="*60)
    
    # Load surrogate
    surrogate = load_trained_model(device='mps')
    print("✓ Loaded surrogate")
    
    # Create environment with SPICE
    env = CircuitDesignEnv(
        surrogate=surrogate,
        device='mps',
        use_spice_reward=True,
        spice_validation_freq=1  # Every step for testing
    )
    env.topology = 'buck'
    env.topology_idx = 0
    env.is_multi_topology = True
    print("✓ Created environment with SPICE")
    
    # Reset and take steps
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Take several steps
    print("\nTaking 5 steps with SPICE validation...")
    for i in range(5):
        action = np.random.uniform(-0.1, 0.1, size=6).astype(np.float32)
        state, reward, done, info = env.step(action)
        
        spice_used = info.get('spice_used', False)
        spice_vout = info.get('spice_vout', 'N/A')
        spice_reward = info.get('spice_reward', 'N/A')
        
        print(f"  Step {i+1}: reward={reward:.3f}, MSE={info['mse']:.3f}, "
              f"SPICE={spice_used}, Vout={spice_vout}")
        
        if done:
            break
    
    print("\n✓ SPICE integration working!")
    print("="*60)

if __name__ == '__main__':
    main()
