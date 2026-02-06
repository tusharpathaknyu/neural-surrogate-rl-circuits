#!/usr/bin/env python3
"""
RESUME TRAINING FROM CRASH
==========================
Continues from flyback (which was at 454/600 iterations)
then completes qr_flyback.

Flyback checkpoint exists at checkpoints/rl_agent_flyback.pt
We'll continue training it and then do qr_flyback.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import time
import os
import signal

from models.multi_topology_surrogate import load_trained_model
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent
from rl.topology_rewards import compute_topology_aware_reward, TOPOLOGY_REWARD_CONFIG

# Import from intensive training script
from train_intensive_spice import (
    TOPOLOGY_CONFIG, 
    TopologySpecificEnv,
    train_topology_agent,
    test_topology_agent,
    DEVICE
)

# Topologies to train (resume from flyback)
RESUME_TOPOLOGIES = ['flyback', 'qr_flyback']


def continue_flyback_training(surrogate, start_iteration=454):
    """
    Continue flyback training from where it crashed.
    Load existing checkpoint and continue training.
    """
    topology = 'flyback'
    config = TOPOLOGY_CONFIG[topology]
    remaining_iters = config['n_iterations'] - start_iteration
    
    print(f"\n{'='*70}")
    print(f"RESUMING FLYBACK TRAINING")
    print(f"{'='*70}")
    print(f"  Starting from iteration: {start_iteration}")
    print(f"  Remaining iterations: {remaining_iters}")
    print(f"  Steps per iteration: {config['steps_per_iter']}")
    
    # Create environment
    env = TopologySpecificEnv(
        surrogate, topology, device=DEVICE,
        use_spice=True, spice_freq=config['spice_freq']
    )
    
    # Create agent with same architecture
    agent = PPOAgent(
        env, 
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        entropy_coef=config['entropy_coef'],
        device=DEVICE
    )
    
    # Load existing checkpoint
    checkpoint_path = f'checkpoints/rl_agent_{topology}.pt'
    if Path(checkpoint_path).exists():
        agent.load(checkpoint_path)
        print(f"  ✓ Loaded checkpoint from iteration ~{start_iteration}")
    else:
        print(f"  ⚠ No checkpoint found, starting fresh")
        start_iteration = 0
        remaining_iters = config['n_iterations']
    
    # Continue training
    best_reward = float('-inf')
    best_mse = float('inf')
    all_rewards = []
    all_mses = []
    spice_validations = 0
    
    start_time = time.time()
    pbar = tqdm(range(remaining_iters), desc=f'{topology} (resumed)')
    
    for iteration in pbar:
        actual_iter = start_iteration + iteration
        
        # Collect rollouts using agent's method (proper API)
        rollout = agent.collect_rollouts(config['steps_per_iter'])
        
        # Update policy with rollout dict
        agent.update(rollout, n_epochs=10, batch_size=64)
        
        # Track performance
        if len(agent.episode_rewards) > 0:
            all_rewards.extend(agent.episode_rewards[-10:])
            episode_reward = agent.episode_rewards[-1] if agent.episode_rewards else 0
        else:
            episode_reward = 0
            
        if len(agent.episode_mses) > 0:
            all_mses.extend(agent.episode_mses[-10:])
            episode_mse = agent.episode_mses[-1] if agent.episode_mses else float('inf')
        else:
            episode_mse = float('inf')
        
        # SPICE validation iteration
        if (actual_iter + 1) % config['spice_freq'] == 0:
            spice_validations += 1
            
            # Test with multiple SPICE runs
            spice_mses = []
            for _ in range(5):
                state = env.reset()
                for step in range(20):
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        action, _, _ = agent.policy.get_action(state_t, deterministic=True)
                    action_np = action.cpu().numpy().squeeze()
                    state, _, done, info = env.step(action_np)
                    if info.get('spice_validated', False):
                        spice_mses.append(info['mse'])
                    if done:
                        break
            
            if spice_mses:
                spice_mse = np.mean(spice_mses)
                pbar.set_postfix({
                    'spice_mse': f'{spice_mse:.1f}',
                    'best': f'{best_mse:.1f}',
                    'iter': f'{actual_iter + 1}/{config["n_iterations"]}'
                })
        
        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward
        if episode_mse < best_mse:
            best_mse = episode_mse
            agent.save(checkpoint_path)
        
        # Periodic update
        if (actual_iter + 1) % 5 != 0:
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'mse': f'{episode_mse:.1f}',
                'iter': f'{actual_iter + 1}/{config["n_iterations"]}'
            })
        
        # Save periodically
        if (actual_iter + 1) % 30 == 0:
            agent.save(checkpoint_path)
            print(f"\nSaved checkpoint at iteration {actual_iter + 1}")
    
    # Final save
    agent.save(checkpoint_path)
    
    train_time = time.time() - start_time
    
    result = {
        'topology': topology,
        'best_reward': best_reward,
        'best_mse': best_mse,
        'total_steps': remaining_iters * config['steps_per_iter'],
        'training_time_sec': train_time,
        'training_time_str': f'{train_time / 60:.1f} min',
        'resumed_from': start_iteration,
        'spice_validations': spice_validations,
    }
    
    print(f"\n  ✓ Completed flyback!")
    print(f"    Best MSE: {best_mse:.2f}")
    print(f"    Training time: {train_time / 60:.1f} min")
    
    return result


def main():
    """Resume training from crash."""
    
    print("\n" + "="*70)
    print("RESUMING TRAINING FROM CRASH")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    
    # Load existing progress
    progress_path = 'checkpoints/training_progress.json'
    if Path(progress_path).exists():
        with open(progress_path) as f:
            progress = json.load(f)
        completed = progress.get('completed', [])
        print(f"Previously completed: {', '.join(completed)}")
    else:
        completed = []
        progress = {'completed': [], 'results': {}}
    
    # Load surrogate
    print("\nLoading multi-topology surrogate...")
    surrogate = load_trained_model(device=DEVICE)
    print(f"  ✓ Loaded surrogate: {sum(p.numel() for p in surrogate.parameters()):,} params")
    
    training_results = progress.get('results', {})
    overall_start = time.time()
    
    # 1. Continue flyback training (was at ~454/600)
    if 'flyback' not in completed:
        flyback_result = continue_flyback_training(surrogate, start_iteration=454)
        training_results['flyback'] = flyback_result
        completed.append('flyback')
        
        # Save progress
        progress['completed'] = completed
        progress['results'] = training_results
        progress['timestamp'] = datetime.now().isoformat()
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
    else:
        print("\nFlyback already completed, skipping...")
    
    # 2. Train qr_flyback from scratch
    if 'qr_flyback' not in completed:
        print(f"\n[2/2] Training qr_flyback (full training)...")
        config = TOPOLOGY_CONFIG['qr_flyback']
        result = train_topology_agent('qr_flyback', surrogate, config)
        training_results['qr_flyback'] = result
        completed.append('qr_flyback')
        
        # Save progress
        progress['completed'] = completed
        progress['results'] = training_results
        progress['timestamp'] = datetime.now().isoformat()
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
    else:
        print("\nqr_flyback already completed, skipping...")
    
    # Test the resumed/trained agents
    print("\n" + "="*70)
    print("TESTING FLYBACK AND QR_FLYBACK")
    print("="*70)
    
    for topology in RESUME_TOPOLOGIES:
        result = test_topology_agent(topology, surrogate)
        if 'error' not in result:
            print(f"  {topology:12s}: MSE = {result['surrogate_mse']:7.1f} ± {result['surrogate_std']:5.1f}")
            if 'spice_mse' in result:
                print(f"                SPICE MSE = {result['spice_mse']:.1f}")
        else:
            print(f"  {topology:12s}: {result['error']}")
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*70)
    print("RESUME TRAINING COMPLETE")
    print("="*70)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  All 7 topologies should now be trained")
    
    # Print final summary
    print("\nFinal Training Summary (all topologies):")
    for topo in ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']:
        if topo in training_results:
            mse = training_results[topo].get('best_mse', 'N/A')
            print(f"  {topo:12s}: Best MSE = {mse}")


if __name__ == '__main__':
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Progress saved.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    main()
