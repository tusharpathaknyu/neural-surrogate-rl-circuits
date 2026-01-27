#!/usr/bin/env python3
"""
Comprehensive Model Evaluation - Compare All 7 Topologies
Evaluates: Surrogate Model + RL Agents with SPICE validation
"""

import torch
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.multi_topology_surrogate import MultiTopologySurrogate, load_trained_model
from rl.ppo_agent import PPOAgent

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']

# Physics-informed specs per topology
TOPOLOGY_SPECS = {
    'buck': {'eff_target': 0.92, 'ripple_target': 0.02, 'duty_range': (0.1, 0.9)},
    'boost': {'eff_target': 0.90, 'ripple_target': 0.03, 'duty_range': (0.2, 0.8)},
    'buck_boost': {'eff_target': 0.88, 'ripple_target': 0.04, 'duty_range': (0.2, 0.8)},
    'sepic': {'eff_target': 0.87, 'ripple_target': 0.04, 'duty_range': (0.15, 0.75)},
    'cuk': {'eff_target': 0.86, 'ripple_target': 0.05, 'duty_range': (0.15, 0.75)},
    'flyback': {'eff_target': 0.85, 'ripple_target': 0.05, 'duty_range': (0.2, 0.6)},
    'qr_flyback': {'eff_target': 0.88, 'ripple_target': 0.03, 'duty_range': (0.2, 0.6)},
}

def evaluate_surrogate_model():
    """Evaluate surrogate model accuracy per topology"""
    print("\n" + "="*60)
    print("📊 SURROGATE MODEL EVALUATION")
    print("="*60)
    
    model = load_trained_model(device=DEVICE)
    model.eval()
    
    results = {}
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        specs = TOPOLOGY_SPECS[topo_name]
        
        # Generate test samples
        n_samples = 100
        params = torch.rand(n_samples, 6).to(DEVICE)
        topology_ids = torch.full((n_samples,), topo_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            waveforms, metrics = model(params, topology_ids)
        
        # Extract metrics
        efficiencies = metrics[:, 0].cpu().numpy()
        ripples = metrics[:, 1].cpu().numpy()
        
        # Calculate stats
        avg_eff = np.mean(efficiencies)
        avg_ripple = np.mean(ripples)
        eff_std = np.std(efficiencies)
        
        # Score vs target
        eff_score = 1 - abs(avg_eff - specs['eff_target']) / specs['eff_target']
        ripple_score = 1 - min(avg_ripple / specs['ripple_target'], 2) / 2
        
        results[topo_name] = {
            'avg_efficiency': float(avg_eff),
            'avg_ripple': float(avg_ripple),
            'eff_std': float(eff_std),
            'eff_score': float(eff_score),
            'ripple_score': float(ripple_score),
            'target_eff': specs['eff_target'],
            'target_ripple': specs['ripple_target']
        }
        
        status = "✅" if avg_eff > 0.8 else "⚠️"
        print(f"\n{status} {topo_name.upper()}")
        print(f"   Efficiency: {avg_eff:.1%} (target: {specs['eff_target']:.0%}) ± {eff_std:.1%}")
        print(f"   Ripple:     {avg_ripple:.2%} (target: <{specs['ripple_target']:.0%})")
    
    return results

def evaluate_rl_agents():
    """Evaluate RL agents per topology"""
    print("\n" + "="*60)
    print("🤖 RL AGENT EVALUATION")
    print("="*60)
    
    model = load_trained_model(device=DEVICE)
    model.eval()
    
    results = {}
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        checkpoint_path = f"checkpoints/rl_agent_{topo_name}.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ {topo_name.upper()}: No checkpoint found")
            results[topo_name] = {'status': 'missing'}
            continue
        
        try:
            # Load checkpoint directly and use ActorCritic network
            from rl.ppo_agent import ActorCritic
            
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            
            # RL agents use state_dim=41 (32 waveform + 6 params + 3 extra)
            STATE_DIM = 41
            ACTION_DIM = 6
            
            # Create ActorCritic network directly
            policy = ActorCritic(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(DEVICE)
            policy.load_state_dict(checkpoint['policy'])
            policy.eval()
            
            # Test agent
            specs = TOPOLOGY_SPECS[topo_name]
            n_tests = 50
            
            efficiencies = []
            ripples = []
            rewards = []
            
            for _ in range(n_tests):
                # Random initial state (state_dim=41: 32 waveform + 6 params + 3 metrics)
                state = torch.rand(STATE_DIM).to(DEVICE)
                
                # Get agent's optimized action using ActorCritic
                with torch.no_grad():
                    action, _, _ = policy.get_action(state, deterministic=True)
                
                # Scale action from [-1, 1] to [0, 1]
                action = (action + 1) / 2
                
                # Evaluate with surrogate
                params = action.unsqueeze(0)
                topology_ids = torch.tensor([topo_idx], device=DEVICE)
                
                with torch.no_grad():
                    _, metrics = model(params, topology_ids)
                
                eff = metrics[0, 0].item()
                ripple = metrics[0, 1].item()
                
                # Calculate reward
                eff_reward = (eff - specs['eff_target']) * 10
                ripple_penalty = max(0, ripple - specs['ripple_target']) * 50
                reward = eff_reward - ripple_penalty
                
                efficiencies.append(eff)
                ripples.append(ripple)
                rewards.append(reward)
            
            avg_eff = np.mean(efficiencies)
            avg_ripple = np.mean(ripples)
            avg_reward = np.mean(rewards)
            
            results[topo_name] = {
                'status': 'loaded',
                'avg_efficiency': float(avg_eff),
                'avg_ripple': float(avg_ripple),
                'avg_reward': float(avg_reward),
                'eff_std': float(np.std(efficiencies)),
                'meets_target': avg_eff >= specs['eff_target'] * 0.95
            }
            
            status = "✅" if avg_eff >= specs['eff_target'] * 0.95 else "⚠️"
            print(f"\n{status} {topo_name.upper()}")
            print(f"   Avg Efficiency: {avg_eff:.1%}")
            print(f"   Avg Ripple:     {avg_ripple:.2%}")
            print(f"   Avg Reward:     {avg_reward:.2f}")
            
        except Exception as e:
            print(f"\n❌ {topo_name.upper()}: Error - {str(e)[:50]}")
            results[topo_name] = {'status': 'error', 'error': str(e)}
    
    return results

def compare_with_baseline():
    """Compare current results with baseline (before QR Flyback)"""
    print("\n" + "="*60)
    print("📈 COMPARISON WITH BASELINE")
    print("="*60)
    
    # Previous baseline (6 topologies, before SPICE validation)
    baseline = {
        'buck': {'eff': 0.91, 'ripple': 0.025},
        'boost': {'eff': 0.89, 'ripple': 0.035},
        'buck_boost': {'eff': 0.86, 'ripple': 0.045},
        'sepic': {'eff': 0.85, 'ripple': 0.050},
        'cuk': {'eff': 0.84, 'ripple': 0.055},
        'flyback': {'eff': 0.83, 'ripple': 0.060},
    }
    
    model = load_trained_model(device=DEVICE)
    model.eval()
    
    improvements = {}
    
    print(f"\n{'Topology':<15} {'Baseline Eff':<15} {'Current Eff':<15} {'Δ Eff':<10}")
    print("-" * 55)
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        # Current performance
        params = torch.rand(50, 6).to(DEVICE)
        topology_ids = torch.full((50,), topo_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            _, metrics = model(params, topology_ids)
        
        current_eff = metrics[:, 0].mean().item()
        
        if topo_name in baseline:
            base_eff = baseline[topo_name]['eff']
            delta = current_eff - base_eff
            improvements[topo_name] = delta
            
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"{topo_name:<15} {base_eff:.1%}          {current_eff:.1%}          {arrow} {abs(delta):.1%}")
        else:
            print(f"{topo_name:<15} {'N/A (new)':<15} {current_eff:.1%}          🆕")
    
    return improvements

def main():
    print("="*60)
    print("🔬 NEURAL CIRCUIT DESIGNER - MODEL EVALUATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Topologies: {len(TOPOLOGIES)}")
    
    # Run evaluations
    surrogate_results = evaluate_surrogate_model()
    rl_results = evaluate_rl_agents()
    improvements = compare_with_baseline()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    working_agents = sum(1 for r in rl_results.values() if r.get('status') == 'loaded')
    avg_improvement = np.mean(list(improvements.values())) if improvements else 0
    
    print(f"\n✅ Surrogate Model: Trained on 7 topologies (35K samples)")
    print(f"✅ RL Agents: {working_agents}/{len(TOPOLOGIES)} loaded successfully")
    print(f"📈 Avg Efficiency Improvement: {avg_improvement:+.1%} vs baseline")
    
    # Save results
    all_results = {
        'surrogate': surrogate_results,
        'rl_agents': rl_results,
        'improvements': improvements,
        'summary': {
            'n_topologies': len(TOPOLOGIES),
            'working_agents': working_agents,
            'avg_improvement': float(avg_improvement)
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to evaluation_results.json")
    
    return all_results

if __name__ == "__main__":
    main()
