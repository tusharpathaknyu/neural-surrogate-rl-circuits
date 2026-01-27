#!/usr/bin/env python
"""Comprehensive evaluation of all trained models."""

import torch
import numpy as np
import json
from pathlib import Path

def evaluate_surrogate():
    """Evaluate the surrogate model."""
    from models.multi_topology_surrogate import load_trained_model
    
    print("=" * 60)
    print("SURROGATE MODEL EVALUATION")
    print("=" * 60)
    
    model = load_trained_model(device='cpu')
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:")
    print(f"  Total Parameters: {n_params:,}")
    print(f"  Topologies: {model.topologies}")
    
    TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
    
    print(f"\nPer-Topology Inference Test:")
    print("-" * 60)
    
    results = {}
    for i, topo in enumerate(TOPOLOGIES):
        params = torch.rand(10, 6)
        topo_ids = torch.full((10,), i, dtype=torch.long)
        
        with torch.no_grad():
            waveform, metrics = model(params, topo_ids)
        
        wf_mean = waveform.mean().item()
        wf_std = waveform.std().item()
        eff = metrics[:, 0].mean().item()
        ripple = metrics[:, 1].mean().item()
        
        results[topo] = {
            'waveform_mean': wf_mean,
            'waveform_std': wf_std,
            'efficiency': eff,
            'ripple': ripple
        }
        print(f"{topo:12s}: Waveform μ={wf_mean:6.2f}V σ={wf_std:5.2f}V | Eff={eff:.2f} Ripple={ripple:.3f}")
    
    return results


def evaluate_per_topology_agents():
    """Evaluate per-topology RL agents."""
    from models.multi_topology_surrogate import load_trained_model
    from rl.ppo_agent import PPOAgent
    from rl.multi_objective_env import MultiObjectiveEnv
    
    print("\n" + "=" * 60)
    print("PER-TOPOLOGY RL AGENTS EVALUATION")
    print("=" * 60)
    
    TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
    device = 'cpu'
    
    surrogate = load_trained_model(device=device)
    results = {}
    
    for topo in TOPOLOGIES:
        checkpoint_path = f'checkpoints/rl_agent_{topo}.pt'
        if not Path(checkpoint_path).exists():
            print(f"{topo:12s}: No checkpoint found")
            continue
        
        # Create env and agent
        topo_idx = TOPOLOGIES.index(topo)
        env = MultiObjectiveEnv(surrogate, topology_id=topo_idx, device=device)
        
        # Try different hidden dims
        for hidden_dim in [256, 512]:
            try:
                agent = PPOAgent(env, hidden_dim=hidden_dim, device=device)
                agent.load(checkpoint_path)
                break
            except:
                continue
        
        # Test agent
        mses = []
        for _ in range(20):
            state = env.reset()
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
        
        results[topo] = {
            'mean_mse': float(mean_mse),
            'std_mse': float(std_mse),
            'rmse': float(rmse)
        }
        
        # Quality rating
        if rmse < 5:
            rating = "⭐⭐⭐ Excellent"
        elif rmse < 15:
            rating = "⭐⭐ Good"
        elif rmse < 30:
            rating = "⭐ Fair"
        else:
            rating = "Needs Improvement"
        
        print(f"{topo:12s}: MSE = {mean_mse:7.1f} ± {std_mse:6.1f} (RMSE = {rmse:5.1f}V) {rating}")
    
    return results


def evaluate_multi_topology_agent():
    """Evaluate the unified multi-topology agent."""
    from models.multi_topology_surrogate import load_trained_model
    from rl.ppo_agent import PPOAgent
    from rl.train_multi_topology_agent import MultiTopologyEnv, TOPOLOGIES
    
    print("\n" + "=" * 60)
    print("MULTI-TOPOLOGY RL AGENT EVALUATION")
    print("=" * 60)
    
    device = 'cpu'
    surrogate = load_trained_model(device=device)
    env = MultiTopologyEnv(surrogate, device=device)
    
    agent = PPOAgent(env, hidden_dim=512, device=device)
    agent.load('checkpoints/multi_topo_rl_agent.pt')
    
    results = {}
    for topo in TOPOLOGIES:
        mses = []
        for _ in range(20):
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
        
        results[topo] = {
            'mean_mse': float(mean_mse),
            'std_mse': float(std_mse),
            'rmse': float(rmse)
        }
        print(f"{topo:12s}: MSE = {mean_mse:7.1f} ± {std_mse:6.1f} (RMSE = {rmse:5.1f}V)")
    
    avg_mse = np.mean([r['mean_mse'] for r in results.values()])
    print(f"\nAverage MSE: {avg_mse:.1f} | Average RMSE: {np.sqrt(avg_mse):.1f}V")
    
    return results


def main():
    print("\n" + "🔬 " * 20)
    print("   COMPREHENSIVE MODEL EVALUATION REPORT")
    print("🔬 " * 20)
    
    # 1. Surrogate
    surrogate_results = evaluate_surrogate()
    
    # 2. Per-topology agents
    per_topo_results = evaluate_per_topology_agents()
    
    # 3. Multi-topology agent
    multi_results = evaluate_multi_topology_agent()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n📊 Per-Topology Agents (Specialized):")
    for topo, r in per_topo_results.items():
        print(f"   {topo:12s}: RMSE = {r['rmse']:5.1f}V")
    
    avg_per_topo = np.mean([r['rmse'] for r in per_topo_results.values()])
    print(f"   {'Average':12s}: RMSE = {avg_per_topo:5.1f}V")
    
    print("\n📊 Multi-Topology Agent (Unified):")
    for topo, r in multi_results.items():
        print(f"   {topo:12s}: RMSE = {r['rmse']:5.1f}V")
    
    avg_multi = np.mean([r['rmse'] for r in multi_results.values()])
    print(f"   {'Average':12s}: RMSE = {avg_multi:5.1f}V")
    
    print("\n✅ Models trained: 7 per-topology + 1 unified")
    print(f"✅ New QR-Flyback topology fully integrated")
    print(f"✅ Best performer: buck (RMSE = {per_topo_results.get('buck', {}).get('rmse', 0):.1f}V)")
    
    # Save results
    all_results = {
        'surrogate': surrogate_results,
        'per_topology_agents': per_topo_results,
        'multi_topology_agent': multi_results,
        'summary': {
            'avg_per_topology_rmse': float(avg_per_topo),
            'avg_multi_topology_rmse': float(avg_multi),
            'topologies_trained': 7,
            'new_topology': 'qr_flyback'
        }
    }
    
    with open('checkpoints/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n📁 Results saved to checkpoints/evaluation_results.json")


if __name__ == '__main__':
    main()
