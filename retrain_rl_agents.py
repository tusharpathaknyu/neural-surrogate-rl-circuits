#!/usr/bin/env python3
"""
Retrain RL Agents for Challenging Topologies
Uses enhanced surrogate and longer training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from rl.ppo_agent import ActorCritic

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

CHALLENGING = ['cuk', 'flyback', 'qr_flyback']
TOPO_IDS = {'cuk': 4, 'flyback': 5, 'qr_flyback': 6}


def load_enhanced_surrogate():
    """Load enhanced surrogate model."""
    from retrain_surrogate_enhanced import MultiTopologySurrogate
    
    model = MultiTopologySurrogate().to(DEVICE)
    checkpoint = torch.load(
        'checkpoints/multi_topology_surrogate_enhanced.pt',
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded enhanced surrogate (val_loss={checkpoint['val_loss']:.2f})")
    return model


def compute_reward(waveform: np.ndarray, target_waveform: np.ndarray) -> float:
    """Compute reward from waveform quality."""
    # MSE
    mse = np.mean((waveform - target_waveform) ** 2)
    
    # Stability (low variance is good)
    stability = 1 - min(np.std(waveform) / (np.abs(np.mean(waveform)) + 1e-6), 1)
    
    # Ripple (lower is better)
    ripple = (np.max(waveform) - np.min(waveform)) / (np.abs(np.mean(waveform)) + 1e-6)
    ripple_score = 1 - min(ripple, 1)
    
    # Combined reward
    reward = -mse * 10 + stability * 2 + ripple_score * 2
    
    return float(reward)


def generate_target_waveform(topology: str, params: np.ndarray) -> np.ndarray:
    """Generate idealized target waveform for topology."""
    t = np.linspace(0, 1, 512)
    
    # Topology-specific ideal waveforms
    duty = params[5] if len(params) > 5 else 0.5
    
    if topology == 'cuk':
        # Cuk: inverted, smooth output
        Vout = -12 * duty / (1 - duty)
        target = Vout * (1 + 0.02 * np.sin(2 * np.pi * 4 * t))
    elif topology == 'flyback':
        # Flyback: positive, slight ringing
        Vout = 12 * duty / (1 - duty)
        target = Vout * (1 + 0.03 * np.sin(2 * np.pi * 4 * t) * np.exp(-t * 3))
    else:  # qr_flyback
        # QR Flyback: smoother transitions
        Vout = 12 * duty / (1 - duty)
        target = Vout * (1 + 0.015 * np.sin(2 * np.pi * 4 * t))
    
    # Normalize
    target = target / (np.abs(target).max() + 1e-6)
    
    return target


def train_agent_for_topology(topology: str, surrogate, n_episodes: int = 500):
    """Train RL agent for specific topology."""
    print(f"\n🏋️ Training agent for {topology.upper()}...")
    
    topo_id = TOPO_IDS[topology]
    
    # Use larger network for complex topologies
    state_dim = 41  # 32 waveform + 6 params + 3 error
    action_dim = 6
    hidden_dim = 512  # Larger for complex topologies
    
    policy = ActorCritic(state_dim, action_dim, hidden_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    best_reward = -float('inf')
    rewards_history = []
    
    for episode in tqdm(range(n_episodes), desc=f"Training {topology}"):
        # Random initial parameters
        params = torch.rand(action_dim).to(DEVICE)
        
        # Create state
        # Simulate waveform features (32) + params (6) + error (3)
        waveform_features = torch.rand(32).to(DEVICE)
        error_signal = torch.rand(3).to(DEVICE)
        state = torch.cat([waveform_features, params, error_signal])
        
        # Get action from policy (with gradients)
        action, log_prob, value = policy.get_action(state)
        
        # Scale action to [0, 1]
        action_scaled = (action + 1) / 2
        
        # Get surrogate prediction (detached - no gradients through surrogate)
        topology_ids = torch.tensor([topo_id], device=DEVICE)
        with torch.no_grad():
            pred_wf, _ = surrogate(action_scaled.unsqueeze(0), topology_ids)
        
        # Generate target
        target_wf = generate_target_waveform(topology, action_scaled.detach().cpu().numpy())
        
        # Compute reward
        reward = compute_reward(
            pred_wf[0].cpu().numpy(),
            target_wf
        )
        
        reward_tensor = torch.tensor(reward, device=DEVICE)
        rewards_history.append(reward)
        
        # Update policy every step
        if log_prob is not None:
            # Advantage
            advantage = reward_tensor - value.detach()
            
            # Policy loss
            policy_loss = -log_prob * advantage
            value_loss = (value - reward_tensor.detach()) ** 2
            
            loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        
        # Save best every 50 episodes
        if episode % 50 == 0 and episode > 0:
            avg_reward = np.mean(rewards_history[-50:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save({
                    'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode': episode,
                    'reward': best_reward,
                }, f'checkpoints/rl_agent_{topology}_enhanced.pt')
    
    # Final save
    torch.save({
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': n_episodes,
        'reward': np.mean(rewards_history[-50:]),
    }, f'checkpoints/rl_agent_{topology}_enhanced.pt')
    
    print(f"   Final avg reward: {np.mean(rewards_history[-50:]):.3f}")
    print(f"   Best reward: {best_reward:.3f}")
    
    return policy, rewards_history


def main():
    print("="*60)
    print("🤖 RETRAINING RL AGENTS FOR CHALLENGING TOPOLOGIES")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load enhanced surrogate
    surrogate = load_enhanced_surrogate()
    
    results = {}
    
    for topology in CHALLENGING:
        policy, rewards = train_agent_for_topology(topology, surrogate, n_episodes=500)
        results[topology] = {
            'final_reward': float(np.mean(rewards[-50:])),
            'best_reward': float(max(rewards)),
            'n_episodes': len(rewards)
        }
    
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    for topo, res in results.items():
        print(f"\n{topo.upper()}:")
        print(f"  Final avg reward: {res['final_reward']:.3f}")
        print(f"  Best reward: {res['best_reward']:.3f}")
    
    print(f"\n✅ Enhanced agents saved to checkpoints/rl_agent_*_enhanced.pt")


if __name__ == '__main__':
    main()
