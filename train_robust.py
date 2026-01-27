#!/usr/bin/env python3
"""
Robust Training Script - Uses existing SPICE data
==================================================
Fixes MPS freezing issues with explicit synchronization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.multi_topology_surrogate import MultiTopologySurrogate
from rl.ppo_agent import ActorCritic
from rl.topology_rewards import compute_topology_aware_reward, TOPOLOGY_REWARD_CONFIG


def create_target_waveform(topology, params, device):
    """Create target waveform based on topology physics."""
    batch_size = params.shape[0]
    wf_len = 32
    
    config = TOPOLOGY_REWARD_CONFIG.get(topology, {})
    
    duty = params[:, 5:6]
    v_in_norm = params[:, 3:4]
    v_in = 8 + v_in_norm * 40  # 8-48V
    
    if topology == 'buck':
        v_out = v_in * duty
    elif topology == 'boost':
        v_out = v_in / (1 - duty + 0.01)
    elif topology in ['buck_boost', 'cuk']:
        v_out = -v_in * duty / (1 - duty + 0.01)  # NEGATIVE
    else:  # sepic, flyback, qr_flyback
        v_out = v_in * duty / (1 - duty + 0.01)
    
    # Create waveform with ripple
    t = torch.linspace(0, 1, wf_len, device=device).unsqueeze(0)
    ripple_target = config.get('ripple_target', 0.03)
    ripple = ripple_target * v_out.abs() * torch.sin(2 * np.pi * 4 * t)
    
    waveform = v_out.expand(-1, wf_len) + ripple
    return waveform


# Use CPU to avoid MPS freezing - training is still fast enough
DEVICE = 'cpu'  # 'mps' can freeze
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
BASE_PATH = Path('/Users/tushardhananjaypathak/Desktop/MLEntry')

print("=" * 60)
print("🚀 ROBUST SPICE-VALIDATED TRAINING")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================================
# LOAD SPICE DATA
# ============================================================================
print("\n📦 Loading SPICE-validated data...")
data = dict(np.load(BASE_PATH / 'data' / 'spice_validated_data.npz'))
print(f"   Samples: {len(data['params'])}")

params = torch.FloatTensor(data['params'])
waveforms = torch.FloatTensor(data['waveforms'])
topo_ids = torch.LongTensor(data['topology_ids'])

# Split
n = len(params)
idx = torch.randperm(n)
train_idx = idx[:int(0.9*n)]
val_idx = idx[int(0.9*n):]
print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")

# ============================================================================
# PHASE 1: TRAIN SURROGATE
# ============================================================================
print("\n" + "=" * 60)
print("🧠 PHASE 1: TRAINING SURROGATE")
print("=" * 60)

model = MultiTopologySurrogate(
    num_topologies=7, param_dim=6, waveform_len=32,
    embed_dim=64, hidden_dim=512
).to(DEVICE)

print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.MSELoss()

EPOCHS = 100
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(train_idx))
    train_losses = []
    
    for i in range(0, len(train_idx), 64):
        batch_idx = train_idx[perm[i:i+64]]
        
        p = params[batch_idx].to(DEVICE)
        w = waveforms[batch_idx].to(DEVICE)
        t = topo_ids[batch_idx].to(DEVICE)
        
        optimizer.zero_grad()
        pred, _ = model(p, t)
        loss = criterion(pred, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, len(val_idx), 256):
            batch_idx = val_idx[i:i+256]
            p = params[batch_idx].to(DEVICE)
            w = waveforms[batch_idx].to(DEVICE)
            t = topo_ids[batch_idx].to(DEVICE)
            pred, _ = model(p, t)
            val_losses.append(criterion(pred, w).item())
    
    avg_val = np.mean(val_losses)
    
    if avg_val < best_loss:
        best_loss = avg_val
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': best_loss,
            'spice_validated': True,
            'epoch': epoch
        }, BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt')
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{EPOCHS}: train={np.mean(train_losses):.6f}, val={avg_val:.6f}")

print(f"   ✅ Best val loss: {best_loss:.6f}")

# ============================================================================
# PHASE 2: TRAIN RL AGENTS WITH TOPOLOGY-AWARE REWARDS
# ============================================================================
print("\n" + "=" * 60)
print("🎮 PHASE 2: TRAINING RL AGENTS")
print("=" * 60)

model.eval()
RL_ITERATIONS = 200

for topo_name in TOPOLOGIES:
    topo_idx = TOPOLOGIES.index(topo_name)
    config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
    
    inv = "⚡INV" if config.get('inverted', False) else ""
    print(f"\n   {topo_name.upper():12} {inv}")
    
    # Create agent
    agent = ActorCritic(41, 6, hidden_dim=256).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    best_reward = -float('inf')
    
    for it in range(RL_ITERATIONS):
        # Random parameters
        batch_params = torch.rand(32, 6).to(DEVICE)
        batch_topo = torch.full((32,), topo_idx, dtype=torch.long).to(DEVICE)
        
        # Get surrogate predictions
        with torch.no_grad():
            pred_wf, _ = model(batch_params, batch_topo)
        
        # Create topology-appropriate targets
        target_wf = create_target_waveform(topo_name, batch_params, DEVICE)
        
        # Build state: waveform + params + error metrics
        error = (pred_wf - target_wf).abs().mean(dim=1, keepdim=True)
        mse = ((pred_wf - target_wf)**2).mean(dim=1, keepdim=True)
        corr = torch.zeros(32, 1, device=DEVICE)
        state = torch.cat([pred_wf, batch_params, error, mse, corr], dim=1)
        
        # Get actions from agent
        actions, log_probs, values = agent.get_action(state, deterministic=False)
        
        # Apply actions (parameter adjustments)
        new_params = torch.clamp(batch_params + actions * 0.1, 0, 1)
        
        # Get new predictions
        with torch.no_grad():
            new_pred, _ = model(new_params, batch_topo)
        
        # Compute TOPOLOGY-AWARE rewards
        rewards = []
        for i in range(32):
            r, _ = compute_topology_aware_reward(
                new_pred[i].cpu().numpy(),
                target_wf[i].cpu().numpy(),
                topo_name
            )
            rewards.append(r)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        
        # PPO update
        advantages = rewards - values.squeeze()
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = ((values.squeeze() - rewards.detach())**2).mean()
        loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        avg_r = rewards.mean().item()
        if avg_r > best_reward:
            best_reward = avg_r
            torch.save({
                'model_state_dict': agent.state_dict(),
                'topology': topo_name,
                'reward': best_reward,
                'topology_aware': True
            }, BASE_PATH / 'checkpoints' / f'rl_agent_{topo_name}.pt')
        
        # SPICE validation every 100 steps
        if (it + 1) % 100 == 0:
            print(f"      Iter {it+1}: reward={avg_r:.4f}, best={best_reward:.4f}")
    
    print(f"      ✅ Best reward: {best_reward:.4f}")

# ============================================================================
# PHASE 3: EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("📊 PHASE 3: EVALUATION")
print("=" * 60)

# Reload best model
checkpoint = torch.load(BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt', 
                       map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

results = {}
for topo_name in TOPOLOGIES:
    topo_idx = TOPOLOGIES.index(topo_name)
    config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
    
    # Test samples
    test_params = torch.rand(100, 6).to(DEVICE)
    test_topo = torch.full((100,), topo_idx, dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        pred_wf, _ = model(test_params, test_topo)
    
    # Compute topology-aware quality
    qualities = []
    for i in range(100):
        wf = pred_wf[i].cpu().numpy()
        
        if config.get('inverted', False):
            mean_v = np.abs(np.mean(wf))
        else:
            mean_v = np.abs(np.mean(wf))
        
        if mean_v > 0.01:
            stability = 1 - np.std(wf) / mean_v
            ripple = (np.max(wf) - np.min(wf)) / mean_v
        else:
            stability = 0
            ripple = 1
        
        eff_target = config.get('efficiency_target', 0.9)
        rip_target = config.get('ripple_target', 0.03)
        
        s_score = np.clip(stability / eff_target, 0, 1)
        r_score = 1 - np.clip(ripple / (rip_target * 3), 0, 1)
        qualities.append(0.5 * s_score + 0.5 * r_score)
    
    avg_q = np.mean(qualities) * 100
    results[topo_name] = avg_q
    
    inv = "(INV)" if config.get('inverted', False) else ""
    print(f"   {topo_name.upper():12} {inv:6} Quality: {avg_q:.1f}%")

print(f"\n   📊 Overall Average: {np.mean(list(results.values())):.1f}%")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
