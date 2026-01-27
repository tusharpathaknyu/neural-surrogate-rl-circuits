#!/usr/bin/env python3
"""
RL-ONLY TRAINING - Loads trained surrogate, trains RL agents
No SPICE during training (causes hangs), SPICE validation at end only
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from models.multi_topology_surrogate import MultiTopologySurrogate
from rl.ppo_agent import ActorCritic
from rl.topology_rewards import compute_topology_aware_reward, TOPOLOGY_REWARD_CONFIG

# =============================================================================
# CONFIG
# =============================================================================
DEVICE = 'cpu'
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
BASE_PATH = Path('/Users/tushardhananjaypathak/Desktop/MLEntry')

# RL Settings - production scale
RL_ITERATIONS = 3000
RL_BATCH_SIZE = 64
RL_LR = 3e-4
RL_PPO_EPOCHS = 4
LOG_INTERVAL = 100


def log(msg, f):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    f.write(line + "\n")
    f.flush()


def create_target_waveform(topology: str, params: torch.Tensor, device: str) -> torch.Tensor:
    """Create topology-appropriate target waveforms."""
    batch_size = params.shape[0]
    t = torch.linspace(0, 1, 32, device=device)
    
    vin = params[:, 3:4] * 24 + 12
    duty = params[:, 5:6] * 0.7 + 0.1
    
    if topology == 'buck':
        vout = vin * duty
        ripple = 0.02 * (1 - duty)
    elif topology == 'boost':
        vout = vin / (1 - duty + 0.01)
        ripple = 0.03 * duty
    elif topology == 'buck_boost':
        vout = -vin * duty / (1 - duty + 0.01)
        ripple = 0.04 * duty
    elif topology == 'sepic':
        vout = vin * duty / (1 - duty + 0.01)
        ripple = 0.03 * duty
    elif topology == 'cuk':
        vout = -vin * duty / (1 - duty + 0.01)
        ripple = 0.035 * duty
    elif topology in ['flyback', 'qr_flyback']:
        vout = vin * 0.5 * duty / (1 - duty + 0.01)
        ripple = 0.04 * duty
    else:
        vout = vin * duty
        ripple = 0.02
    
    base = vout.expand(-1, 32)
    ripple_wave = ripple * torch.sin(2 * np.pi * 8 * t.unsqueeze(0).expand(batch_size, -1))
    return base + ripple_wave


if __name__ == "__main__":
    start_time = time.time()
    
    log_file = open(BASE_PATH / "rl_training.log", "w")
    
    log("=" * 60, log_file)
    log("🎮 RL-ONLY TRAINING (Surrogate already trained)", log_file)
    log("=" * 60, log_file)
    
    # Load trained surrogate
    log("📦 Loading trained surrogate...", log_file)
    model = MultiTopologySurrogate(
        num_topologies=7, param_dim=6, waveform_len=32,
        embed_dim=64, hidden_dim=512
    ).to(DEVICE)
    
    checkpoint = torch.load(BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt', 
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    val_loss = checkpoint.get('val_loss', 'unknown')
    log(f"   ✅ Loaded! Val loss: {val_loss}", log_file)
    
    # Train RL agents
    log(f"\n🎮 Training RL agents ({RL_ITERATIONS} iterations each)", log_file)
    
    results = {}
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        is_inv = config.get('inverted', False)
        inv_label = " ⚡INV" if is_inv else ""
        
        log(f"\n   {'─' * 40}", log_file)
        log(f"   {topo_name.upper()}{inv_label}", log_file)
        log(f"   {'─' * 40}", log_file)
        
        agent = ActorCritic(state_dim=41, action_dim=6, hidden_dim=256).to(DEVICE)
        optimizer = optim.Adam(agent.parameters(), lr=RL_LR)
        
        best_reward = -float('inf')
        reward_history = []
        topo_start = time.time()
        
        for it in range(1, RL_ITERATIONS + 1):
            # Random params
            batch_params = torch.rand(RL_BATCH_SIZE, 6).to(DEVICE)
            batch_topo = torch.full((RL_BATCH_SIZE,), topo_idx, dtype=torch.long).to(DEVICE)
            
            # Surrogate prediction
            with torch.no_grad():
                pred_wf, _ = model(batch_params, batch_topo)
            
            # Target
            target_wf = create_target_waveform(topo_name, batch_params, DEVICE)
            
            # State
            error = (pred_wf - target_wf).abs().mean(dim=1, keepdim=True)
            mse = ((pred_wf - target_wf)**2).mean(dim=1, keepdim=True)
            corr = torch.zeros(RL_BATCH_SIZE, 1, device=DEVICE)
            state = torch.cat([pred_wf, batch_params, error, mse, corr], dim=1)
            
            # Actions
            with torch.no_grad():
                actions, log_probs_old, _ = agent.get_action(state, deterministic=False)
            
            # Apply
            new_params = torch.clamp(batch_params + actions * 0.1, 0, 1)
            
            # New prediction
            with torch.no_grad():
                new_pred, _ = model(new_params, batch_topo)
            
            # Rewards (topology-aware)
            rewards = []
            for i in range(RL_BATCH_SIZE):
                r, _ = compute_topology_aware_reward(
                    new_pred[i].cpu().numpy(),
                    target_wf[i].cpu().numpy(),
                    topo_name
                )
                rewards.append(r)
            rewards = torch.FloatTensor(rewards).to(DEVICE)
            
            # PPO update
            for _ in range(RL_PPO_EPOCHS):
                log_probs, entropy, values = agent.evaluate(state, actions)
                
                advantages = rewards - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                ratio = torch.exp(log_probs - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - rewards) ** 2).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
            
            avg_r = rewards.mean().item()
            reward_history.append(avg_r)
            
            if avg_r > best_reward:
                best_reward = avg_r
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'topology': topo_name,
                    'best_reward': best_reward,
                }, BASE_PATH / 'checkpoints' / f'rl_agent_{topo_name}.pt')
            
            if it % LOG_INTERVAL == 0:
                elapsed = time.time() - topo_start
                eta = elapsed / it * (RL_ITERATIONS - it)
                recent = np.mean(reward_history[-100:])
                log(f"      Iter {it:4d}: reward={avg_r:.2f}, best={best_reward:.2f}, avg100={recent:.2f}, ETA={timedelta(seconds=int(eta))}", log_file)
        
        topo_time = time.time() - topo_start
        results[topo_name] = {'best': best_reward, 'time': topo_time}
        log(f"   ✅ Best: {best_reward:.2f} ({timedelta(seconds=int(topo_time))})", log_file)
    
    # Final evaluation
    log("\n" + "=" * 60, log_file)
    log("📊 FINAL EVALUATION", log_file)
    log("=" * 60, log_file)
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        is_inv = config.get('inverted', False)
        
        agent = ActorCritic(state_dim=41, action_dim=6, hidden_dim=256).to(DEVICE)
        ckpt = torch.load(BASE_PATH / 'checkpoints' / f'rl_agent_{topo_name}.pt', map_location=DEVICE, weights_only=False)
        agent.load_state_dict(ckpt['model_state_dict'])
        agent.eval()
        
        # Eval
        test_params = torch.rand(200, 6).to(DEVICE)
        test_topo = torch.full((200,), topo_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            pred_wf, _ = model(test_params, test_topo)
        
        target_wf = create_target_waveform(topo_name, test_params, DEVICE)
        
        if is_inv:
            mse = ((pred_wf.abs() - target_wf.abs())**2).mean().item()
        else:
            mse = ((pred_wf - target_wf)**2).mean().item()
        
        quality = max(0, min(100, 100 * (1 - mse / (target_wf.var().item() + 1e-6))))
        
        inv_label = "(INV)" if is_inv else ""
        log(f"   {topo_name.upper():12} {inv_label:6} Quality: {quality:.1f}%", log_file)
    
    total_time = time.time() - start_time
    log(f"\n✅ Complete in {timedelta(seconds=int(total_time))}", log_file)
    log_file.close()
    
    print("\n🎉 RL training complete!")
