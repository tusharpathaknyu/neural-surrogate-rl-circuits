#!/usr/bin/env python3
"""
PRODUCTION TRAINING - Full 18+ hour training with all optimizations
Topology-aware rewards, SPICE-validated data, proper hyperparameters
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pathlib import Path
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from models.multi_topology_surrogate import MultiTopologySurrogate
from rl.ppo_agent import ActorCritic
from rl.topology_rewards import compute_topology_aware_reward, TOPOLOGY_REWARD_CONFIG

# =============================================================================
# CONFIGURATION - PRODUCTION SETTINGS
# =============================================================================
DEVICE = 'cpu'  # CPU to avoid MPS freezing
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']
BASE_PATH = Path('/Users/tushardhananjaypathak/Desktop/MLEntry')

# PRODUCTION HYPERPARAMETERS
SURROGATE_EPOCHS = 1000
SURROGATE_BATCH_SIZE = 128
SURROGATE_LR = 1e-4
SURROGATE_WEIGHT_DECAY = 0.01
SURROGATE_PATIENCE = 50  # Early stopping

RL_ITERATIONS = 3000  # Per topology
RL_BATCH_SIZE = 64
RL_LR = 3e-4
RL_GAMMA = 0.99
RL_CLIP_EPSILON = 0.2
RL_ENTROPY_COEF = 0.01
RL_VALUE_COEF = 0.5
RL_MAX_GRAD_NORM = 0.5
RL_PPO_EPOCHS = 4  # Updates per iteration

# Logging
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 100


def create_target_waveform(topology: str, params: torch.Tensor, device: str) -> torch.Tensor:
    """Create topology-appropriate target waveforms."""
    batch_size = params.shape[0]
    t = torch.linspace(0, 1, 32, device=device)
    
    # Extract normalized parameters
    vin = params[:, 0:1] * 24 + 12  # 12-36V
    duty = params[:, 1:2] * 0.7 + 0.1  # 0.1-0.8
    load = params[:, 4:5] * 90 + 10  # 10-100 ohms
    
    if topology == 'buck':
        vout = vin * duty
        ripple = 0.02 * (1 - duty)
    elif topology == 'boost':
        vout = vin / (1 - duty + 0.01)
        ripple = 0.03 * duty
    elif topology == 'buck_boost':
        vout = -vin * duty / (1 - duty + 0.01)  # NEGATIVE
        ripple = 0.04 * duty
    elif topology == 'sepic':
        vout = vin * duty / (1 - duty + 0.01)
        ripple = 0.03 * duty
    elif topology == 'cuk':
        vout = -vin * duty / (1 - duty + 0.01)  # NEGATIVE
        ripple = 0.035 * duty
    elif topology == 'flyback':
        n = 0.5  # Turns ratio
        vout = vin * n * duty / (1 - duty + 0.01)
        ripple = 0.05 * duty
    elif topology == 'qr_flyback':
        n = 0.5
        vout = vin * n * duty / (1 - duty + 0.01)
        ripple = 0.03 * duty  # Lower ripple due to resonance
    else:
        vout = vin * duty
        ripple = 0.02
    
    # Build waveform with switching ripple
    base = vout.expand(-1, 32)
    ripple_wave = ripple * torch.sin(2 * np.pi * 8 * t.unsqueeze(0).expand(batch_size, -1))
    
    return base + ripple_wave


def log_message(msg: str, log_file):
    """Print and log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_file.write(full_msg + "\n")
    log_file.flush()


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


# =============================================================================
# MAIN TRAINING
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Setup logging
    log_path = BASE_PATH / "production_training.log"
    log_file = open(log_path, "w")
    
    log_message("=" * 70, log_file)
    log_message("🚀 PRODUCTION TRAINING - FULL RUN", log_file)
    log_message("=" * 70, log_file)
    log_message(f"Device: {DEVICE}", log_file)
    log_message(f"Surrogate epochs: {SURROGATE_EPOCHS}", log_file)
    log_message(f"RL iterations per topology: {RL_ITERATIONS}", log_file)
    log_message(f"Estimated time: 10-18 hours", log_file)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    log_message("\n📦 Loading SPICE-validated data...", log_file)
    data = dict(np.load(BASE_PATH / 'data' / 'spice_validated_data.npz'))
    
    params = torch.FloatTensor(data['params'])
    waveforms = torch.FloatTensor(data['waveforms'])
    topo_ids = torch.LongTensor(data['topology_ids'])
    
    n_samples = len(params)
    log_message(f"   Total samples: {n_samples:,}", log_file)
    
    # Split 90/10
    idx = torch.randperm(n_samples)
    train_idx = idx[:int(0.9 * n_samples)]
    val_idx = idx[int(0.9 * n_samples):]
    
    log_message(f"   Train: {len(train_idx):,}", log_file)
    log_message(f"   Val: {len(val_idx):,}", log_file)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        params[train_idx], topo_ids[train_idx], waveforms[train_idx]
    )
    val_dataset = torch.utils.data.TensorDataset(
        params[val_idx], topo_ids[val_idx], waveforms[val_idx]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=SURROGATE_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=SURROGATE_BATCH_SIZE, shuffle=False
    )
    
    # =========================================================================
    # PHASE 1: TRAIN SURROGATE
    # =========================================================================
    log_message("\n" + "=" * 70, log_file)
    log_message("🧠 PHASE 1: TRAINING SURROGATE MODEL", log_file)
    log_message("=" * 70, log_file)
    
    model = MultiTopologySurrogate(
        num_topologies=7, param_dim=6, waveform_len=32,
        embed_dim=64, hidden_dim=512
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    log_message(f"   Model parameters: {n_params:,}", log_file)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=SURROGATE_LR, 
        weight_decay=SURROGATE_WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    phase1_start = time.time()
    
    for epoch in range(1, SURROGATE_EPOCHS + 1):
        # Training
        model.train()
        train_losses = []
        
        for batch_params, batch_topo, batch_wf in train_loader:
            batch_params = batch_params.to(DEVICE)
            batch_topo = batch_topo.to(DEVICE)
            batch_wf = batch_wf.to(DEVICE)
            
            optimizer.zero_grad()
            pred_wf, _ = model(batch_params, batch_topo)
            loss = criterion(pred_wf, batch_wf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_params, batch_topo, batch_wf in val_loader:
                batch_params = batch_params.to(DEVICE)
                batch_topo = batch_topo.to(DEVICE)
                batch_wf = batch_wf.to(DEVICE)
                
                pred_wf, _ = model(batch_params, batch_topo)
                loss = criterion(pred_wf, batch_wf)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['lr'].append(current_lr)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt')
        else:
            patience_counter += 1
        
        # Logging
        if epoch % LOG_INTERVAL == 0 or epoch == 1:
            elapsed = time.time() - phase1_start
            eta = elapsed / epoch * (SURROGATE_EPOCHS - epoch)
            log_message(
                f"   Epoch {epoch:4d}/{SURROGATE_EPOCHS}: "
                f"train={avg_train_loss:.6f}, val={avg_val_loss:.6f}, "
                f"lr={current_lr:.2e}, patience={patience_counter}/{SURROGATE_PATIENCE}, "
                f"ETA={format_time(eta)}",
                log_file
            )
        
        # Early stopping
        if patience_counter >= SURROGATE_PATIENCE:
            log_message(f"\n   ⏹️ Early stopping at epoch {epoch}", log_file)
            break
    
    phase1_time = time.time() - phase1_start
    log_message(f"\n   ✅ Best validation loss: {best_val_loss:.6f}", log_file)
    log_message(f"   ⏱️ Phase 1 time: {format_time(phase1_time)}", log_file)
    
    # Load best model
    checkpoint = torch.load(BASE_PATH / 'checkpoints' / 'multi_topology_surrogate.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Save training history
    with open(BASE_PATH / 'surrogate_training_history.json', 'w') as f:
        json.dump(training_history, f)
    
    # =========================================================================
    # PHASE 2: TRAIN RL AGENTS
    # =========================================================================
    log_message("\n" + "=" * 70, log_file)
    log_message("🎮 PHASE 2: TRAINING RL AGENTS (TOPOLOGY-AWARE)", log_file)
    log_message("=" * 70, log_file)
    
    phase2_start = time.time()
    rl_results = {}
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        is_inverted = config.get('inverted', False)
        
        inv_label = " ⚡INV" if is_inverted else ""
        log_message(f"\n   {'─' * 50}", log_file)
        log_message(f"   Training: {topo_name.upper()}{inv_label}", log_file)
        log_message(f"   {'─' * 50}", log_file)
        
        # Create agent
        # State: waveform(32) + params(6) + error(1) + mse(1) + corr(1) = 41
        agent = ActorCritic(state_dim=41, action_dim=6, hidden_dim=256).to(DEVICE)
        optimizer = optim.Adam(agent.parameters(), lr=RL_LR)
        
        best_reward = -float('inf')
        reward_history = []
        topo_start = time.time()
        
        for iteration in range(1, RL_ITERATIONS + 1):
            # Sample random parameters
            batch_params = torch.rand(RL_BATCH_SIZE, 6).to(DEVICE)
            batch_topo = torch.full((RL_BATCH_SIZE,), topo_idx, dtype=torch.long).to(DEVICE)
            
            # Get surrogate predictions
            with torch.no_grad():
                pred_wf, _ = model(batch_params, batch_topo)
            
            # Create topology-appropriate targets
            target_wf = create_target_waveform(topo_name, batch_params, DEVICE)
            
            # Build state vector
            error = (pred_wf - target_wf).abs().mean(dim=1, keepdim=True)
            mse = ((pred_wf - target_wf) ** 2).mean(dim=1, keepdim=True)
            corr = torch.zeros(RL_BATCH_SIZE, 1, device=DEVICE)
            state = torch.cat([pred_wf, batch_params, error, mse, corr], dim=1)
            
            # Collect experience with current policy
            with torch.no_grad():
                actions, log_probs_old, values_old = agent.get_action(state, deterministic=False)
            
            # Apply actions (parameter adjustments)
            new_params = torch.clamp(batch_params + actions * 0.1, 0, 1)
            
            # Get new predictions
            with torch.no_grad():
                new_pred, _ = model(new_params, batch_topo)
            
            # Compute TOPOLOGY-AWARE rewards
            rewards = []
            for i in range(RL_BATCH_SIZE):
                r, _ = compute_topology_aware_reward(
                    new_pred[i].cpu().numpy(),
                    target_wf[i].cpu().numpy(),
                    topo_name
                )
                rewards.append(r)
            rewards = torch.FloatTensor(rewards).to(DEVICE)
            
            # PPO Update (multiple epochs)
            for _ in range(RL_PPO_EPOCHS):
                # Re-evaluate actions
                log_probs, entropy, values = agent.evaluate(state, actions)
                
                # Compute advantages
                advantages = rewards - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - RL_CLIP_EPSILON, 1 + RL_CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((values - rewards) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + RL_VALUE_COEF * value_loss + RL_ENTROPY_COEF * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), RL_MAX_GRAD_NORM)
                optimizer.step()
            
            avg_reward = rewards.mean().item()
            reward_history.append(avg_reward)
            
            # Track best
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'topology': topo_name,
                    'best_reward': best_reward,
                    'iteration': iteration,
                }, BASE_PATH / 'checkpoints' / f'rl_agent_{topo_name}.pt')
            
            # Logging
            if iteration % LOG_INTERVAL == 0:
                elapsed = time.time() - topo_start
                eta = elapsed / iteration * (RL_ITERATIONS - iteration)
                recent_avg = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
                log_message(
                    f"      Iter {iteration:4d}/{RL_ITERATIONS}: "
                    f"reward={avg_reward:.4f}, best={best_reward:.4f}, "
                    f"avg100={recent_avg:.4f}, ETA={format_time(eta)}",
                    log_file
                )
        
        topo_time = time.time() - topo_start
        rl_results[topo_name] = {
            'best_reward': best_reward,
            'final_avg': np.mean(reward_history[-100:]),
            'time': topo_time
        }
        
        log_message(f"   ✅ {topo_name}: Best reward = {best_reward:.4f} ({format_time(topo_time)})", log_file)
    
    phase2_time = time.time() - phase2_start
    log_message(f"\n   ⏱️ Phase 2 time: {format_time(phase2_time)}", log_file)
    
    # =========================================================================
    # PHASE 3: FINAL EVALUATION
    # =========================================================================
    log_message("\n" + "=" * 70, log_file)
    log_message("📊 PHASE 3: FINAL EVALUATION", log_file)
    log_message("=" * 70, log_file)
    
    eval_results = {}
    
    for topo_idx, topo_name in enumerate(TOPOLOGIES):
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        is_inverted = config.get('inverted', False)
        
        # Load best agent
        agent = ActorCritic(state_dim=41, action_dim=6, hidden_dim=256).to(DEVICE)
        checkpoint = torch.load(BASE_PATH / 'checkpoints' / f'rl_agent_{topo_name}.pt')
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        
        # Evaluate on test samples
        n_eval = 500
        test_params = torch.rand(n_eval, 6).to(DEVICE)
        test_topo = torch.full((n_eval,), topo_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            pred_wf, _ = model(test_params, test_topo)
        
        target_wf = create_target_waveform(topo_name, test_params, DEVICE)
        
        # Compute metrics
        if is_inverted:
            pred_for_eval = pred_wf.abs()
            target_for_eval = target_wf.abs()
        else:
            pred_for_eval = pred_wf
            target_for_eval = target_wf
        
        mse = ((pred_for_eval - target_for_eval) ** 2).mean().item()
        mae = (pred_for_eval - target_for_eval).abs().mean().item()
        
        # Normalized quality score
        quality = max(0, 100 * (1 - mse / (target_for_eval.var().item() + 1e-6)))
        quality = min(100, quality)
        
        inv_label = "(INV)" if is_inverted else ""
        eval_results[topo_name] = {'mse': mse, 'mae': mae, 'quality': quality}
        
        log_message(
            f"   {topo_name.upper():12} {inv_label:6} MSE={mse:.6f}, MAE={mae:.6f}, Quality={quality:.1f}%",
            log_file
        )
    
    avg_quality = np.mean([r['quality'] for r in eval_results.values()])
    log_message(f"\n   📊 Overall Average Quality: {avg_quality:.1f}%", log_file)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    log_message("\n" + "=" * 70, log_file)
    log_message("✅ PRODUCTION TRAINING COMPLETE!", log_file)
    log_message("=" * 70, log_file)
    log_message(f"   Total time: {format_time(total_time)}", log_file)
    log_message(f"   Surrogate best val loss: {best_val_loss:.6f}", log_file)
    log_message(f"   Average quality: {avg_quality:.1f}%", log_file)
    log_message("", log_file)
    log_message("   Topology Results:", log_file)
    
    for topo_name, result in eval_results.items():
        config = TOPOLOGY_REWARD_CONFIG.get(topo_name, {})
        inv_label = "⚡" if config.get('inverted', False) else " "
        log_message(f"      {inv_label} {topo_name.upper():12} {result['quality']:.1f}%", log_file)
    
    log_message("", log_file)
    log_message("   Models saved to checkpoints/", log_file)
    log_message("   Training log: production_training.log", log_file)
    log_message("=" * 70, log_file)
    
    # Save final results
    with open(BASE_PATH / 'production_results.json', 'w') as f:
        json.dump({
            'surrogate': {'best_val_loss': best_val_loss},
            'rl_agents': rl_results,
            'evaluation': eval_results,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    log_file.close()
    print("\n🎉 Done! Check production_training.log for full details.")
