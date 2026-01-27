#!/usr/bin/env python3
"""
Retrain Surrogate with Enhanced Challenging Topology Data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class MultiTopologySurrogate(nn.Module):
    """Enhanced surrogate with better architecture for challenging topologies."""
    
    def __init__(self, num_topologies=7, param_dim=6, waveform_len=512, 
                 embed_dim=64, hidden_dim=512):  # Larger embedding and hidden
        super().__init__()
        
        self.num_topologies = num_topologies
        self.param_dim = param_dim
        self.waveform_len = waveform_len
        
        # Larger topology embedding
        self.topology_embedding = nn.Embedding(num_topologies, embed_dim)
        
        # Deeper encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(param_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Waveform head
        self.waveform_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, waveform_len),
        )
        
        # Enhanced 1D conv refiner
        self.refiner = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )
        
        # Metrics head
        self.metrics_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
    
    def forward(self, params, topology_ids):
        topo_embed = self.topology_embedding(topology_ids)
        x = torch.cat([params, topo_embed], dim=1)
        features = self.encoder(x)
        
        waveform = self.waveform_head(features)
        waveform = waveform.unsqueeze(1)
        waveform = waveform + self.refiner(waveform)
        waveform = waveform.squeeze(1)
        
        metrics = self.metrics_head(features)
        
        return waveform, metrics


def load_all_data():
    """Load both original and enhanced data."""
    print("\n📂 Loading datasets...")
    
    all_params = []
    all_waveforms = []
    all_topologies = []
    
    # Original extended topologies
    orig_path = Path('data/extended_topologies/combined_dataset.npz')
    if orig_path.exists():
        data = np.load(orig_path)
        params = data['params']
        waveforms = data['waveforms']
        topologies = data['topologies']
        
        print(f"  Original data: {len(params)} samples")
        all_params.append(params)
        all_waveforms.append(waveforms)
        all_topologies.append(topologies)
    
    # Enhanced challenging topology data
    challenging_dir = Path('data/challenging_topologies')
    topo_map = {'cuk': 4, 'flyback': 5, 'qr_flyback': 6}
    
    for topo_name, topo_id in topo_map.items():
        path = challenging_dir / f'{topo_name}_enhanced.npz'
        if path.exists():
            data = np.load(path)
            n = len(data['waveforms'])
            
            # Normalize params
            params = data['params']
            for i in range(params.shape[1]):
                col = params[:, i]
                if col.max() > col.min():
                    params[:, i] = (col - col.min()) / (col.max() - col.min())
            
            # Normalize waveforms
            waveforms = data['waveforms']
            wf_max = np.abs(waveforms).max(axis=1, keepdims=True)
            wf_max = np.where(wf_max > 0, wf_max, 1)
            waveforms = waveforms / wf_max
            
            all_params.append(params)
            all_waveforms.append(waveforms)
            all_topologies.append(np.full(n, topo_id))
            
            print(f"  Enhanced {topo_name}: {n} samples")
    
    # Combine
    params = np.vstack(all_params)
    waveforms = np.vstack(all_waveforms)
    topologies = np.concatenate(all_topologies)
    
    print(f"\n  Total: {len(params)} samples")
    print(f"  Topology distribution: {np.bincount(topologies.astype(int))}")
    
    return (
        torch.FloatTensor(params),
        torch.FloatTensor(waveforms),
        torch.LongTensor(topologies),
    )


def train_enhanced(epochs=50, batch_size=128, lr=5e-4):
    """Train enhanced surrogate model."""
    
    params, waveforms, topologies = load_all_data()
    
    # Train/val split
    n = len(params)
    indices = torch.randperm(n)
    train_size = int(0.9 * n)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_dataset = TensorDataset(params[train_idx], waveforms[train_idx], topologies[train_idx])
    val_dataset = TensorDataset(params[val_idx], waveforms[val_idx], topologies[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"\n📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = MultiTopologySurrogate().to(DEVICE)
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    print("\n🏋️ Training Enhanced Surrogate...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_params, batch_wf, batch_topo in pbar:
            batch_params = batch_params.to(DEVICE)
            batch_wf = batch_wf.to(DEVICE)
            batch_topo = batch_topo.to(DEVICE)
            
            optimizer.zero_grad()
            pred_wf, _ = model(batch_params, batch_topo)
            loss = mse_loss(pred_wf, batch_wf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_params, batch_wf, batch_topo in val_loader:
                batch_params = batch_params.to(DEVICE)
                batch_wf = batch_wf.to(DEVICE)
                batch_topo = batch_topo.to(DEVICE)
                
                pred_wf, _ = model(batch_params, batch_topo)
                val_loss += mse_loss(pred_wf, batch_wf).item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'checkpoints/multi_topology_surrogate_enhanced.pt')
            print(f"  ✓ Saved best (val_loss={val_loss:.6f})")
    
    print(f"\n✅ Training complete! Best val_loss: {best_val_loss:.6f}")
    return model


if __name__ == '__main__':
    train_enhanced(epochs=50, batch_size=128, lr=5e-4)
