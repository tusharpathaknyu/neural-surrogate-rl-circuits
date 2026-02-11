"""
Train Multi-Topology Surrogate on Extended Dataset.

Trains the neural surrogate to predict waveforms for all 7 topologies:
- Buck, Boost, Buck-Boost, SEPIC, Ćuk, Flyback, QR Flyback

Uses topology embeddings for efficient multi-task learning.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json

# Device
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class MultiTopologySurrogate(nn.Module):
    """Neural surrogate supporting multiple circuit topologies."""
    
    def __init__(self, num_topologies=7, param_dim=6, waveform_len=512, 
                 embed_dim=32, hidden_dim=256):
        super().__init__()
        
        self.num_topologies = num_topologies
        self.param_dim = param_dim
        self.waveform_len = waveform_len
        
        # Topology embedding
        self.topology_embedding = nn.Embedding(num_topologies, embed_dim)
        
        # Shared encoder
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
        
        # Waveform generator
        self.waveform_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, waveform_len),
        )
        
        # 1D conv refiner for temporal coherence
        self.refiner = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )
        
        # Metrics head (efficiency, ripple)
        self.metrics_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # efficiency, ripple
            nn.Sigmoid(),
        )
    
    def forward(self, params, topology_ids):
        """
        Args:
            params: (batch, param_dim) - normalized circuit parameters
            topology_ids: (batch,) - integer topology IDs
        
        Returns:
            waveforms: (batch, waveform_len)
            metrics: (batch, 2) - efficiency, ripple
        """
        # Get topology embedding
        topo_embed = self.topology_embedding(topology_ids)  # (batch, embed_dim)
        
        # Concatenate params and topology
        x = torch.cat([params, topo_embed], dim=1)  # (batch, param_dim + embed_dim)
        
        # Encode
        features = self.encoder(x)  # (batch, hidden_dim)
        
        # Generate waveform
        waveform = self.waveform_head(features)  # (batch, waveform_len)
        
        # Refine with conv
        waveform = waveform.unsqueeze(1)  # (batch, 1, waveform_len)
        waveform = waveform + self.refiner(waveform)  # residual
        waveform = waveform.squeeze(1)  # (batch, waveform_len)
        
        # Predict metrics
        metrics = self.metrics_head(features)
        
        return waveform, metrics


def load_extended_dataset(data_dir: Path):
    """Load the extended multi-topology dataset.
    
    NORMALIZATION FIX (v2):
    - Old: per-sample waveform normalization (wf / max(abs(wf)))
      This destroyed voltage magnitude info — the model couldn't learn that
      buck outputs 6V vs boost outputs 24V. It only learned waveform shapes.
      
    - New: per-topology global normalization
      Compute mean/std per topology → model learns actual voltage ranges.
      Save normalization stats with checkpoint for proper denormalization.
    """
    print(f"\nLoading dataset from {data_dir}...")
    
    data = np.load(data_dir / 'combined_dataset.npz')
    
    params = data['params']
    waveforms = data['waveforms']
    topologies = data['topologies']
    
    print(f"  Params shape: {params.shape}")
    print(f"  Waveforms shape: {waveforms.shape}")
    print(f"  Topologies shape: {topologies.shape}")
    print(f"  Topology distribution: {np.bincount(topologies)}")
    
    # Normalize parameters (log-scale for component values)
    params_norm = params.copy()
    # Columns 0-3 are typically component values (L, C, R, V) - log normalize
    for i in range(min(4, params.shape[1])):
        col = params_norm[:, i]
        col = np.where(col > 0, col, 1e-10)
        params_norm[:, i] = np.log10(col)
    
    # Normalize to [-1, 1] range
    param_stats = {}
    for i in range(params_norm.shape[1]):
        col = params_norm[:, i]
        col_min, col_max = col.min(), col.max()
        param_stats[i] = {'min': float(col_min), 'max': float(col_max)}
        if col_max > col_min:
            params_norm[:, i] = 2 * (col - col_min) / (col_max - col_min) - 1
    
    # Per-TOPOLOGY normalization for waveforms (preserves magnitude info across topologies)
    num_topologies = len(np.unique(topologies))
    waveform_stats = {}
    waveforms_norm = waveforms.copy()
    
    for topo_id in range(num_topologies):
        mask = topologies == topo_id
        topo_waveforms = waveforms[mask]
        topo_mean = topo_waveforms.mean()
        topo_std = topo_waveforms.std()
        if topo_std < 1e-6:
            topo_std = 1.0
        
        waveforms_norm[mask] = (topo_waveforms - topo_mean) / topo_std
        waveform_stats[topo_id] = {
            'mean': float(topo_mean),
            'std': float(topo_std),
        }
        print(f"  Topo {topo_id}: wf_mean={topo_mean:.3f}V, wf_std={topo_std:.3f}V")
    
    return (
        torch.FloatTensor(params_norm),
        torch.FloatTensor(waveforms_norm),
        torch.LongTensor(topologies),
        waveform_stats,
        param_stats,
    )


def train_model(epochs=100, batch_size=128, lr=1e-3):
    """Train the multi-topology surrogate."""
    
    # Load data
    data_dir = Path('data/extended_topologies')
    params, waveforms, topologies, waveform_stats, param_stats = load_extended_dataset(data_dir)
    
    # Train/val split
    n = len(params)
    indices = torch.randperm(n)
    train_size = int(0.9 * n)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_dataset = TensorDataset(
        params[train_idx], waveforms[train_idx], topologies[train_idx]
    )
    val_dataset = TensorDataset(
        params[val_idx], waveforms[val_idx], topologies[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = MultiTopologySurrogate().to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Loss
    mse_loss = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_params, batch_waveforms, batch_topos in pbar:
            batch_params = batch_params.to(DEVICE)
            batch_waveforms = batch_waveforms.to(DEVICE)
            batch_topos = batch_topos.to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_waveforms, pred_metrics = model(batch_params, batch_topos)
            
            loss = mse_loss(pred_waveforms, batch_waveforms)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        history['train'].append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_params, batch_waveforms, batch_topos in val_loader:
                batch_params = batch_params.to(DEVICE)
                batch_waveforms = batch_waveforms.to(DEVICE)
                batch_topos = batch_topos.to(DEVICE)
                
                pred_waveforms, _ = model(batch_params, batch_topos)
                loss = mse_loss(pred_waveforms, batch_waveforms)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val'].append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'waveform_stats': waveform_stats,
                'param_stats': param_stats,
            }, checkpoint_dir / 'multi_topology_surrogate.pt')
            
            print(f"  ✓ Saved best model (val_loss = {val_loss:.6f})")
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    
    # Save training history
    with open('checkpoints/multi_topology_history.json', 'w') as f:
        json.dump(history, f)
    
    return model, history


if __name__ == '__main__':
    train_model(epochs=100, batch_size=128, lr=1e-3)
