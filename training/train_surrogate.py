"""
Training script for the forward surrogate model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.forward_surrogate import ForwardSurrogate, ForwardSurrogateWithMetrics
from training.losses import PowerElectronicsLoss


class BuckConverterDataset(Dataset):
    """Dataset for buck converter surrogate training."""
    
    def __init__(self, data_dir: str, normalize_waveforms: bool = True):
        self.data_dir = Path(data_dir)
        
        # Load numpy arrays
        self.params = np.load(self.data_dir / 'params.npy')
        self.waveforms = np.load(self.data_dir / 'waveforms.npy')
        
        # Normalize waveforms to [0, 1] range per sample
        if normalize_waveforms:
            self.waveform_min = self.waveforms.min(axis=1, keepdims=True)
            self.waveform_max = self.waveforms.max(axis=1, keepdims=True)
            # Keep original for denormalization
            self.waveforms_raw = self.waveforms.copy()
        
        # Convert to tensors
        self.params = torch.tensor(self.params, dtype=torch.float32)
        self.waveforms = torch.tensor(self.waveforms, dtype=torch.float32)
        
        print(f"Loaded {len(self)} samples")
        print(f"  Params shape: {self.params.shape}")
        print(f"  Waveforms shape: {self.waveforms.shape}")
    
    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        return {
            'params': self.params[idx],
            'waveform': self.waveforms[idx],
        }


def train_surrogate(
    data_dir: str = './data',
    output_dir: str = './checkpoints',
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'auto',
):
    """Train the forward surrogate model."""
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = BuckConverterDataset(data_dir)
    
    # Train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = ForwardSurrogate(
        num_params=6,
        waveform_length=dataset.waveforms.shape[1]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = PowerElectronicsLoss(
        mse_weight=1.0,
        spectral_weight=0.3,
        derivative_weight=0.2,
        peak_weight=0.5,
        ripple_weight=0.3,
        settling_weight=0.1,
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            params = batch['params'].to(device)
            waveform = batch['waveform'].to(device)
            
            optimizer.zero_grad()
            
            pred = model(params, normalize=True)
            loss, loss_dict = loss_fn(pred, waveform)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        val_mses = []
        
        with torch.no_grad():
            for batch in val_loader:
                params = batch['params'].to(device)
                waveform = batch['waveform'].to(device)
                
                pred = model(params, normalize=True)
                loss, loss_dict = loss_fn(pred, waveform)
                
                val_losses.append(loss.item())
                val_mses.append(loss_dict['mse'])
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_mse = np.mean(val_mses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val MSE={avg_val_mse:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, output_path / 'best_model.pt')
            print(f"  Saved new best model!")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'history': history,
    }, output_path / 'final_model.pt')
    
    # Plot training curves
    plot_training_history(history, output_path / 'training_curves.png')
    
    return model, history


def plot_training_history(history: dict, save_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = axes[1]
    ax2.plot(history['val_mse'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Validation MSE')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_surrogate(
    model: ForwardSurrogate,
    dataset: BuckConverterDataset,
    device: str = 'cpu',
    num_samples: int = 5,
):
    """Evaluate and visualize surrogate predictions."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            params = sample['params'].unsqueeze(0).to(device)
            target = sample['waveform'].numpy()
            
            pred = model(params, normalize=True).cpu().numpy().squeeze()
            
            ax = axes[i] if num_samples > 1 else axes
            ax.plot(target, 'b-', label='Ground Truth (SPICE)', linewidth=2)
            ax.plot(pred, 'r--', label='Surrogate Prediction', linewidth=2)
            ax.set_xlabel('Sample')
            ax.set_ylabel('Voltage (V)')
            ax.legend()
            ax.grid(True)
            
            # Show params in title
            p = sample['params'].numpy()
            ax.set_title(f"L={p[0]*1e6:.1f}µH, C={p[1]*1e6:.0f}µF, "
                        f"R={p[2]:.1f}Ω, Vin={p[3]:.1f}V, "
                        f"fsw={p[4]/1e3:.0f}kHz, D={p[5]:.2f}")
    
    plt.tight_layout()
    plt.savefig('./checkpoints/evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    model, history = train_surrogate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
