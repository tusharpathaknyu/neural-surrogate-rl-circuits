"""Check surrogate checkpoint details."""
import torch, os, sys
from datetime import datetime

sys.path.insert(0, '.')

ckpt_path = 'checkpoints/multi_topology_surrogate.pt'
mod_time = os.path.getmtime(ckpt_path)
print(f'Checkpoint modified: {datetime.fromtimestamp(mod_time)}')
print(f'File size: {os.path.getsize(ckpt_path) / 1e6:.1f} MB')

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f'Keys: {list(ckpt.keys())}')
print(f'Val loss: {ckpt.get("val_loss", "N/A")}')
print(f'Epoch: {ckpt.get("epoch", "N/A")}')
print(f'Has waveform_stats: {"waveform_stats" in ckpt}')
print(f'Has param_stats: {"param_stats" in ckpt}')

if 'waveform_stats' in ckpt:
    ws = ckpt['waveform_stats']
    print(f'waveform_stats type: {type(ws)}')
    if isinstance(ws, dict):
        print(f'waveform_stats topologies: {list(ws.keys())}')
        for k in list(ws.keys())[:2]:
            print(f'  {k}: mean={ws[k]["mean"]:.4f}, std={ws[k]["std"]:.4f}')

# Check data files
import glob
data_files = sorted(glob.glob('data/extended_topologies/*_data.npz'))
print(f'\nData files:')
for f in data_files:
    mt = os.path.getmtime(f)
    sz = os.path.getsize(f) / 1e6
    print(f'  {os.path.basename(f)}: modified={datetime.fromtimestamp(mt)}, size={sz:.1f}MB')

# Check combined dataset
combined = glob.glob('data/extended_topologies/combined_*.npz')
for f in sorted(combined):
    mt = os.path.getmtime(f)
    print(f'  {os.path.basename(f)}: modified={datetime.fromtimestamp(mt)}')

# Check model param count
from models.multi_topology_surrogate import MultiTopologySurrogate
model = MultiTopologySurrogate(num_topologies=7)
model.load_state_dict(ckpt['model_state_dict'])
n_params = sum(p.numel() for p in model.parameters())
print(f'\nModel params: {n_params:,}')
