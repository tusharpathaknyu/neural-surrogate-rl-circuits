#!/bin/bash
# Monitor data generation and automatically start surrogate retraining when done.
# Usage: bash monitor_and_retrain.sh

cd /Users/tushardhananjaypathak/Desktop/MLEntry
source .venv/bin/activate

DATAGEN_PID=15812
LOG_FILE="data_generation.log"
RETRAIN_LOG="surrogate_retrain.log"

echo "=== Monitoring data generation (PID $DATAGEN_PID) ==="
echo "Started at: $(date)"
echo ""

# Wait for data gen to finish
while kill -0 $DATAGEN_PID 2>/dev/null; do
    PROGRESS=$(tail -1 "$LOG_FILE" 2>/dev/null | sed -n 's/.*\([0-9]\{1,3\}%\).*/\1/p' | tail -1)
    TOPO=$(tail -10 "$LOG_FILE" 2>/dev/null | grep "Generating" | tail -1 | sed 's/.*for //')
    echo "[$(date +%H:%M:%S)] Data gen running — $TOPO: $PROGRESS"
    sleep 60
done

echo ""
echo "=== Data generation finished at $(date) ==="
echo ""

# Check the log for success
if tail -20 "$LOG_FILE" | grep -q "combined_dataset.npz"; then
    echo "✓ Dataset saved successfully"
else
    echo "⚠ WARNING: Could not confirm dataset was saved. Check $LOG_FILE"
    echo "Last 10 lines:"
    tail -10 "$LOG_FILE"
fi

# Verify dataset
echo ""
echo "=== Verifying dataset ==="
python3 -c "
import numpy as np
data = np.load('data/extended_topologies/combined_dataset.npz')
print(f'  params:     {data[\"params\"].shape}')
print(f'  waveforms:  {data[\"waveforms\"].shape}')
print(f'  topologies: {data[\"topologies\"].shape}')
print(f'  topology dist: {dict(zip(*np.unique(data[\"topologies\"], return_counts=True)))}')
# Verify all 7 topologies present
n_topos = len(np.unique(data['topologies']))
assert n_topos == 7, f'Expected 7 topologies, got {n_topos}'
# Verify 6 params per sample
assert data['params'].shape[1] == 6, f'Expected 6 params, got {data[\"params\"].shape[1]}'
print(f'  ✓ All checks passed ({n_topos} topologies, {data[\"params\"].shape[1]} params)')
"

if [ $? -ne 0 ]; then
    echo "✗ Dataset verification failed! Aborting."
    exit 1
fi

# Backup old checkpoint
echo ""
echo "=== Backing up old checkpoint ==="
if [ -f checkpoints/multi_topology_surrogate.pt ]; then
    cp checkpoints/multi_topology_surrogate.pt checkpoints/multi_topology_surrogate_v1_broken.pt
    echo "  ✓ Backed up to multi_topology_surrogate_v1_broken.pt"
fi

# Retrain surrogate
echo ""
echo "=== Starting surrogate retraining ==="
echo "Log: $RETRAIN_LOG"
echo "Started at: $(date)"
python3 -u models/train_multi_topology.py 2>&1 | tee "$RETRAIN_LOG"

echo ""
echo "=== Surrogate retraining finished at $(date) ==="

# Validate new checkpoint
echo ""
echo "=== Validating new checkpoint ==="
python3 -c "
import torch
ckpt = torch.load('checkpoints/multi_topology_surrogate.pt', map_location='cpu', weights_only=False)
print(f'  epoch:          {ckpt[\"epoch\"]}')
print(f'  val_loss:       {ckpt[\"val_loss\"]:.6f}')
print(f'  waveform_stats: {\"yes\" if \"waveform_stats\" in ckpt else \"NO — BUG!\"}')
print(f'  param_stats:    {\"yes\" if \"param_stats\" in ckpt else \"NO — BUG!\"}')
if 'waveform_stats' in ckpt:
    ws = ckpt['waveform_stats']
    for tid, stats in sorted(ws.items(), key=lambda x: int(x[0])):
        print(f'    topo {tid}: mean={stats[\"mean\"]:8.3f}V, std={stats[\"std\"]:6.3f}V')
"

echo ""
echo "=== ALL DONE ==="
echo "To start RL training:"
echo "  nohup .venv/bin/python -u train_intensive_spice.py > training_all_topologies.log 2>&1 &"
