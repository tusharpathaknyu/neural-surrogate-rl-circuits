#!/bin/bash
# =============================================================================
# AUTOMATED PIPELINE: Retrain surrogate + Restart RL training
# =============================================================================
# Run this after data generation completes:
#   bash retrain_and_run.sh
#
# Steps:
#   1. Verify data generation completed (combined_dataset.npz exists and is recent)
#   2. Backup old surrogate checkpoint
#   3. Retrain surrogate on new data 
#   4. Verify new checkpoint has waveform_stats
#   5. Start RL training with new surrogate
# =============================================================================

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "============================================================"
echo "RETRAIN & RUN PIPELINE"
echo "============================================================"
echo "Time: $(date)"
echo ""

# 1. Check data
DATASET="data/extended_topologies/combined_dataset.npz"
if [ ! -f "$DATASET" ]; then
    echo "ERROR: $DATASET not found. Run data generation first:"
    echo "  python data/generate_extended_topologies.py"
    exit 1
fi

DATASET_TIME=$(stat -f %m "$DATASET")
NOW=$(date +%s)
AGE=$(( (NOW - DATASET_TIME) / 3600 ))
echo "Dataset: $DATASET (${AGE}h old)"

# Quick sanity check
python -c "
import numpy as np
d = np.load('$DATASET')
params = d['params']
topos = d['topologies']
print(f'  Samples: {len(params)}')
print(f'  Params shape: {params.shape}')
print(f'  Topologies: {np.bincount(topos)}')
# Verify all params have f_sw and duty (cols 4,5 should never be 0)
for i in range(7):
    mask = topos == i
    fsw_col = params[mask, 4]
    duty_col = params[mask, 5]
    assert fsw_col.min() > 0, f'Topo {i}: f_sw is zero! Param mapping broken'
    assert duty_col.min() > 0, f'Topo {i}: duty is zero! Param mapping broken'
print('  ✓ All topologies have valid f_sw and duty columns')
"

echo ""

# 2. Backup old checkpoint
CKPT="checkpoints/multi_topology_surrogate.pt"
if [ -f "$CKPT" ]; then
    BACKUP="${CKPT%.pt}_v1_old_denorm.pt"
    cp "$CKPT" "$BACKUP"
    echo "Backed up old checkpoint → $BACKUP"
fi

echo ""

# 3. Retrain surrogate
echo "============================================================"
echo "RETRAINING SURROGATE (100 epochs)"
echo "============================================================"
python -u models/train_multi_topology.py 2>&1 | tee surrogate_training.log

echo ""

# 4. Verify new checkpoint
python -c "
import torch
ckpt = torch.load('$CKPT', map_location='cpu', weights_only=False)
print(f'  Epoch: {ckpt[\"epoch\"]}')
print(f'  Val loss: {ckpt[\"val_loss\"]:.6f}')
has_stats = 'waveform_stats' in ckpt
print(f'  Has waveform_stats: {has_stats}')
if has_stats:
    for k, v in ckpt['waveform_stats'].items():
        print(f'    Topo {k}: mean={v[\"mean\"]:.3f}V, std={v[\"std\"]:.3f}V')
assert has_stats, 'Checkpoint missing waveform_stats!'
print('  ✓ Checkpoint valid')
"

echo ""

# 5. Start RL training
echo "============================================================"
echo "STARTING RL TRAINING WITH NEW SURROGATE"
echo "============================================================"
nohup python -u train_intensive_spice.py > training_all_topologies.log 2>&1 &
RL_PID=$!
echo "RL training started (PID: $RL_PID)"
echo "Monitor: tail -f training_all_topologies.log"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Time: $(date)"
