#!/bin/bash
# Train remaining topologies: SEPIC, Cuk, Flyback
# Run this after Buck-Boost training completes

cd /Users/tushardhananjaypathak/Desktop/MLEntry

echo "=========================================="
echo "Starting SEPIC training (~3 hours)..."
echo "=========================================="
python rl/train_per_topology_agents.py --topology sepic

echo ""
echo "=========================================="
echo "Starting Cuk training (~1.5 hours)..."
echo "=========================================="
python rl/train_per_topology_agents.py --topology cuk

echo ""
echo "=========================================="
echo "Starting Flyback training (~3 hours)..."
echo "=========================================="
python rl/train_per_topology_agents.py --topology flyback

echo ""
echo "=========================================="
echo "ALL TRAINING COMPLETE!"
echo "=========================================="
echo "Trained agents saved in checkpoints/"
ls -la checkpoints/rl_agent_*.pt
