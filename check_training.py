#!/usr/bin/env python3
"""Quick status check for training."""
import subprocess
import json
from pathlib import Path

print("="*60)
print("TRAINING STATUS")
print("="*60)

# Check if process is running
result = subprocess.run(
    "ps aux | grep train_intensive | grep -v grep | wc -l",
    shell=True, capture_output=True, text=True
)
running = int(result.stdout.strip()) > 0
print(f"Training running: {'✓ Yes' if running else '✗ No'}")

# Check log file
log_file = Path("training_intensive.log")
if log_file.exists():
    with open(log_file) as f:
        lines = f.readlines()
    
    # Find last progress line
    for line in reversed(lines):
        if "|" in line and "%" in line:
            print(f"\nCurrent progress: {line.strip()[:80]}")
            break
    
    # Check completed topologies
    completed = []
    for line in lines:
        if "✓" in line and "Best MSE" in line:
            completed.append(line.strip())
    
    if completed:
        print(f"\nCompleted ({len(completed)}/7):")
        for c in completed:
            print(f"  {c}")

# Check saved checkpoints
print("\nSaved agents:")
for f in sorted(Path("checkpoints").glob("rl_agent_*.pt")):
    import os
    mtime = os.path.getmtime(f)
    from datetime import datetime
    time_str = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
    print(f"  {f.name} (updated {time_str})")

# Check progress file
progress_file = Path("checkpoints/training_progress.json")
if progress_file.exists():
    with open(progress_file) as f:
        progress = json.load(f)
    print(f"\nProgress file: {len(progress.get('completed', []))}/7 topologies saved")
