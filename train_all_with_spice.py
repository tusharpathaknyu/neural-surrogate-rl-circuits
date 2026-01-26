#!/usr/bin/env python3
"""
Complete Training Pipeline for All 7 Topologies with SPICE Validation.

This script:
1. Checks that all data is available (including QR Flyback)
2. Retrains the multi-topology surrogate model
3. Trains all 7 per-topology RL agents with SPICE validation

Run: python train_all_with_spice.py [--quick] [--skip-surrogate]
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import argparse
import subprocess
import time
from datetime import datetime
import json

# All 7 topologies
TOPOLOGIES = ['buck', 'boost', 'buck_boost', 'sepic', 'cuk', 'flyback', 'qr_flyback']

def check_prerequisites():
    """Check that all required data and tools are available."""
    print("\n" + "=" * 70)
    print("Checking Prerequisites")
    print("=" * 70)
    
    issues = []
    
    # Check ngspice
    try:
        result = subprocess.run(['ngspice', '--version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✓ ngspice available")
        else:
            issues.append("ngspice not working properly")
    except:
        issues.append("ngspice not installed")
    
    # Check data files
    data_dir = Path(__file__).parent / 'data' / 'spice_data'
    combined_path = data_dir / 'combined_dataset.npz'
    qr_path = data_dir / 'qr_flyback_dataset.npz'
    
    if combined_path.exists():
        import numpy as np
        data = np.load(combined_path)
        print(f"✓ Combined dataset: {len(data['params'])} samples")
        print(f"  Topologies: {list(data['topology_names'])}")
    else:
        issues.append(f"Combined dataset not found at {combined_path}")
    
    if qr_path.exists():
        import numpy as np
        data = np.load(qr_path)
        print(f"✓ QR Flyback dataset: {len(data['params'])} samples")
    else:
        issues.append(f"QR Flyback dataset not found at {qr_path}")
    
    # Check model checkpoint
    model_path = Path(__file__).parent / 'checkpoints' / 'multi_topology_surrogate.pt'
    if model_path.exists():
        print(f"✓ Existing surrogate checkpoint found")
    else:
        print("⚠ No existing surrogate checkpoint (will train from scratch)")
    
    # Check existing RL agents
    for topo in TOPOLOGIES:
        agent_path = Path(__file__).parent / 'checkpoints' / f'rl_agent_{topo}.pt'
        if agent_path.exists():
            print(f"  ✓ {topo} agent checkpoint exists")
        else:
            print(f"  ⚠ {topo} agent not trained yet")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("\n✓ All prerequisites satisfied!")
    return True


def retrain_surrogate(quick_mode: bool = False):
    """Retrain the multi-topology surrogate model."""
    print("\n" + "=" * 70)
    print("Retraining Multi-Topology Surrogate Model")
    print("=" * 70)
    
    # Update num_topologies to 7
    cmd = ['python', 'models/train_multi_topology.py']
    if quick_mode:
        cmd.extend(['--epochs', '50'])
    
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    elapsed = time.time() - start
    print(f"\nSurrogate training completed in {elapsed/60:.1f} minutes")
    
    return result.returncode == 0


def train_topology_agent(topology: str, quick_mode: bool = False, use_spice: bool = True):
    """Train a single topology's RL agent."""
    print(f"\n{'='*70}")
    print(f"Training {topology.upper()} Agent")
    print("=" * 70)
    
    cmd = [
        'python', 'rl/train_per_topology_agents.py',
        '--topology', topology,
    ]
    
    if quick_mode:
        cmd.append('--quick')
    
    if use_spice:
        cmd.extend(['--spice', '--spice-freq', '100'])
    
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    elapsed = time.time() - start
    print(f"\n{topology} training completed in {elapsed/60:.1f} minutes")
    
    return result.returncode == 0


def train_all_agents(quick_mode: bool = False, use_spice: bool = True):
    """Train all 7 RL agents sequentially."""
    print("\n" + "=" * 70)
    print("Training All 7 Per-Topology RL Agents")
    print("=" * 70)
    print(f"SPICE validation: {'Enabled' if use_spice else 'Disabled'}")
    print(f"Quick mode: {'Yes' if quick_mode else 'No (full training)'}")
    
    results = {}
    total_start = time.time()
    
    for i, topology in enumerate(TOPOLOGIES):
        print(f"\n[{i+1}/{len(TOPOLOGIES)}] Training {topology}...")
        success = train_topology_agent(topology, quick_mode, use_spice)
        results[topology] = 'success' if success else 'failed'
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    
    for topo, status in results.items():
        icon = "✓" if status == 'success' else "✗"
        print(f"  {icon} {topo}: {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Complete training pipeline for all 7 topologies')
    parser.add_argument('--quick', action='store_true', help='Quick mode with reduced iterations')
    parser.add_argument('--skip-surrogate', action='store_true', help='Skip surrogate retraining')
    parser.add_argument('--no-spice', action='store_true', help='Disable SPICE validation')
    parser.add_argument('--topology', type=str, default=None, help='Train only a specific topology')
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPLETE TRAINING PIPELINE")
    print("7 Topologies with SPICE Validation")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quick mode: {args.quick}")
    print(f"SPICE validation: {not args.no_spice}")
    print(f"Topologies: {', '.join(TOPOLOGIES)}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix issues and retry.")
        return 1
    
    results = {
        'start_time': datetime.now().isoformat(),
        'quick_mode': args.quick,
        'spice_enabled': not args.no_spice,
        'surrogate_trained': False,
        'agent_results': {},
    }
    
    # Step 1: Retrain surrogate (unless skipped)
    if not args.skip_surrogate:
        success = retrain_surrogate(args.quick)
        results['surrogate_trained'] = success
        if not success:
            print("\n⚠ Surrogate training failed, but continuing with agents...")
    else:
        print("\n⚠ Skipping surrogate retraining (--skip-surrogate)")
    
    # Step 2: Train RL agents
    if args.topology:
        # Train single topology
        success = train_topology_agent(args.topology, args.quick, not args.no_spice)
        results['agent_results'][args.topology] = 'success' if success else 'failed'
    else:
        # Train all topologies
        results['agent_results'] = train_all_agents(args.quick, not args.no_spice)
    
    results['end_time'] = datetime.now().isoformat()
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results saved to: training_results.json")
    print(f"\nTo test agents:")
    print(f"  python rl/train_per_topology_agents.py --test-only")
    print(f"\nTo run web demo:")
    print(f"  python web_demo_enhanced.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
