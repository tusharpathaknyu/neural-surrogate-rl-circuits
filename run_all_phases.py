#!/usr/bin/env python3
"""
Neural Surrogate + RL for Power Electronics Circuit Design

This script runs all 3 phases:
  Phase A: Generate SPICE training data (ngspice simulations)
  Phase B: Train surrogate model (1D-CNN: params → waveform)  
  Phase C: Train RL agent (PPO learns to design circuits)

The magic:
  - ngspice simulation: ~100ms per circuit
  - Surrogate prediction: ~0.001ms per circuit
  - RL can explore 100,000x more designs!
"""

import subprocess
import sys
from pathlib import Path


def run_phase(phase_name: str, script_path: str) -> bool:
    """Run a phase script and return success status."""
    print("\n" + "="*70)
    print(f"  {phase_name}")
    print("="*70 + "\n")
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(Path(script_path).parent.parent),
    )
    
    return result.returncode == 0


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     NEURAL SURROGATE + RL FOR POWER ELECTRONICS DESIGN              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Phase A: Harvest SPICE data (ngspice simulations)                  ║
║  Phase B: Train surrogate (circuit params → waveform)               ║
║  Phase C: Train RL agent (learn optimal design policy)              ║
║                                                                      ║
║  Result: Instant circuit design that would take hours with SPICE!   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    project_root = Path(__file__).parent
    
    phases = [
        ("PHASE A: Generate SPICE Training Data", 
         project_root / "data" / "generate_spice_data.py"),
        
        ("PHASE B: Train Surrogate Model",
         project_root / "training" / "train_surrogate.py"),
        
        ("PHASE C: Train RL Agent",
         project_root / "rl" / "train_agent.py"),
    ]
    
    for phase_name, script_path in phases:
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            sys.exit(1)
        
        success = run_phase(phase_name, str(script_path))
        
        if not success:
            print(f"\n❌ {phase_name} FAILED")
            print("   Fix the error and re-run this script.")
            sys.exit(1)
        
        print(f"\n✓ {phase_name.split(':')[0]} Complete!")
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        ALL PHASES COMPLETE!                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Your RL agent can now design power electronics circuits instantly! ║
║                                                                      ║
║  Files created:                                                      ║
║    - data/spice_dataset.npz      (SPICE simulation data)           ║
║    - checkpoints/surrogate_best.pt (trained surrogate)              ║
║    - checkpoints/rl_agent.pt      (trained RL policy)               ║
║                                                                      ║
║  To use the trained agent:                                           ║
║    from rl import PPOAgent, CircuitDesignEnv                        ║
║    agent.load('checkpoints/rl_agent.pt')                            ║
║    result = agent.design_circuit(target_waveform)                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
