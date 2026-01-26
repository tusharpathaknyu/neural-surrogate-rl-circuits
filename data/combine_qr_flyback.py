#!/usr/bin/env python3
"""
Combine QR Flyback data with existing 6-topology dataset to create 7-topology dataset.
Recreates from individual files to avoid duplicates.
"""
import numpy as np
from pathlib import Path

def main():
    print("=" * 70)
    print("Creating 7-topology combined dataset")
    print("=" * 70)
    
    data_dir = Path(__file__).parent / 'extended_topologies'
    
    # Individual topology datasets
    topologies = ['BUCK', 'BOOST', 'BUCK_BOOST', 'SEPIC', 'CUK', 'FLYBACK']
    files = ['buck_data.npz', 'boost_data.npz', 'buck_boost_data.npz', 
             'sepic_data.npz', 'cuk_data.npz', 'flyback_data.npz']
    
    all_waveforms = []
    all_params = []
    all_topologies = []
    
    for i, (name, fname) in enumerate(zip(topologies, files)):
        d = np.load(data_dir / fname)
        all_waveforms.append(d['waveforms'])
        # Normalize params to 6 dimensions [L, C, R_load, V_in, f_sw, duty]
        params = d['params']
        if params.shape[1] > 6:
            params = params[:, :6]  # Truncate to first 6 params
        elif params.shape[1] < 6:
            # Pad with zeros
            padding = np.zeros((params.shape[0], 6 - params.shape[1]))
            params = np.hstack([params, padding])
        all_params.append(params.astype(np.float32))
        all_topologies.append(np.full(d['waveforms'].shape[0], i, dtype=np.int32))
        print(f"  {name}: {d['waveforms'].shape[0]} samples, params normalized to {params.shape[1]}")
    
    # Add QR Flyback (index 6)
    qr_path = Path(__file__).parent / 'spice_data' / 'qr_flyback_dataset.npz'
    qr_data = np.load(qr_path)
    all_waveforms.append(qr_data['waveforms'])
    all_params.append(qr_data['params'])
    all_topologies.append(np.full(qr_data['waveforms'].shape[0], 6, dtype=np.int32))
    topologies.append('QR_FLYBACK')
    print(f"  QR_FLYBACK: {qr_data['waveforms'].shape[0]} samples")
    
    # Combine
    waveforms = np.vstack(all_waveforms)
    params = np.vstack(all_params)
    topology_ids = np.concatenate(all_topologies)
    
    print(f"\nTotal: {waveforms.shape[0]} samples, {len(topologies)} topologies")
    print(f"Topologies: {topologies}")
    
    # Save to extended_topologies folder
    output_path = data_dir / 'combined_dataset.npz'
    np.savez_compressed(
        output_path,
        waveforms=waveforms,
        params=params,
        topologies=topology_ids,
        topology_names=topologies
    )
    print(f"\nSaved to {output_path}")
    
    # Also save to spice_data for backup
    backup_path = Path(__file__).parent / 'spice_data' / 'combined_dataset.npz'
    np.savez_compressed(
        backup_path,
        waveforms=waveforms,
        params=params,
        topologies=topology_ids,
        topology_names=topologies
    )
    print(f"Backup saved to {backup_path}")
    
    print("\n" + "=" * 70)
    print("7-topology dataset ready for training!")
    print("=" * 70)

if __name__ == '__main__':
    main()
