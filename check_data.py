#!/usr/bin/env python3
"""Check training data metrics"""
import numpy as np
import os

data_files = [
    'data/extended_topologies_35k.npz',
    'data/multi_topology_data.npz', 
    'data/training_data.npz'
]

for f in data_files:
    if os.path.exists(f):
        data = np.load(f)
        print(f'{f}:')
        print(f'  Keys: {list(data.keys())}')
        if 'metrics' in data.keys():
            m = data['metrics']
            print(f'  Metrics shape: {m.shape}')
            print(f'  Eff range: [{m[:,0].min():.2f}, {m[:,0].max():.2f}]')
            print(f'  Eff mean: {m[:,0].mean():.2f}')
            print(f'  Ripple range: [{m[:,1].min():.2f}, {m[:,1].max():.2f}]')
            print(f'  Ripple mean: {m[:,1].mean():.2f}')
        print()
