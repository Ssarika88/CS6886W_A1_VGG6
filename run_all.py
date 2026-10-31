# run_all.py
# Single command to run VGG6 on CIFAR-10 with different configurations:
# Usage: python run_all.py

import sys
import os
import json
import glob
import subprocess

PY = sys.executable

def run(cmd):
    print('\n>>>', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def best_sweep_config():
    # pick the sweep run with the highest validation accuracy
    best = None
    for path in glob.glob('results/*.json'):
        with open(path) as f:
            r = json.load(f)
        if r.get('group') != 'sweep':
            continue
        if best is None or r['best_val_acc'] > best['best_val_acc']:
            best = r
    return best

def main():
    # 1. baseline (its own name/group so it does not clash with the sweep)
    run([PY, '-m', 'src.train', '--run-name', 'baseline', '--group', 'baseline'])

    # 2. all experiments (activations, optimizers, learning rates, batch sizes)
    run([PY, '-m', 'src.experiments'])

    # 3. graphs from the sweep
    run([PY, '-m', 'src.plotting'])

    # 4. best configuration from the sweep, trained longer, then verified
    best = best_sweep_config()
    c = best['config']
    print('best sweep config:', best['run_name'], 'val', best['best_val_acc'])
    run([PY, '-m', 'src.train',
         '--activation', str(c['activation']),
         '--optimizer', str(c['optimizer']),
         '--lr', str(c['lr']),
         '--batch-size', str(c['batch_size']),
         '--epochs', '40', '--group', 'final', '--run-name', 'best_final'])
    run([PY, '-m', 'src.evaluate', '--checkpoint', os.path.join('checkpoints', 'best_final.pth')])

    # rebuilding graphs to include
    run([PY, '-m', 'src.plotting'])
    print('\nAll steps finished.')

if __name__ == '__main__':
    main()