# plotting.py
# Builds the graphs from the saved results in results/*
# The parallel-coordinates plot comes from the W&B dashboard.

import os
import csv
import glob
import json
import matplotlib.pyplot as plt

def load_results():
    runs = []
    for path in sorted(glob.glob('results/*.json')):
        with open(path) as f:
            runs.append(json.load(f))
    return runs

def write_summary(runs):
    with open('results/summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_name', 'activation', 'optimizer', 'lr', 'batch_size',
                    'epochs', 'best_val_acc', 'test_acc'])
        for r in runs:
            c = r['config']
            w.writerow([r['run_name'], c['activation'], c['optimizer'], c['lr'],
                        c['batch_size'], c['epochs'],
                        round(r['best_val_acc'], 2), round(r['test_acc'], 2)])

def plot_metric(runs, key, ylabel, filename):
    plt.figure(figsize=(9, 6))
    for r in runs:
        h = r['history']
        plt.plot(h['epoch'], h[key], label=r['run_name'])
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend(fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('plots', filename), dpi=150, bbox_inches='tight')
    plt.close()

def scatter_val_acc(runs, filename):
    plt.figure(figsize=(9, 6))
    for r in runs:
        h = r['history']
        plt.scatter(h['epoch'], h['val_acc'], s=15, label=r['run_name'])
    plt.xlabel('step (epoch)')
    plt.ylabel('validation accuracy (%)')
    plt.legend(fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('plots', filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_single(name):
    path = os.path.join('results', name + '.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        h = json.load(f)['history']

    plt.figure()
    plt.plot(h['epoch'], h['train_acc'], label='train')
    plt.plot(h['epoch'], h['val_acc'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('plots', name + '_acc.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(h['epoch'], h['train_loss'], label='train')
    plt.plot(h['epoch'], h['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('plots', name + '_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs('plots', exist_ok=True)
    runs = load_results()
    # the sweep graphs use the sweep runs only
    sweep = [r for r in runs if r.get('group') not in ('final', 'baseline')]
    write_summary(runs)
    scatter_val_acc(sweep, 'val_acc_vs_step.png')
    plot_metric(sweep, 'train_loss', 'training loss', 'train_loss.png')
    plot_metric(sweep, 'train_acc', 'training accuracy (%)', 'train_acc.png')
    plot_metric(sweep, 'val_loss', 'validation loss', 'val_loss.png')
    plot_metric(sweep, 'val_acc', 'validation accuracy (%)', 'val_acc.png')
    plot_single('baseline')
    plot_single('best_final')

    print('saved graphs to plots/ and results/summary.csv')

if __name__ == '__main__':
    main()
