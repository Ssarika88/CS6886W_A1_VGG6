# train.py
# Train VGG6 on CIFAR-10 for one configuration.
# Baseline: python -m src.train
# Experiments: python -m src.train --optimizer rmsprop --epochs 40

import os
import json
import argparse
import torch
import torch.nn as nn
import wandb

from .data import get_loaders
from .model import vgg, cfg_vgg6
from .optim import get_optimizer
from .engine import train_one_epoch, evaluate
from .utils import set_seed, save_checkpoint, load_checkpoint

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--activation', type=str, default='relu')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--no-batch-norm', action='store_true')
    p.add_argument('--group', type=str, default='sweep')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--save-dir', type=str, default='checkpoints')
    p.add_argument('--results-dir', type=str, default='results')
    p.add_argument('--wandb-project', type=str, default='vgg6-cifar10')
    p.add_argument('--wandb-mode', type=str, default='offline')
    return p.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_name = args.run_name
    if run_name is None:
        run_name = args.activation + '_' + args.optimizer + '_lr' + str(args.lr) + '_bs' + str(args.batch_size)

    wandb.init(project=args.wandb_project, name=run_name, config=vars(args), mode=args.wandb_mode)

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size, val_split=args.val_split,
        num_workers=args.num_workers, augment=not args.no_augment, seed=args.seed)

    model = vgg(cfg_vgg6, num_classes=10, activation=args.activation,
                batch_norm=not args.no_batch_norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, run_name + '.pth')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc}, step=epoch)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print("epoch %d/%d  train_loss %.4f  train_acc %.2f  val_loss %.4f  val_acc %.2f"
              % (epoch, args.epochs, train_loss, train_acc, val_loss, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ckpt_path, model, vars(args), best_val_acc)

    load_checkpoint(ckpt_path, model, map_location=device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("best_val_acc %.2f  test_acc %.2f" % (best_val_acc, test_acc))

    wandb.run.summary['best_val_acc'] = best_val_acc
    wandb.run.summary['test_acc'] = test_acc

    os.makedirs(args.results_dir, exist_ok=True)
    result = {'run_name': run_name, 'group': args.group, 'config': vars(args),
              'best_val_acc': best_val_acc, 'test_acc': test_acc, 'history': history}
    with open(os.path.join(args.results_dir, run_name + '.json'), 'w') as f:
        json.dump(result, f, indent=2)

    wandb.finish()

if __name__ == '__main__':
    main()
