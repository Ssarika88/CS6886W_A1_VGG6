# evaluate.py
# Load a saved model and report its test accuracy.

import argparse
import torch
import torch.nn as nn

from .data import get_loaders
from .model import vgg, cfg_vgg6
from .engine import evaluate
from .utils import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt['config']
    set_seed(cfg.get('seed', 42))

    model = vgg(cfg_vgg6, num_classes=10,
                activation=cfg.get('activation', 'relu'),
                batch_norm=not cfg.get('no_batch_norm', False)).to(device)
    model.load_state_dict(ckpt['model_state'])

    _, _, test_loader = get_loaders(batch_size=args.batch_size,
                                    val_split=cfg.get('val_split', 0.1),
                                    num_workers=args.num_workers,
                                    augment=not cfg.get('no_augment', False),
                                    seed=cfg.get('seed', 42))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("test_loss %.4f  test_acc %.2f" % (test_loss, test_acc))

if __name__ == '__main__':
    main()
