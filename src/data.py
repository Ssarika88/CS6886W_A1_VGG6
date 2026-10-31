# data.py
# Loads CIFAR-10 and builds train, validation and test loaders.
# The training set is split into train and validation using random_split.

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def get_loaders(batch_size=128, val_split=0.1, num_workers=2, augment=True, seed=42):
    # augmentation for training
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(n_holes=1, length=16),
    ])
    # no augmentation for validation and test
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if not augment:
        train_tf = test_tf

    full_train = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    # split training images into train and validation
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(full_train, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
