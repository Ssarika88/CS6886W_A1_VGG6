# utils.py
# Helper functions- set the random seed and save/load the model.

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(path, model, config, best_acc):
    torch.save({'model_state': model.state_dict(),
                'config': config,
                'best_acc': best_acc}, path)

def load_checkpoint(path, model, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])
    return ckpt