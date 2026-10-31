# optim.py
# Builds an optimizer by name, used to get the optimizer.

import torch.optim as optim

def get_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    name = name.lower()
    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'nesterov-sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'nadam':
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer: " + name)
