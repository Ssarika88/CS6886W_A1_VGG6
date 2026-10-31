# experiments.py
# Runs Model Performance on Different Configurations by calling train.py.
# Skips a run if its results file already exists, so it can be stopped and restarted.

import os
import sys
import subprocess

EPOCHS = 15

# each experiment is a list of extra flags for train.py
experiments = []

# activation function
for act in ['relu', 'sigmoid', 'tanh', 'silu', 'gelu']:
    experiments.append(['--activation', act])

# optimizer (with a learning rate that suits each one)
opt_lr = {'sgd': 0.05, 'nesterov-sgd': 0.05, 'adam': 0.001,
          'adagrad': 0.01, 'rmsprop': 0.001, 'nadam': 0.001}
for opt, lr in opt_lr.items():
    experiments.append(['--optimizer', opt, '--lr', str(lr)])

# learning rate
for lr in [0.01, 0.001, 0.0001]:
    experiments.append(['--lr', str(lr)])

# batch size
for bs in [64, 128, 256]:
    experiments.append(['--batch-size', str(bs)])

def run_name_from_flags(flags):
    act, opt, lr, bs = 'relu', 'adam', '0.001', '128'   # defaults
    for i in range(0, len(flags), 2):
        key, val = flags[i], flags[i + 1]
        if key == '--activation':
            act = val
        elif key == '--optimizer':
            opt = val
        elif key == '--lr':
            lr = val
        elif key == '--batch-size':
            bs = val
    return act + '_' + opt + '_lr' + lr + '_bs' + bs

done = set()
for flags in experiments:
    name = run_name_from_flags(flags)
    if name in done:
        continue
    done.add(name)

    result_file = os.path.join('results', name + '.json')
    if os.path.exists(result_file):
        print('skip', name)
        continue

    cmd = [sys.executable, '-m', 'src.train', '--epochs', str(EPOCHS)] + flags
    print('running', name)
    subprocess.run(cmd)
