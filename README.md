# VGG6 on CIFAR-10

Training VGG6 on CIFAR-10 and studying how the activation function, optimizer,
learning rate, and batch size affect performance. Runs are tracked with Weights
& Biases, and all graphs are generated from the saved results.

All settings are command-line flags (no config files), so every experiment is
just `python -m src.train` with different flags.

## Folder layout

```
vgg6-cifar10/
├── README.md
├── requirements.txt
├── run_all.py          # runs all steps (baseline, sweep, plots, best model)
├── src/
│   ├── __init__.py
│   ├── data.py         # CIFAR-10 + transforms + train/val/test loaders
│   ├── model.py        # VGG6 (activation configurable)
│   ├── optim.py        # builds the optimizer by name
│   ├── engine.py       # train-one-epoch and evaluate loops
│   ├── utils.py        # seed + save/load model
│   ├── train.py        # MAIN: train one configuration
│   ├── evaluate.py     # load a checkpoint and report test accuracy
│   ├── experiments.py  # run all the sweep configurations
│   └── plotting.py     # build the graphs from results/
├── data/           # CIFAR-10 (downloaded automatically on first run)
├── results/        # one JSON per run (+ summary.csv) -> used for plots
├── plots/          # generated graphs
├── checkpoints/    # saved models (best_final.pth is the final model)
```

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision numpy wandb matplotlib
```

CIFAR-10 downloads automatically on the first run into `data/`. (If a machine
has no internet, drop `cifar-10-python.tar.gz` into `data/` and it will be used.)

## How to run

All commands are run from the project root, with the virtual environment active.

### Run everything at once
```powershell
python run_all.py
```
This runs the baseline, the full sweep, the graphs, and the final model + verification in order. Or run the steps individually below.

### Q1 — baseline
Defaults are the baseline (ReLU + Adam, lr 0.001, batch 128, 20 epochs, seed 42):
```powershell
python -m src.train --run-name baseline --group baseline
```

### Q2 — configuration sweep
Runs every configuration (activations, optimizers, learning rates, batch sizes):
```powershell
python -m src.experiments
```
Or run a single experiment by hand, e.g.:
```powershell
python -m src.train --activation gelu
python -m src.train --optimizer rmsprop
python -m src.train --lr 0.01
python -m src.train --batch-size 64
```

### Q3 — graphs
After the sweep, build all the plots and `results/summary.csv`:
```powershell
python -m src.plotting
```

### Q4 — final best model
Train the best configuration longer, then verify it reproduces:
```powershell
python -m src.train --optimizer rmsprop --lr 0.001 --batch-size 128 --epochs 40 --group final --run-name best_final
python -m src.evaluate --checkpoint checkpoints/best_final.pth
```

## Reproducibility (Q5)

- Every run uses `--seed 42` (default); `set_seed` fixes Python/NumPy/PyTorch and
  makes cuDNN deterministic.
- The exact commands above reproduce every result.
- Dependency versions are what `pip` installs on your machine

## Weights & Biases

Runs log in offline mode by default (no account needed), saved under `wandb/`.
To use the online dashboard (parallel-coordinates panel, etc.):
```powershell
wandb login
python -m src.train --wandb-mode online
# or upload existing offline runs:
wandb sync wandb/offline-run-*
```
