# JEPA Collapse Diagnostics

## Frozen hypothesis

A JEPA trained in representation space will learn higher-quality semantic
representations than an autoencoder with comparable capacity while resisting
representation collapse. The primary metric is centered effective rank; feature
variance and average pairwise cosine similarity are secondary diagnostics.

The autoencoder reconstructs the same 392-pixel context used by JEPA. Its 98,254
trainable parameters are closely matched to JEPA's 97,956 trainable parameters.

The collapsed control is a negative control: it always emits the same vector,
so it should have near-zero variance, effective rank near one (or zero for the
centered covariance), cosine similarity near one, and chance-level linear-probe
accuracy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Or, with `uv`:

```bash
uv sync
```

The first run downloads MNIST into `data/` through torchvision.

## Run

```bash
python run_experiment.py --epochs 3 --probe-epochs 2
```

With `uv`, use `uv run` instead:

```bash
uv run python run_experiment.py --epochs 3 --probe-epochs 2
```

For a quick smoke run:

```bash
python run_experiment.py --epochs 1 --probe-epochs 1 --max-train-samples 2048 --max-test-samples 1024
```

Results are written as JSON to a UTC timestamped directory with microseconds in
`results/`; an additional suffix prevents collisions. Each report records the
seed, deterministic settings, run configuration, and parameter counts.