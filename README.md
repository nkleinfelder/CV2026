# CV2026

See [`docs/overview.md`](docs/overview.md) for a repository overview and
[`docs/run-commands.md`](docs/run-commands.md) for `train.py` usage.

## Setting up Symlink (Quota Management)

To address storage quota issues, create a symbolic link for your virtual environment:

```bash
ln -s /scratch/cvcdt011/.venv .venv
```

## Hyperparameters

- half_size: Used for preprocessing, the area around the bee that is cut out
