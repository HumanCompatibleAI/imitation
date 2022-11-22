# Benchmarking imitation

This directory contains sacred configuration files for benchmarking imitation's algorithms. For v0.3.2, these correspond to the hyperparameters used in the paper [imitation: Clean Imitation Learning Implementations](https://www.rocamonde.com/publication/gleave-imitation-2022/).

Configuration files can be loaded either from the CLI or from the Python API. The examples below assume that your current working directory is the root of the `imitation` repository. This is not necessarily the case and you should point your paths accordingly.

## CLI

```bash
python -m imitation.scripts.train_rl with benchmarking/<config_name>.json
```

## Python

```python
...
ex.add_config('benchmarking/<config_name>.json')
```
