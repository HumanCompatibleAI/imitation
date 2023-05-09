# Benchmarking imitation

This directory contains Sacred configuration files for benchmarking imitation's algorithms. For v0.3.2, these correspond to the hyperparameters used in the paper [imitation: Clean Imitation Learning Implementations](https://www.rocamonde.com/publication/gleave-imitation-2022/).

Configuration files can be loaded either from the CLI or from the Python API. The examples below assume that your current working directory is the root of the `imitation` repository.

## Single benchmark

To run a single benchmark from the command line:

```bash
python -m imitation.scripts.<train_script> <algo> \
    --name=<name> with benchmarking/<config_name>.json
```

`train_script` can be either 1) `train_imitation` with `algo` as `bc` or `dagger` or 2) `train_adversarial`  with `algo` as `gail` or `airl`.

To view the results:

```bash
python -m imitation.scripts.analyze analyze_imitation with \
    source_dir_str="output/sacred" table_verbosity=0  \
    csv_output_path=results.csv \
    run_name="<name>"
```

To run a single benchmark from Python add the config to your Sacred experiment `ex`:

```python
...
ex.add_config('benchmarking/<config_name>.json')
```

## Entire benchmark suite

### Running locally

To generate the commands to run the entire benchmarking suite with multiple random seeds:

```bash
python experiments/commands.py \
  --name=<name> \
  --cfg_pattern "benchmarking/example_*.json" \
  --seeds 0 1 2 \
  --output_dir=output
```

To run those commands in parallel:

```bash
python experiments/commands.py \
  --name=<name> \
  --cfg_pattern "benchmarking/example_*.json" \
  --seeds 0 1 2 \
  --output_dir=output | parallel -j 8
```

(You may need to `brew install parallel` to get this to work on Mac.)

### Running on Hofvarpnir

To generate the commands for the Hofvarpnir cluster:

```bash
python experiments/commands.py \
  --name=<name> \
  --cfg_pattern "benchmarking/example_*.json" \
  --seeds 0 1 2 \
  --output_dir=/data/output \
  --remote
```

To run those commands pipe them into bash:

```bash
python experiments/commands.py \
  --name <name> \
  --cfg_pattern "benchmarking/example_*.json" \
  --seeds 0 1 2 \
  --output_dir /data/output \
  --remote | bash
```

### Results

To produce a table with all the results:

```bash
python -m imitation.scripts.analyze analyze_imitation with \
    source_dir_str="output/sacred" table_verbosity=0  \
    csv_output_path=results.csv \
    run_name="<name>"
```

To compute a p-value to test whether the differences from the paper are statistically significant:

```bash
python -m imitation.scripts.compare_to_baseline results.csv
```
