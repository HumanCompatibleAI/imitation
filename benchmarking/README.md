# Benchmarking imitation

This directory contains sacred configuration files for benchmarking imitation's algorithms. For v0.3.2, these correspond to the hyperparameters used in the paper [imitation: Clean Imitation Learning Implementations](https://www.rocamonde.com/publication/gleave-imitation-2022/).

Configuration files can be loaded either from the CLI or from the Python API. The examples below assume that your current working directory is the root of the `imitation` repository. This is not necessarily the case and you should adjust your paths accordingly.

To run a single benchmark from the command line:

```bash
python -m imitation.scripts.<train_script> <algo> with benchmarking/<config_name>.json
```

`train_script` can be either 1) `train_imitation` with `algo` as `bc` or `dagger` or 2) `train_adversarial`  with `algo` as `gail` or `airl`.

To run a single benchmark from Python add the config to your experiment:

```python
...
ex.add_config('benchmarking/<config_name>.json')
```

To generate the commands to run the entire benchmarking suite with multiple random seeds:

```bash
python experiments/commands.py \
  --name=run0 \
  --cfg_pattern=benchmarking/example_*.json \
  --seeds 0,1,2 \
  --output_dir=output
```

To run those commands in parallel:

```bash
python experiments/commands.py ... | parallel -j 8
```

To generate the commands for the Hofvarpnir cluster:

```bash
python experiments/commands.py \
  --name=run0 \
  --cfg_pattern=benchmarking/example_*.json \
  --seeds 0,1,2 \
  --output_dir=/data/output \
  --remote
```

To run those commands pipe them into bash:

```bash
python experiments/commands.py ... | bash
```

To produce a table with all the results:

```bash
python -m imitation.scripts.analyze analyze_imitation with \
    source_dir_str="output/sacred" table_verbosity=0  \
    csv_output_path=results.csv \
    run_name="run0"
```

To compute a p-value to test whether the differences from the paper are statistically significant:

```python
import pandas as pd
import numpy as np
import scipy

data = pd.read_csv("results.csv")
data["imit_return"] = data["imit_return_summary"].apply(lambda x: float(x.split(" ")[0]))
summary = data[["algo", "env_name", "imit_return"]].groupby(["algo", "env_name"]).describe()
summary.columns = summary.columns.get_level_values(1)
summary = summary.reset_index()

# Table 2 (https://arxiv.org/pdf/2211.11972.pdf)
paper = pd.DataFrame.from_records([
    {"algo": "BC", "env_name": "seals/Ant-v0", "mean": 1953, "margin": 123},
    {"algo": "BC", "env_name": "seals/HalfCheetah-v0", "mean": 3446, "margin": 130},
])
paper["count"] = 5
paper["confidence_level"] = 0.95
# Back out the standard deviation from the margin of error.
paper["std"] = (paper["margin"] * paper["count"]) / scipy.stats.t.ppf(1-((1-paper["confidence_level"])/2), paper["count"] -1)

comparison = pd.merge(summary, paper, on=["algo", "env_name"])

comparison["pvalue"] = scipy.stats.ttest_ind_from_stats(
    comparison["mean_x"],
    comparison["std_x"],
    comparison["count_x"],
    comparison["mean_y"],
    comparison["std_y"],
    comparison["count_y"]).pvalue
```
