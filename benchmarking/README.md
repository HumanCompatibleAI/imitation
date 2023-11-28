# Benchmarking imitation

The imitation library is benchmarked by running the algorithms BC, DAgger, AIRL and GAIL
on five different environments from the
[seals environment suite](https://github.com/HumanCompatibleAI/seals)
each with 10 different random seeds.
You will find the benchmark results in the release artifacts, e.g. for the v1.0 release
[here](https://github.com/HumanCompatibleAI/imitation/releases/download/v1.0.0/benchmark_runs.zip).


## Running a Single Benchmark

To run a single benchmark from the commandline, you may use:

```bash
python -m imitation.scripts.<train_script> <algo> with <algo>_<env>
```

There are two different `train_scripts`: `train_imitation` and `train_adversarial` each running different algorithms:

| train_script      | algo       |
|-------------------|------------|
| train_imitation   | bc, dagger |
| train_adversarial | gail, airl |

There are five environment configurations for which we have tuned hyperparameters:

| environment        |
|--------------------|
| seals_ant          |
| seals_half_cheetah |
| seals_hopper       |
| seals_swimmer      |
| seals_walker       |


If you want to run the same benchmark from a python script, you can use the following code:

```python
...
from imitation.scripts.<train_script> import <train_script>_ex
<train_script>_ex.run(command_name="<algo>", named_configs=["<algo>_<env>"])
```

### Inputs

The tuned hyperparameters can be found in `src/imitation/scripts/config/tuned_hps`.
For v0.4.0, they correspond to the hyperparameters used in the paper
[imitation: Clean Imitation Learning Implementations](https://arxiv.org/abs/2211.11972).
You may be able to get reasonable performance by using hyperparameters tuned for a similar environment.

The experts and expert demonstrations are loaded from the HuggingFace model hub and
are grouped under the [HumanCompatibleAI Organization](https://huggingface.co/HumanCompatibleAI).

### Outputs

The training scripts are [sacred experiments](https://sacred.readthedocs.io) which place
their output in an output folder structured like this:

```
output
├── airl
│ └── seals-Swimmer-v1
│     └── 20231012_121226_c5c0e4
│         └── sacred -> ../../../sacred/train_adversarial/2
├── dagger
│ └── seals-CartPole-v0
│     └── 20230927_095917_c29dc2
│         └──  sacred -> ../../../sacred/train_imitation/1
└── sacred
    ├── train_adversarial
    │ ├── 1
    │ ├── 2
    │ ├── 3
    │ ├── 4
    │ ├── ...
    │ └── _sources
    └── train_imitation
        ├── 1
        └── _sources
```

In the `sacred` folder all runs are grouped by the training script, and each gets a
folder with their run id.
That run folder contains
- a `config.json` file with the hyperparameters used for that run
- a `run.json` file with run information with the final score and expert score
- a `cout.txt` file with the stdout of the run

Additionally, there are run folders grouped by algorithm and environment.
They contain further log files and model checkpoints as well as a symlink to the
corresponding sacred run folder.

Important entries in the json files are:
- `run.json`
  - `command`: The name of the algorithm
  - `result.imit_stats.monitor_return_mean`: the score of a run
  - `result.expert_stats.monitor_return_mean`: the score of the expert policy that was used for a run
- `config.json`
  - `environment.gym_id` The environment name of the run

## Running the Complete Benchmark Suite

To execute the entire benchmarking suite with 10 seeds for each configuration,
you can utilize the `run_all_benchmarks.sh` script.
This script will consecutively run all configurations.
To optimize the process, consider parallelization options.
You can either send all commands to GNU Parallel,
use SLURM by invoking `run_all_benchmarks_on_slurm.sh` or
split up the lines in multiple scripts to run on multiple machines manually.

### Generating Benchmark Summaries

There are scripts to summarize all runs in a folder in a CSV file or in a markdown file.
For the CSV, run:

```shell
python sacred_output_to_csv.py output/sacred > summary.csv
```

This generates a csv file like this:

```
algo, env, score, expert_score
gail, seals/Walker2d-v1, 2298.883520464286, 2502.8930135576925
gail, seals/Swimmer-v1, 287.33667667857145, 295.40472964423077
airl, seals/Walker2d-v1, 310.4065185178571, 2502.8930135576925
...
```

For a more comprehensive summary that includes aggregate statistics such as
mean, standard deviation, IQM (Inter Quartile Mean) with confidence intervals,
as recommended by the [rliable library](https://github.com/google-research/rliable),
use the following command:

```shell
python sacred_output_to_markdown_summary output/sacred --output summary.md
```

This will produce a markdown summary file named `summary.md`.



**Hint:**
If you have multiple output folders, because you ran different parts of the
benchmark on different machines, you can copy the output folders into a common root
folder.
The above scripts will search all nested directories for folders with
a `run.json` and a `config.json` file.
For example, calling `python sacred_output_to_csv.py benchmark_runs/ > summary.csv`
on an output folder structured like this:
```
benchmark_runs
├── first_batch
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── ...
└── second_batch
    ├── 1
    ├── 2
    ├── 3
    ├── ...
```
will aggregate all runs from both `first_batch` and `second_batch` into a single
csv file.

## Comparing an Algorithm against the Benchmark Runs

If you modified one of the existing algorithms or implemented a new one, you might want
to compare it to the benchmark runs to see if there is a significant improvement or not.

If your algorithm has the same file output format as described above, you can use the
`compute_probability_of_improvement.py` script to do the comparison.
It uses the "Probability of Improvement" metric as recommended by the
[rliable library](https://github.com/google-research/rliable).

```shell
python compute_probability_of_improvement.py <your_runs_dir> <baseline_runs_dir> --baseline-algo <algo>
```

where:
- `your_runs_dir` is the directory containing the runs for your algorithm
- `baseline_runs_dir` is the directory containing runs for a known algorithm. Hint: you do not need to re-run our benchmarks. We provide our run folders as release artifacts.
- `algo` is the algorithm you want to compare against

If `your_runs_dir` contains runs for more than one algorithm, you will have to
disambiguate using the `--algo` option.

## Tuning Hyperparameters

The hyperparameters of any algorithm in imitation can be tuned using `src/imitation/scripts/tuning.py`.
The benchmarking hyperparameter configs were generated by tuning the hyperparameters using
the search space defined in the `scripts/config/tuning.py`.

The tuning script proceeds in two phases:
1. Tune the hyperparameters using the search space provided.
2. Re-evaluate the best hyperparameter config found in the first phase based on the maximum mean return on a separate set of seeds. Report the mean and standard deviation of these trials.

To use it with the default search space:
```bash
python -m imitation.scripts.tuning with <algo> 'parallel_run_config.base_named_configs=["<env>"]'
```

In this command:
- `<algo>` provides the default search space and settings for the specific algorithm, which is defined in the `scripts/config/tuning.py`
- `<env>` sets the environment to tune the algorithm in. They are defined in the algo-specifc `scripts/config/train_[adversarial|imitation|preference_comparisons|rl].py` files. For the already tuned environments, use the `<algo>_<env>` named configs here.

See the documentation of `scripts/tuning.py` and `scripts/parallel.py` for many other arguments that can be
provided through the command line to change the tuning behavior.
