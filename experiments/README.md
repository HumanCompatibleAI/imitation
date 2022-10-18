Experiment scripts are compatible with Linux and macOS.

## (macOS only) macOS compatibility setup

macOS to install some GNU-compatible binaries before all experiments scripts will work.

```
brew install coreutils gnu-getopt parallel
```

## Scripts

### Phase 1: Generate expert demonstrations from models.

Run `experiments/rollouts_from_policies.sh`. (Rollouts saved in `output/train_experts/`).
Demonstrations are used in Phase 2 for imitation learning.

### Phase 2: Train imitation learning.

Run `experiments/imit_benchmark.sh --run_name RUN_NAME`. To choose AIRL or GAIL, add the `--airl` and `--gail` flags (default is GAIL).

To analyze these results, run `python -m imitation.scripts.analyze with run_name=RUN_NAME`. Analysis can be run even while training is midway (will only show completed imitation learner's results). [Example output.](https://gist.github.com/shwang/4049cd4fb5cab72f2eeb7f3d15a7ab47)

### Phase 3: Transfer learning.

Run `experiments/transfer_learn_benchmark.sh`. To choose AIRL or GAIL, add the `--airl` and `--gail` flags (default is GAIL). Transfer rewards are loaded from `data/reward_models`.

## Hyperparameter tuning

Add a named config containing the hyperparameter search space and other settings to `src/imitation/scripts/config/parallel.py`. (`def example_cartpole_rl():` is an example).

Run your hyperparameter tuning experiment using `python -m imitation.scripts.parallel with YOUR_NAMED_CONFIG inner_run_name=RUN_NAME`.

Analyze imitation learning experiments using `python -m imitation.scripts.analyze with run_name=RUN_NAME source_dir=~/ray_results`.

View Stable Baselines training stats on TensorBoard (available for regular RL, imitation learning, and transfer learning) using `tensorboard --log_dir ~/ray_results`. To view only a subset of TensorBoard training progress use `imitation.scripts.analyze gather_tb_directories with source_dir=~/ray_results run_name=RUN_NAME`.
