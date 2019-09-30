### Phase 1 Download PPO2 experts.

Use `experiments/download_experts.sh` . (Saves experts to `expert_models/` )

(Optional) Train your own experts. Run `experiments/train_experts.sh` .

### Phase 2 Generate expert demonstrations.

Run `experiments/rollouts_from_policies.sh` .

### Phase 3 Train imitation learning.

Run `experiments/imit_benchmark.sh --run_name RUN_NAME` . To choose AIRL or GAIL, add the `--airl` and `--gail` flags.

To analyze these results, run `python -m imitation.scripts.analyze with source_dir=output/sacred/train/ csv_output_path="analysis.csv" run_name=RUN_NAME` . Analysis can be run even while training is midway (will only show completed imitation learner's results). [Example output.](https://gist.github.com/shwang/4049cd4fb5cab72f2eeb7f3d15a7ab47)

### Phase 4 Transfer learning.

TBD.

### Hyperparameter tuning

Add a named config to `src/imitation/scripts/config/parallel.py` . (`def example_cartpole_rl():` is an example).

Run your hyperparameter tuning experiment using `python -m imitation.scripts.parallel with YOUR_NAMED_CONFIG inner_run_name=RUN_NAME`.

Analyze imitation learning experiments using `python -m imitation.scripts.analyze with run_name=RUN_NAME source_dir=~/ray_results`.

View Stable Baselines training stats on Tensorboard (available for regular RL, imitation learning, and transfer learning) using `tensorboard --log_dir ~/ray_results` . To view only a subset of Tensorboard training progress use `imitation.scripts.analyze tensorboard_filtered_launch with source_dir=~/ray_results run_name=RUN_NAME` .
