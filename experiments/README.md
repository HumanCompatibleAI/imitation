Experiment scripts are only compatible with Linux.

### Setup

Phase 1 requires [AWS CLI](https://aws.amazon.com/cli/) because it downloads data from AWS S3.

### Phase 1: Download RL (PPO2) expert policies and AIRL/GAIL reward models.

Use `experiments/download_models.sh`. (Downloads to `data/{expert,reward}_models/`).
Expert policies are used in Phase 2 to generate demonstrations.
Reward models are used in Phase 4 for transfer learning.

Want to use other policies or reward models for Phase 2 or 4? 
  * New policies can be trained using `experiments/train_experts.sh`. 
  * New reward models are generated in Phase 3 `experiments/imit_benchmark.sh`.
  * For both scripts, you can enter the optional commands suggested at the end of the script to upload new files to S3 (will need write access to our S3 bucket). Or, you can manually patch `data/{expert,reward}_models` using the script's output files.


### Phase 2: Generate expert demonstrations from models.

Run `experiments/rollouts_from_policies.sh`. (Rollouts saved in `data/expert_models/`).
Demonstrations are used in Phase 3 for imitation learning.

### Phase 3: Train imitation learning.

Run `experiments/imit_benchmark.sh --run_name RUN_NAME`. To choose AIRL or GAIL, add the `--airl` and `--gail` flags (default is GAIL).

To analyze these results, run `python -m imitation.scripts.analyze with run_name=RUN_NAME`. Analysis can be run even while training is midway (will only show completed imitation learner's results). [Example output.](https://gist.github.com/shwang/4049cd4fb5cab72f2eeb7f3d15a7ab47)

### Phase 4: Transfer learning.

Run `experiments/transfer_benchmark.sh`. To choose AIRL or GAIL, add the `--airl` and `--gail` flags (default is GAIL). Transfer rewards are loaded from `data/reward_models`.

## Hyperparameter tuning

Add a named config containing the hyperparameter search space and other settings to `src/imitation/scripts/config/parallel.py`. (`def example_cartpole_rl():` is an example).

Run your hyperparameter tuning experiment using `python -m imitation.scripts.parallel with YOUR_NAMED_CONFIG inner_run_name=RUN_NAME`.

Analyze imitation learning experiments using `python -m imitation.scripts.analyze with run_name=RUN_NAME source_dir=~/ray_results`.

View Stable Baselines training stats on TensorBoard (available for regular RL, imitation learning, and transfer learning) using `tensorboard --log_dir ~/ray_results`. To view only a subset of TensorBoard training progress use `imitation.scripts.analyze gather_tb_directories with source_dir=~/ray_results run_name=RUN_NAME`.
