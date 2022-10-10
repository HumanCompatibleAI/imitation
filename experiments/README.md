Experiment scripts are compatible with Linux and macOS.

## (macOS only) macOS compatibility setup

macOS to install some GNU-compatible binaries before all experiments scripts will work.

```
brew install coreutils gnu-getopt parallel
```

## Scripts

### Phase 1: Download RL (PPO2) expert policies.

Expert policies have been saved in HuggingFace, and to work with these scripts should be downloaded to `data/expert_models/`. Environments with pre-trained models:
- [CartPole](https://huggingface.co/HumanCompatibleAI/ppo-seals-CartPole-v0)
- [MountainCar](https://huggingface.co/HumanCompatibleAI/ppo-seals-MountainCar-v0)
- [HalfCheetah](https://huggingface.co/HumanCompatibleAI/ppo-seals-HalfCheetah-v0)
- [Hopper](https://huggingface.co/HumanCompatibleAI/ppo-seals-Hopper-v0)
- [Walker](https://huggingface.co/HumanCompatibleAI/ppo-seals-Walker2d-v0)
- [Swimmer](https://huggingface.co/HumanCompatibleAI/ppo-seals-Swimmer-v0)
- [Ant](https://huggingface.co/HumanCompatibleAI/ppo-seals-Ant-v0)
- [Humanoid](https://huggingface.co/HumanCompatibleAI/ppo-seals-Humanoid-v0)

To download, clone the [rl-baselines3-zoo repository](https://github.com/DLR-RM/rl-baselines3-zoo), and run a command like `python rl_zoo3/load_from_hub.py --algo ppo --env seals/Ant-v0 -orga HumanCompatibleAI -f ../imitation/data/expert_models/` (but modifying the path if necessary to ensure the correct download location).

### Phase 2: Generate expert demonstrations from models.

Run `experiments/rollouts_from_policies.sh`. (Rollouts saved in `data/expert_models/`).
Demonstrations are used in Phase 3 for imitation learning.

### Phase 3: Train imitation learning.

Run `experiments/imit_benchmark.sh --run_name RUN_NAME`. To choose AIRL or GAIL, add the `--airl` and `--gail` flags (default is GAIL).

To analyze these results, run `python -m imitation.scripts.analyze with run_name=RUN_NAME`. Analysis can be run even while training is midway (will only show completed imitation learner's results). [Example output.](https://gist.github.com/shwang/4049cd4fb5cab72f2eeb7f3d15a7ab47)

## Hyperparameter tuning

Add a named config containing the hyperparameter search space and other settings to `src/imitation/scripts/config/parallel.py`. (`def example_cartpole_rl():` is an example).

Run your hyperparameter tuning experiment using `python -m imitation.scripts.parallel with YOUR_NAMED_CONFIG inner_run_name=RUN_NAME`.

Analyze imitation learning experiments using `python -m imitation.scripts.analyze with run_name=RUN_NAME source_dir=~/ray_results`.

View Stable Baselines training stats on TensorBoard (available for regular RL, imitation learning, and transfer learning) using `tensorboard --log_dir ~/ray_results`. To view only a subset of TensorBoard training progress use `imitation.scripts.analyze gather_tb_directories with source_dir=~/ray_results run_name=RUN_NAME`.
