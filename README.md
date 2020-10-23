[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/imitation.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/imitation)
[![Documentation Status](https://readthedocs.org/projects/imitation/badge/?version=latest)](https://imitation.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/HumanCompatibleAI/imitation/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/imitation)
[![PyPI version](https://badge.fury.io/py/imitation.svg)](https://badge.fury.io/py/imitation)


# Imitation Learning Baseline Implementations

This project aims to provide clean implementations of imitation learning algorithms.
Currently we have implementations of [AIRL](https://arxiv.org/abs/1710.11248) and 
[GAIL](https://arxiv.org/abs/1606.03476), and intend to add more in the future.

### To install:
```
conda create -n imitation python=3.7
conda activate imitation
pip install -e '.[dev]'  # install `imitation` in developer mode
```

### Optional Mujoco Dependency:

Follow instructions to install [mujoco\_py v1.5 here](https://github.com/openai/mujoco-py/tree/498b451a03fb61e5bdfcb6956d8d7c881b1098b5#install-mujoco).


## Sacred CLI Quickstart:

```bash
# Train PPO agent on cartpole and collect expert demonstrations. Tensorboard logs saved in `quickstart/rl/`
python -m imitation.scripts.expert_demos with fast cartpole log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast gail cartpole rollout_path=quickstart/rl/rollouts/final.pkl

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast airl cartpole rollout_path=quickstart/rl/rollouts/final.pkl
```
Tip: Remove the "fast" option from the above runs to run training to completion.
Tip: `python -m imitation.scripts.expert_demos print_config` will list Sacred script options, which are documented
in each script's docstrings.

For more information configuring Sacred CLI options, see [Sacred docs](https://sacred.readthedocs.io/en/stable/).


## Python Interface Quickstart:

See [examples/quickstart.py](examples/quickstart.py) for an example script that loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.

BC, GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that iterates over dictionaries containing observations, actions, and next\_observations.

# Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).
