[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/imitation.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/imitation)
[![Documentation Status](https://readthedocs.org/projects/imitation/badge/?version=latest)](https://imitation.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/HumanCompatibleAI/imitation/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/imitation)
[![PyPI version](https://badge.fury.io/py/imitation.svg)](https://badge.fury.io/py/imitation)


# Imitation Learning Baseline Implementations

This project aims to provide clean implementations of imitation learning algorithms.
Currently we have implementations of Behavioral Cloning, [DAgger](https://arxiv.org/pdf/1011.0686.pdf) (with synthetic examples), [Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248), and [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476).

## Installation:

### Installing PyPI release

```
pip install imitation
```

### Install latest commit

```
git clone http://github.com/HumanCompatibleAI/imitation
cd imitation
pip install -e .
```

### Optional Mujoco Dependency:

Follow instructions to install [mujoco\_py v1.5 here](https://github.com/openai/mujoco-py/tree/498b451a03fb61e5bdfcb6956d8d7c881b1098b5#install-mujoco).


## CLI Quickstart:

We provide several CLI scripts as a front-end to the algorithms implemented in `imitation`. These use [Sacred](https://github.com/idsia/sacred) for configuration and replicability.

From [examples/quickstart.sh:](examples/quickstart.sh)

```bash
# Train PPO agent on cartpole and collect expert demonstrations. Tensorboard logs saved in `quickstart/rl/`
python -m imitation.scripts.expert_demos with fast cartpole log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast gail cartpole rollout_path=quickstart/rl/rollouts/final.pkl

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast airl cartpole rollout_path=quickstart/rl/rollouts/final.pkl
```
Tips:
  * Remove the "fast" option from the commands above to allow training run to completion.
  * `python -m imitation.scripts.expert_demos print_config` will list Sacred script options. These configuration options are documented in each script's docstrings.

For more information on how to configure Sacred CLI options, see the [Sacred docs](https://sacred.readthedocs.io/en/stable/).


## Python Interface Quickstart:

See [examples/quickstart.py](examples/quickstart.py) for an example script that loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.


### Density reward baseline

We also implement a density-based reward baseline. You can find an [example notebook here](examples/density_baseline_demo.ipynb).

# Citations (BibTeX)
```
@misc{wang2020imitation,
  author = {Wang, Steven and Toyer, Sam and Gleave, Adam and Emmons, Scott},
  title = {The {\tt imitation} Library for Imitation Learning and Inverse Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HumanCompatibleAI/imitation}},
}
```

# Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).
