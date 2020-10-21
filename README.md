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

Follow instructions to install [mujoco_py v1.5 here](https://github.com/openai/mujoco-py/tree/498b451a03fb61e5bdfcb6956d8d7c881b1098b5#install-mujoco).


## Sacred CLI Quickstart:

```bash
# Train PPO agent on cartpole and collect expert demonstrations
python -m imitation.scripts.expert_demos with cartpole log_dir=quickstart

# Train GAIL from demonstrations
python -m imitation.scripts.train_adversarial with gail cartpole rollout_path=quickstart/rollouts/final.pkl

# Train AIRL from demonstrations
python -m imitation.scripts.train_adversarial with airl cartpole rollout_path=quickstart/rollouts/final.pkl

# Tip: `python -m imitation.scripts.* print_config` will list Sacred script options, which are documented
# in `src/imitation/scripts/`.
# For more information configuring Sacred options, see docs at https://sacred.readthedocs.io/en/stable/.
```


## Functional Interface Quickstart:

Here's an example script that loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.

```python
import gym
import pickle

import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.data import types
from imitation.util import logger, util


# Load pickled test demonstrations.
with open("tests/data/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
    # This is a list of `types.Trajectory`, where
    # every instance contains observations and actions for a single expert demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `types.Transitions`.
# This is a more general dataclass containing unordered (observation, actions, next_observation)
# transitions.
transitions = types.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1")

# Train BC on expert data. 
logger.configure("quickstart/tensorboard_dir_bc/")
bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
bc_trainer.train(n_epochs=2)

# Train GAIL on expert data.
logger.configure("quickstart/tensorboard_dir_gail/")
gail_trainer = GAIL(venv, expert_data=transitions, expert_batch_size=32, gen_algo=sb3.PPO(venv))
gail_trainer.train(total_timesteps=2000)

# Train AIRL on expert data.
logger.configure("quickstart/tensorboard_dir_airl/")
airl_trainer = AIRL(venv, expert_data=transitions, expert_batch_size=32, gen_algo=sb3.PPO(venv))
airl_trainer.train(total_timesteps=2000)
```

BC, GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that iterates over dictionaries containing observations, actions, and next_observations.


# Contributing

Please follow a coding style of:
  * PEP8, with line width 88.
  * Use the `black` autoformatter.
  * Follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) unless
    it conflicts with the above. Examples of Google-style docstrings can be found
    [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

PRs should include unit tests for any new features, and add type annotations where possible. 
It is OK to omit annotations when it would make the code significantly more complex.

We use `pytest` for unit testing: run `pytest tests/` to run the test suite.
We use `pytype` for static type analysis.
You should run `ci/code_checks.sh` to run linting and static type checks, and may wish
to configure this as a Git pre-commit hook.

These checks are run on CircleCI and are required to pass before merging.
Additionally, we track test coverage by CodeCov, and mandate that code coverage
should not decrease. This can be overridden by maintainers in exceptional cases.
Files in `imitation/{examples,scripts}/` have no coverage requirements.
