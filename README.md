[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/imitation.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/imitation)
[![codecov](https://codecov.io/gh/HumanCompatibleAI/imitation/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/imitation)

# Imitation Learning Baseline Implementations

This project aims to provide clean implementations of imitation learning algorithms.
Currently we have implementations of [AIRL](https://arxiv.org/abs/1710.11248) and 
[GAIL](https://arxiv.org/abs/1606.03476), and intend to add more in the future.

### To install:
```
conda create -n imitation python=3.8  # python 3.7 and virtualenv are also okay.
conda activate imitation
pip install -e '.[dev]'  # install `imitation` in developer mode
```

### Optional Mujoco Dependency:

Follow instructions to install [mujoco_py v1.5 here](https://github.com/openai/mujoco-py/tree/498b451a03fb61e5bdfcb6956d8d7c881b1098b5#install-mujoco).

### To run:
```
# Train PPO2 agent on cartpole and collect expert demonstrations
python -m imitation.scripts.expert_demos with cartpole
# Train AIRL on from demonstrations
python -m imitation.scripts.train_adversarial with cartpole airl
```
View Tensorboard with `tensorboard --logdir output/`.


# Contributing

All PRs should include unit tests for any new features, using `pytest`.
We use `pytype` for static type checking. Please add type annotations where possible.
It is OK to omit annotations when it would make the code significantly more complex.

Code coverage is automatically enforced by CodeCov.
The exact coverage required by CodeCov depends on the previous code coverage %.
Files in `imitation/{examples,scripts}/` have no coverage requirements.

Please follow a coding style of:
  * PEP8, with line width 88.
  * Use the `black` autoformatter.
  * Follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) unless
    it conflicts with the above. Examples of Google-style docstrings can be found
    [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Run `ci/code_checks.sh` to run linting and static type checks, and `pytest tests/` for unit tests.
These also run on CircleCI and are required to pass before merging. You may wish to configure
`ci/code_checks.sh` as a Git commit hook.
