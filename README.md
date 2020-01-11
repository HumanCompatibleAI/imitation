[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/imitation.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/imitation)
[![codecov](https://codecov.io/gh/HumanCompatibleAI/imitation/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/imitation)

# Imitation Learning Baseline Implementations

This project aims to provide clean implementations of imitation learning algorithms.
Currently we have implementations of [AIRL](https://arxiv.org/abs/1710.11248) and 
[GAIL](https://arxiv.org/abs/1606.03476), and intend to add more in the future.

To install:
```
conda create -n imitation python=3.8  # python 3.7 and virtualenv are also okay.
conda activate imitation
pip install -e '.[dev]'  # install `imitation` in developer mode
```

To run:
```
# Train PPO2 agent on cartpole and collect expert demonstrations
python -m imitation.scripts.expert_demos with cartpole
# Train AIRL on from demonstrations
python -m imitation.scripts.train_adversarial with cartpole airl
```

# Contributing
  * Follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html). Examples of Google-style
docstrings can be found [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
  * Add units tests covering any new features, or bugs that are being fixed.
  * PEP8 guidelines with line width 80 and 2-space indents are enforced by `ci/lint.sh`,
which is automatically run by Travis CI.
  * Static type checking via `pytype` is automatically run in `ci/type_check.sh`.
  * Code coverage is automatically enforced by CodeCov.
    The exact coverage required by CodeCov depends on the previous
    code coverage %. Files in `imitation/{examples,scripts}/` have no
    coverage requirements.
