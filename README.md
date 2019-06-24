[![Build Status](https://travis-ci.com/HumanCompatibleAI/imitation.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/imitation)
[![codecov](https://codecov.io/gh/HumanCompatibleAI/imitation/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/imitation)

# Imitation Learning Baseline Implementations

This project aims to provide clean implementations of imitation learning algorithms.
Currently we have implementations of [AIRL](https://arxiv.org/abs/1710.11248) and 
[GAIL](https://arxiv.org/abs/1606.03476), and intend to add more in the future.

To install:
```
sudo apt install libopenmpi-dev
conda create -n imitation python=3.7  # py3.6 is also okay.
conda activate imitation
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .  # install `imitation` in developer mode
```

To run:
```
# train demos with normal AIRL
python -m imitation.scripts.data_collect --gin_config configs/cartpole_data_collect.gin
# do AIRL magic to get back reward from demos
python -m imitation.scripts.train --gin_config configs/cartpole_orig_airl_repro.gin
```

# Contributing
  * Follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html). Examples of Google-style
docstrings can be found [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
  * PEP8 guidelines and line-width=80 are enforced by `ci/code_checks.sh`, which is automatically run by Travis CI.
  * Add units tests covering any new features, or bugs that are being fixed.
  * Code coverage is automatically enforced by CodeCov.
    The exact coverage required by CodeCov depends on the previous
    code coverage %. Files in `imitation/{examples,scripts}/` have no
    coverage requirements.
