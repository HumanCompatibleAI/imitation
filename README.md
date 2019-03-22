[![Build Status](https://travis-ci.com/HumanCompatibleAI/airl.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/airl)

# Adversarial Inverse Reinforcment Learning

This repository reimplements AIRL with modern dependencies.
Paper: https://arxiv.org/abs/1710.11248
Original Implementation: https://github.com/justinjfu/inverse_rl

To install:
```
sudo apt install libopenmpi-dev
conda create -n yairl python=3.7.2
conda activate yairl
pip install -r requirements.txt -r jupt_requirements.txt
pip install -e .  # install yairl in developer mode
```

To run:
```
# train demos with normal AIRL
python scripts/cartpole_data_collect.py
# do AIRL magic to get back reward from demos
python scripts/cartpole_irl.py
```
