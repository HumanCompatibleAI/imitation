[![Build Status](https://travis-ci.com/HumanCompatibleAI/airl.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/airl)

# Adversarial Inverse Reinforcment Learning

This repository reimplements AIRL with modern dependencies.
Paper: https://arxiv.org/abs/1710.11248
Original Implementation: https://github.com/justinjfu/inverse_rl

To install:
```
sudo apt install libopenmpi-dev
conda create -n yairl python
conda activate yairl
pip install -r requirements.txt -r jupt_requirements.txt
pip install -e .  # install yairl in developer mode
```

To run:
```
# train demos with normal AIRL
python scripts/data_collect.py --gin_config configs/cartpole_data_collect.gin
# do AIRL magic to get back reward from demos
python scripts/run_training.py --gin_config configs/cartpole_orig_airl_repro.gin
```
