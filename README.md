[![Build Status](https://travis-ci.com/HumanCompatibleAI/imitation.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/imitation)

# Imitation Learning Baseline Implementations

This projects aim to provide clean implementations of imitation learning algorithms. 
Currently we have implementations of [AIRL](https://arxiv.org/abs/1710.11248) and 
[GAIL](https://arxiv.org/abs/1606.03476), and intend to add more in the future.

To install:
```
sudo apt install libopenmpi-dev
conda create -n imitation python
conda activate imitation
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .  # install yairl in developer mode
```

To run:
```
# train demos with normal AIRL
python scripts/data_collect.py --gin_config configs/cartpole_data_collect.gin
# do AIRL magic to get back reward from demos
python scripts/run_training.py --gin_config configs/cartpole_orig_airl_repro.gin
```
