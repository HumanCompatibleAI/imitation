# Adversarial Inverse Reinforcment Learning

This repository reimplements AIRL with modern dependencies.
Paper: https://arxiv.org/abs/1710.11248
Original Implementation: https://github.com/justinjfu/inverse_rl

To install:

```
conda create -n yairl python=3.6.8  # Tensorflow doesn't work with python 3.7 at the time of writing.
conda activate yairl

pip install -r requirements.txt -r jupt_requirements.txt
```

Now install stable-baselines.
```
sudo apt install libopenmpi-dev
pip install git+git://github.com/hill-a/stable-baselines.git
pip install -e .
```

To run:

```
# train demos with normal AIRL
python scripts/data_collect.py --gin_config configs/cartpole_data_collect.gin
# do AIRL magic to get back reward from demos
python scripts/run_training.py --gin_config configs/cartpole_orig_airl_repro.gin
```
