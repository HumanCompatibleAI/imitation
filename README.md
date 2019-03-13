# Yet Another AIRL (Implementation?)

To install:

```
conda create -n yairl python=3.6.8  # Tensorflow doesn't work with python 3.7 at the time of writing.
conda activate yairl

pip install -r requirements.txt -r jupt_requirements.txt
```

Now install experimental stable-baselines. (Contains a PPO2 logging
improvement and merges an experimental action_probabilities
function needed to evaluate continuous action space probabilities)

```
sudo apt install libopenmpi-dev
pip install git+git://github.com/hill-a/stable-baselines.git
pip install -e .
```

To run:

```
# train demos with normal AIRL
python scripts/cartpole_data_collect.py
# do AIRL magic to get back reward from demos
python scripts/cartpole_irl.py
```
