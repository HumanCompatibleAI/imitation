# Yet Another AIRL (Implementation?)

To install:

```
conda env create -n yairl
source activate yairl
pip install -r requirements.txt -r jupt_requirements.txt
pip install -e .
```

To run:

```
# train demos with normal AIRL
python scripts/cartpole_data_collect.py
# do AIRL magic to get back reward from demos
python scripts/cartpole_irl.py
```
