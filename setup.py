"""Setup for imitation: a reward and imitation learning library."""

from setuptools import find_packages, setup

import src.imitation  # pytype: disable=import-error

TESTS_REQUIRE = [
    "seals",
    "black",
    "coverage",
    "codecov",
    "codespell",
    "darglint",
    "filelock",
    "flake8",
    "flake8-blind-except",
    "flake8-builtins",
    "flake8-commas",
    "flake8-debugger",
    "flake8-docstrings",
    "flake8-isort",
    "ipykernel",
    "jupyter",
    # remove pin once https://github.com/jupyter/jupyter_client/issues/637 fixed
    "jupyter-client<7.0",
    "pandas",
    "pytest",
    "pytest-cov",
    "pytest-notebook",
    "pytest-xdist",
    "pytype",
    "ray[debug,tune]~=0.8.5",
    "scipy>=1.8.0",
    "wandb",
    "huggingface_sb3",
]
DOCS_REQUIRE = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme",
    "sphinxcontrib-napoleon",
]
PARALLEL_REQUIRE = ["ray[debug,tune]~=0.8.5"]


def get_readme() -> str:
    """Retrieve content from README."""
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="imitation",
    version=src.imitation.__version__,
    description="Implementation of modern reward and imitation learning algorithms.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="Center for Human-Compatible AI and Google",
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"imitation": ["py.typed", "envs/examples/airl_envs/assets/*.xml"]},
    install_requires=[
        # If you change gym version here, change it in "mujoco" below too.
        # pinned to 0.21 until https://github.com/DLR-RM/stable-baselines3/pull/780
        # goes upstream.
        "gym[classic_control]==0.21.0",
        "matplotlib",
        "numpy>=1.15",
        "torch>=1.4.0",
        "tqdm",
        "scikit-learn>=0.21.2",
        # TODO(adam): switch back to PyPi once following makes it to release:
        # https://github.com/DLR-RM/stable-baselines3/pull/734 is released
        (
            "stable-baselines3@git+https://github.com/carlosluis/stable-baselines3.git"
            "@gym_fixes#egg=stable-baselines3"
        ),
        "stable-baselines3>=1.4.0",
        "sacred~=0.8.1",
        "tensorboard>=1.14",
    ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        # recommended packages for development
        "dev": [
            "autopep8",
            "awscli",
            "ntfy[slack]",
            "ipdb",
            "isort~=5.0",
            "pytype",
            "codespell",
            # for convenience
            *TESTS_REQUIRE,
            *DOCS_REQUIRE,
        ],
        "test": TESTS_REQUIRE,
        "docs": DOCS_REQUIRE,
        "parallel": PARALLEL_REQUIRE,
        "mujoco": [
            "gym[classic_control,mujoco]==0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imitation-eval-policy=imitation.scripts.eval_policy:main_console",
            "imitation-parallel=imitation.scripts.parallel:main_console",
            (
                "imitation-train-adversarial="
                "imitation.scripts.train_adversarial:main_console"
            ),
            "imitation-train-imitation=imitation.scripts.train_imitation:main_console",
            (
                "imitation-train-preference-comparisons="
                "imitation.scripts.train_preference_comparisons:main_console"
            ),
            "imitation-train-rl=imitation.scripts.train_rl:main_console",
        ],
    },
    url="https://github.com/HumanCompatibleAI/imitation",
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
