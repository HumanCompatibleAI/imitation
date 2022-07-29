"""Setup for imitation: a reward and imitation learning library."""

import os
import warnings
from sys import platform

from setuptools import find_packages, setup
from setuptools.command.install import install

IS_NOT_WINDOWS = os.name != "nt"

PARALLEL_REQUIRE = ["ray[debug,tune]>=1.13.0"]
PYTYPE = ["pytype"] if IS_NOT_WINDOWS else []
if IS_NOT_WINDOWS:
    # TODO(adam): use this for Windows as well once PyPI is at >=1.6.1
    STABLE_BASELINES3 = "stable-baselines3>=1.6.0"
else:
    STABLE_BASELINES3 = (
        "stable-baselines3@git+"
        "https://github.com/DLR-RM/stable-baselines3.git@master"
    )

TESTS_REQUIRE = (
    [
        "seals",
        "black[jupyter]",
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
        "hypothesis",
        "ipykernel",
        "jupyter",
        # remove pin once https://github.com/jupyter/jupyter_client/issues/637 fixed
        "jupyter-client<7.0",
        "pandas",
        "pytest",
        "pytest-cov",
        "pytest-notebook",
        "pytest-xdist",
        "scipy>=1.8.0",
        "wandb",
    ]
    + PARALLEL_REQUIRE
    + PYTYPE
)
DOCS_REQUIRE = [
    # TODO(adam): unpin once https://github.com/sphinx-doc/sphinx/issues/10705 fixed
    "sphinx~=5.0.2",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme",
    "sphinxcontrib-napoleon",
]


def get_readme() -> str:
    """Retrieve content from README."""
    with open("README.md", "r") as f:
        return f.read()


class InstallCommand(install):
    """Custom install command to throw warnings about external dependencies."""

    def run(self):
        """Run the install command."""
        install.run(self)

        if platform == "darwin":
            warnings.warn(
                "Installation of important packages for macOS is required. "
                "Scripts in the experiments folder will likely not run without these "
                "packages: gnu-getopt, parallel, coreutils. They can be installed with "
                "Homebrew by running `brew install gnu-getopt parallel coreutils`."
                "See https://brew.sh/ for installation instructions.",
            )


setup(
    cmdclass={"install": InstallCommand},
    name="imitation",
    # Disable local scheme to allow uploads to Test PyPI.
    # See https://github.com/pypa/setuptools_scm/issues/342
    use_scm_version={"local_scheme": "no-local-version"},
    setup_requires=["setuptools_scm"],
    description="Implementation of modern reward and imitation learning algorithms.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="Center for Human-Compatible AI and Google",
    python_requires=">=3.8.0",
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
        STABLE_BASELINES3,
        # TODO(adam) switch to upstream release if they make it
        #  See https://github.com/IDSIA/sacred/issues/879
        "chai-sacred>=0.8.3",
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
            "codespell",
            # for convenience
            *TESTS_REQUIRE,
            *DOCS_REQUIRE,
        ]
        + PYTYPE,
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
