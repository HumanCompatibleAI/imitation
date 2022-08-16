"""Setup for imitation: a reward and imitation learning library."""

import os
import warnings
from sys import platform

from setuptools import find_packages, setup
from setuptools.command.install import install

IS_NOT_WINDOWS = os.name != "nt"

PARALLEL_REQUIRE = ["ray[debug,tune]>=1.13.0"]
PYTYPE = ["pytype==2022.7.26"] if IS_NOT_WINDOWS else []
if IS_NOT_WINDOWS:
    # TODO(adam): use this for Windows as well once PyPI is at >=1.6.1
    STABLE_BASELINES3 = "stable-baselines3>=1.6.0"
else:
    STABLE_BASELINES3 = (
        "stable-baselines3@git+"
        "https://github.com/DLR-RM/stable-baselines3.git@master"
    )

# pinned to 0.21 until https://github.com/DLR-RM/stable-baselines3/pull/780 goes
# upstream.
GYM_VERSION_SPECIFIER = "==0.21.0"

# Note: the versions of the test and doc requirements should be tightly pinned to known
#   working versions to make our CI/CD pipeline as stable as possible.
TESTS_REQUIRE = (
    [
        "seals==0.1.2",
        "black[jupyter]~=22.6.0",
        "coverage~=6.4.2",
        "codecov~=2.1.12",
        "codespell~=2.1.0",
        "darglint~=1.8.1",
        "filelock~=3.7.1",
        "flake8~=4.0.1",
        "flake8-blind-except==0.2.1",
        "flake8-builtins~=1.5.3",
        "flake8-commas~=2.1.0",
        "flake8-debugger~=4.1.2",
        "flake8-docstrings~=1.6.0",
        "flake8-isort~=4.1.2",
        "hypothesis~=6.54.1",
        "ipykernel~=6.15.1",
        "jupyter~=1.0.0",
        # TODO: upgrade jupyter-client once
        #  https://github.com/jupyter/jupyter_client/issues/637 is fixed
        "jupyter-client~=6.1.12",
        "pandas~=1.4.3",
        "pytest~=7.1.2",
        "pytest-cov~=3.0.0",
        "pytest-notebook==0.8.0",
        "pytest-xdist~=2.5.0",
        "scipy~=1.9.0",
        "wandb==0.12.21",
    ]
    + PARALLEL_REQUIRE
    + PYTYPE
)
DOCS_REQUIRE = [
    "sphinx~=5.1.1",
    "sphinx-autodoc-typehints~=1.19.1",
    "sphinx-rtd-theme~=1.0.0",
    "sphinxcontrib-napoleon==0.7",
    "furo==2022.6.21",
    "sphinx-copybutton==0.5.0",
    "sphinx-github-changelog~=1.2.0",
]


def get_readme() -> str:
    """Retrieve content from README."""
    with open("README.md", "r", encoding="utf-8") as f:
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
    # Note: while we are strict with our test and doc requirement versions, we try to
    #   impose as little restrictions on the install requirements as possible. Try to
    #   encode only known incompatibilities here. This prevents nasty dependency issues
    #   for our users.
    install_requires=[
        "gym[classic_control]" + GYM_VERSION_SPECIFIER,
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
            "sphinx-autobuild",
            # for convenience
            *TESTS_REQUIRE,
            *DOCS_REQUIRE,
        ]
        + PYTYPE,
        "test": TESTS_REQUIRE,
        "docs": DOCS_REQUIRE,
        "parallel": PARALLEL_REQUIRE,
        "mujoco": [
            "gym[classic_control,mujoco]" + GYM_VERSION_SPECIFIER,
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
