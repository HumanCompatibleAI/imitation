"""Setup for imitation: a reward and imitation learning library."""

import os
import warnings
from sys import platform
from typing import TYPE_CHECKING

from setuptools import find_packages, setup
from setuptools.command.install import install

if TYPE_CHECKING:
    from setuptools_scm.version import ScmVersion

IS_NOT_WINDOWS = os.name != "nt"

PARALLEL_REQUIRE = ["ray[debug,tune]~=2.0.0"]
ATARI_REQUIRE = [
    "opencv-python",
    "ale-py==0.7.4",
    "pillow",
    "autorom[accept-rom-license]~=0.6.0",
]
PYTYPE = ["pytype==2022.7.26"] if IS_NOT_WINDOWS else []
STABLE_BASELINES3 = "stable-baselines3>=1.7.0,<2.0.0"
# pinned to 0.21 until https://github.com/DLR-RM/stable-baselines3/pull/780 goes
# upstream.
GYM_VERSION_SPECIFIER = "==0.21.0"

# Note: the versions of the test and doc requirements should be tightly pinned to known
#   working versions to make our CI/CD pipeline as stable as possible.
TESTS_REQUIRE = (
    [
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
        "mypy~=0.990",
        "pandas~=1.4.3",
        "pytest~=7.1.2",
        "pytest-cov~=3.0.0",
        "pytest-notebook==0.8.0",
        "pytest-xdist~=2.5.0",
        "scipy~=1.9.0",
        "wandb==0.12.21",
        "setuptools_scm~=7.0.5",
        "pre-commit>=2.20.0",
    ]
    + PARALLEL_REQUIRE
    + ATARI_REQUIRE
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
    "myst-nb==0.16.0",
    "ipykernel~=6.15.2",
] + ATARI_REQUIRE


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


def get_version(version: "ScmVersion") -> str:
    """Generates the version string for the package.

    This function replaces the default version format used by setuptools_scm
    to allow development builds to be versioned using the git commit hash
    instead of the number of commits since the last release, which leads to
    duplicate version identifiers when using multiple branches
    (see https://github.com/HumanCompatibleAI/imitation/issues/500).

    The version has the following format:

    {version}[.dev{build}]
    where build is the shortened commit hash converted to base 10.

    Args:
        version: The version object given by setuptools_scm, calculated
            from the git repository.

    Returns:
        The formatted version string to use for the package.
    """
    # We import setuptools_scm here because it is only installed after the module
    # is loaded and the setup function is called.
    from setuptools_scm import version as scm_version

    if version.node:
        # By default node corresponds to the short commit hash when using git,
        # plus a "g" prefix. We remove the "g" prefix from the commit hash which
        # is added by setuptools_scm by default ("g" for git vs. mercurial etc.)
        # because letters are not valid for version identifiers in PEP 440.
        # We also convert from hexadecimal to base 10 for the same reason.
        version.node = str(int(version.node.lstrip("g"), 16))
    if version.exact:
        # an exact version is when the current commit is tagged with a version.
        return version.format_with("{tag}")
    else:
        # the current commit is not tagged with a version, so we guess
        # what the "next" version will be (this can be disabled but is the
        # default behavior of setuptools_scm so it has been left in).
        return version.format_next_version(
            scm_version.guess_next_version,
            fmt="{guessed}.dev{node}",
        )


def get_local_version(version: "ScmVersion", time_format="%Y%m%d") -> str:
    """Generates the local version string for the package.

    By default, when commits are made on top of a release version, setuptools_scm
    sets the version to be {version}.dev{distance}+{node} where {distance} is the number
    of commits since the last release and {node} is the short commit hash.
    This function replaces the default version format used by setuptools_scm
    so that committed changes away from a release version are not considered
    local versions but dev versions instead (by using the format
    {version}.dev{node} instead. This is so that we can push test releases
    to TestPyPI (it does not accept local versions).

    Local versions are still present if there are uncommitted changes (if the tree
    is dirty), in which case the current date is added to the version.

    Args:
        version: The version object given by setuptools_scm, calculated
            from the git repository.
        time_format: The format to use for the date.

    Returns:
        The formatted local version string to use for the package.
    """
    return version.format_choice(
        "",
        "+d{time:{time_format}}",
        time_format=time_format,
    )


setup(
    cmdclass={"install": InstallCommand},
    name="imitation",
    use_scm_version={"local_scheme": get_local_version, "version_scheme": get_version},
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
        # TODO(adam): remove pyglet dependency once Gym upgraded to >0.21
        # Workaround for https://github.com/openai/gym/issues/2986
        # Discussed in https://github.com/HumanCompatibleAI/imitation/pull/603
        "pyglet==1.5.27",
        "matplotlib",
        "numpy>=1.15",
        "torch>=1.4.0",
        "tqdm",
        "scikit-learn>=0.21.2",
        "seals>=0.1.5",
        STABLE_BASELINES3,
        "sacred>=0.8.4",
        "tensorboard>=1.14",
        "huggingface_sb3>=2.2.1,<=2.2.5",
        "datasets>=2.8.0",
    ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        # recommended packages for development
        "dev": [
            "autopep8",
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
        "atari": ATARI_REQUIRE,
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
