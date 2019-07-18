from setuptools import find_packages, setup

import imitation

# TF 1.14.0 is not compatible with sacred because of a TF bug.
TF_VERSION = '>=1.13.1,<2.0,!=1.14.0'
TESTS_REQUIRE = [
    'codecov',
    'pytest',
    'pytest-cov',
]

setup(
    name='imitation',
    version=imitation.__version__,
    description=(
        'Implementation of modern IRL and imitation learning algorithms.'),
    author='Center for Human-Compatible AI and Google',
    python_requires='>=3.6.0',
    packages=find_packages(exclude=['test*']),
    install_requires=[
        'gym',
        'numpy>=1.15',
        'tqdm',
        # FIXME: replace this with PyPI link once commit 2bc3c8722b9eb5 appears
        # in a release (will probably happen in version 2.6.1 or 2.7, when they
        # come out)
        'stable-baselines @ git+https://github.com/hill-a/stable-baselines.git',
        'jax!=0.1.37',
        'jaxlib~=0.1.20',
        # sacred==0.7.5 build is broken without pymongo
        'sacred>=0.7.4,!=0.7.5',
    ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        'gpu': [f'tensorflow-gpu{TF_VERSION}'],
        'cpu': [f'tensorflow{TF_VERSION}'],
        # recommended packages for development
        'dev': [
            'autopep8',
            'flake8',
            'flake8-blind-except',
            'flake8-builtins',
            'flake8-commas',
            'flake8-debugger',
            'flake8-isort',
            'sphinx',
            'sphinxcontrib-napoleon',
            'ipdb',
            'isort',
            'jupyter',
            'pytype',
            # for convenience
            *TESTS_REQUIRE,
        ],
        'test':
        TESTS_REQUIRE,
    },
    url='https://github.com/HumanCompatibleAI/airl',
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
