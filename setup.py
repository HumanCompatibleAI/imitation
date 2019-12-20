from setuptools import find_packages, setup
import src.imitation  # pytype: disable=import-error

TF_VERSION = '>=1.15.0,<2.0'
TESTS_REQUIRE = [
    # remove pin once https://github.com/nedbat/coveragepy/issues/881 fixed
    'coverage==4.5.4',
    'codecov',
    'codespell',
    'flake8',
    'flake8-blind-except',
    'flake8-builtins',
    'flake8-commas',
    'flake8-debugger',
    'flake8-isort',
    'pytest',
    'pytest-cov',
    'pytest-shard',
    'pytest-xdist',
    'pytype',
    'sphinx',
    'sphinxcontrib-napoleon',
]

setup(
    name='imitation',
    version=src.imitation.__version__,
    description=(
        'Implementation of modern IRL and imitation learning algorithms.'),
    author='Center for Human-Compatible AI and Google',
    python_requires='>=3.7.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        'imitation': [
            'py.typed',
            'envs/examples/airl_envs/assets/*.xml',
        ],
    },
    install_requires=[
        'awscli',
        'cloudpickle>=0.5.5',
        'gym[classic_control]',
        'numpy>=1.15',
        'ray[debug]==0.7.4',
        'tqdm',
        'scikit-learn>=0.21.2',
        # TODO(adam): Change to >=2.10.0 once 2.10.0 released
        'stable-baselines @ git+https://github.com/hill-a/stable-baselines.git',
        'jax!=0.1.37',
        'jaxlib~=0.1.20',
        # sacred==0.7.5 build is broken without pymongo
        # sacred>0.7.4 have non-picklable config objects (see GH #109)
        'sacred==0.7.4',
    ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        'gpu': [f'tensorflow-gpu{TF_VERSION}'],
        'cpu': [f'tensorflow{TF_VERSION}'],
        # recommended packages for development
        'dev': [
            'autopep8',
            'ntfy[slack]',
            'ipdb',
            'isort',
            'jupyter',
            *TESTS_REQUIRE,
        ],
        'test': TESTS_REQUIRE,
    },
    entry_points={
        'console_scripts': [
            ('imitation-expert-demos=imitation.scripts.expert_demos'
             ':main_console'),
            'imitation-train=imitation.scripts.train_adversarial:main_console',
        ],
    },
    url='https://github.com/HumanCompatibleAI/imitation',
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
