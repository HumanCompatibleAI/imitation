from setuptools import find_packages, setup

import src.imitation

setup(
    name='imitation',
    version=src.imitation.__version__,
    description=(
        'Implementation of modern IRL and imitation learning algorithms.'),
    author='Center for Human-Compatible AI and Google',
    python_requires='>=3.6.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'gin-config',
        'gym',
        'numpy',
        'tensorflow',
        'tqdm',
        'stable-baselines',
        'jax',
        'jaxlib',
    ],
    tests_require=[
        'codecov',
        'pytest',
        'pytest-cov',
    ],
    extras_require={
        'gpu':  ['tensorflow-gpu'],
        'cpu': ['tensorflow']
    },
    url='https://github.com/HumanCompatibleAI/imitation',
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
        'Programming Language :: Python :: Implementation :: PyPy'
    ],

)
