from setuptools import find_packages, setup

setup(
    name='imitation',
    version='0.1',
    description=(
        'Implementation of modern IRL and imitation learning algorithms.'),
    author='Center for Human-Compatible AI and Google',
    python_requires='>=3.6.0',
    packages=find_packages(exclude=['test*']),
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
    url='https://github.com/HumanCompatibleAI/airl',
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],

)
