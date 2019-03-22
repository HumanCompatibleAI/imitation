from distutils.core import setup

setup(
    name='yairl',
    version='0.1',
    packages=['yairl', ],
    install_requires=['gym', 'numpy', 'tqdm', ],
    extras_require = {
        'gpu':  ['tensorflow-gpu'],
        'cpu': ['tensorflow']
    },
    url='https://github.com/HumanCompatibleAI/airl',
    description='Implementation of modern IRL and imitation learning algorithms.',
    author='Center for Human-Compatible AI and Google',
)
