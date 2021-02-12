===============
Getting Started
===============


CLI Quickstart
==============

We provide several CLI scripts as front-ends to the algorithms implemented in ``imitation``.
These use `Sacred <https://github.com/idsia/sacred>`_ for configuration and replicability.

For information on how to configure Sacred CLI options, see the `Sacred docs <https://sacred.readthedocs.io/en/stable/>`_.

.. code-block:: bash

    # Train PPO agent on cartpole and collect expert demonstrations. Tensorboard logs saved
    # in `quickstart/rl/`
    python -m imitation.scripts.expert_demos with fast cartpole log_dir=quickstart/rl/

    # Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
    python -m imitation.scripts.train_adversarial with fast gail cartpole \
        rollout_path=quickstart/rl/rollouts/final.pkl

    # Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
    python -m imitation.scripts.train_adversarial with fast airl cartpole \
        rollout_path=quickstart/rl/rollouts/final.pkl


.. note::
  Remove the ``fast`` option from the commands above to allow training run to completion.

.. tip::
  ``python -m imitation.scripts.expert_demos print_config`` will list Sacred script options.
  These configuration options are also documented in each script's docstrings.


Python Interface Quickstart
===========================

Here's an `example script`_ that loads CartPole-v1 demonstrations and trains BC, GAIL, and
AIRL models on that data.

.. _example script: https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py

.. code-block:: python

    """Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
    """
    import pathlib
    import pickle
    import tempfile

    import stable_baselines3 as sb3

    from imitation.algorithms import adversarial, bc
    from imitation.data import rollout
    from imitation.util import logger, util

    # Load pickled test demonstrations.
    with open("tests/data/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
        # This is a list of `imitation.data.types.Trajectory`, where
        # every instance contains observations and actions for a single expert
        # demonstration.
        trajectories = pickle.load(f)

    # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # This is a more general dataclass containing unordered
    # (observation, actions, next_observation) transitions.
    transitions = rollout.flatten_trajectories(trajectories)

    venv = util.make_vec_env("CartPole-v1", n_envs=2)

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    # Train BC on expert data.
    # BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
    # dictionaries containing observations and actions.
    logger.configure(tempdir_path / "BC/")
    bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
    bc_trainer.train(n_epochs=1)

    # Train GAIL on expert data.
    # GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
    # iterates over dictionaries containing observations, actions, and next_observations.
    logger.configure(tempdir_path / "GAIL/")
    gail_trainer = adversarial.GAIL(
        venv,
        expert_data=transitions,
        expert_batch_size=32,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
    )
    gail_trainer.train(total_timesteps=2048)

    # Train AIRL on expert data.
    logger.configure(tempdir_path / "AIRL/")
    airl_trainer = adversarial.AIRL(
        venv,
        expert_data=transitions,
        expert_batch_size=32,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
    )
    airl_trainer.train(total_timesteps=2048)
