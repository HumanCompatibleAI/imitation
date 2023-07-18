=======
Experts
=======

The algorithms in the imitation library are all about learning from some kind of
expert.
In many cases this expert is a piece of software itself.
The `imitation` library natively supports experts trained using the
`stable-baselines3 <https://github.com/DLR-RM/stable-baselines3>`_ reinforcement
learning library.

For example, BC and DAgger can learn from an expert policy and the command line
interface of AIRL/GAIL allows one to specify an expert to sample demonstrations from.

In the :doc:`../getting-started/first-steps` tutorial, we first train an expert policy
using the stable-baselines3 library and then imitate it's behavior using
:doc:`../algorithms/bc`.
In practice, you may want to load a pre-trained policy for performance reasons.

Loading a policy from a file
----------------------------

The Python interface provides a :func:`~imitation.policies.serialize.load_policy`
function to which you pass a `policy_type`, a VecEnv and any extra kwargs to pass to the
corresponding policy loader.

.. code-block:: python

    import numpy as np
    from imitation.policies.serialize import load_policy
    from imitation.util import util

    venv = util.make_vec_env("your-env", n_envs=4, rng=np.random.default_rng())
    local_policy = load_policy("ppo", venv, path="path/to/model.zip")

To load a policy from disk, use either `ppo` or `sac` as the policy type.
The path is specified by `path` in the `loader_kwargs` and it should either point
to a zip file containing the policy or a directory containing a `model.zip` file that
was created by stable-baselines3.

In the command line interface the `expert.policy_type` and `expert.loader_kwargs`
parameters define the expert policy to load.
For example, to train AIRL on a PPO expert, you would use the following command:

.. code-block:: bash

    python -m imitation.scripts.train_adversarial airl \
        with expert.policy_type=ppo expert.loader_kwargs.path="path/to/model.zip"


Loading a policy from HuggingFace
---------------------------------

`HuggingFace <https://huggingface.co/>`_ is a popular repository for pre-trained models.

To load a stable-baselines3 policy from HuggingFace, use either `ppo-huggingface` or
`sac-huggingface` as the policy type.
By default, the policies are loaded from the
`HumanCompatibleAI organization <https://huggingface.co/HumanCompatibleAI>`_, but you
can override this by setting the `organization` parameter in the `loader_kwargs`.
When using the Python API, you also have to specify the environment name as `env_name`.

.. code-block:: python

    import numpy as np
    from imitation.policies.serialize import load_policy
    from imitation.util import util

    venv = util.make_vec_env("your-env", n_envs=4, rng=np.random.default_rng())
    remote_policy = load_policy(
        "ppo-huggingface",
        organization="your-org",
        env_name="your-env"
        )
    )

In the command line interface, the `env-name` is automatically injected into the
`loader_kwargs` and does not need to be defined explicitly.
In this example, to train AIRL on a PPO expert that was loaded from `your-org` on
HuggingFace:

.. code-block:: bash

    python -m imitation.scripts.train_adversarial airl \
        with expert.policy_type=ppo-huggingface expert.loader_kwargs.organization=your-org

Uploading a policy to HuggingFace
---------------------------------

The `huggingface-sb3 package <https://github.com/huggingface/huggingface_sb3>`_ provides
utilities to push your models to HuggingFace and load them from there.
Make sure to use the naming scheme helpers
`as described in the readme <https://github.com/huggingface/huggingface_sb3#case-5-i-want-to-automate-uploaddownload-from-the-hub>`_.
Otherwise, the loader will not be able to find your model in the repository.

For a convenient high-level interface to train RL models and upload them to HuggingFace,
we recommend using the
`rl-baselines3-zoo <https://github.com/DLR-RM/rl-baselines3-zoo/>`_.


Custom expert types
-------------------------

If you want to use a custom expert type, you can write a corresponding factory
function according to :py:func:`~imitation.policies.serialize.PolicyLoaderFn` and then
register it at the :py:data:`~imitation.policies.serialize.policy_registry`.
For example:

.. code-block:: python

    from imitation.policies.serialize import policy_registry
    from stable_baselines3.common import policies

    def my_policy_loader(venv, some_param: int) -> policies.BasePolicy:
        # load your policy here
        return policy

    policy_registry.register("my-policy", my_policy_loader)

Then, you can use `my-policy` as the `policy_type` in the command line interface or the
Python API:

.. code-block:: bash

    python -m imitation.scripts.train_adversarial airl \
        with expert.policy_type=my-policy expert.loader_kwargs.some_param=42
