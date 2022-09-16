===============
Loading Experts
===============

Some of the algorithms, such as DAgger, can use an expert policy to learn from.

In the :doc:`../getting-started/first-steps` tutorial, we trained our algorithm on a
policy that was trained in the same script.
In practice you want to load a pretrained policy for performance reasons.

The python interface provides a :func:`~imitation.policies.serialize.load_policy`
function to which you pass a `policy type` and any extra kwargs which are passed to the
policy loader that is responsible for the policy type.

When using the CLI interface, you set the `expert.policy_type` and
`expert.loader_kwargs` parameters.

There are a number of existing policy types, and you can also define your own in the
registry.

Loading a policy from a file
----------------------------

To load a policy from disk, use either `ppo` or `sac` as the policy type.
The path is specified as `path` to the `loader_kwargs` and it should either point
to a zip file containing the policy or a directory containing a `model.zip` file.

Loading a policy from HuggingFace
---------------------------------

`HuggingFace <https://huggingface.co/>`_ is a popular repository for pretrained models.

To load a policy from HuggingFace, use either `ppo-huggingface` or `sac-huggingface` as
the policy type.
By default, the policies are loaded from the HumanCompatibleAI repository, but you can
override this by setting the `organization` parameter in the `loader_kwargs`.
When using the python API, you also have to specify the environment name as `env_name`.

Uploading a policy to HuggingFace
---------------------------------

The `huggingface-sb3 package <https://github.com/huggingface/huggingface_sb3>`_ provides
utilities to push your own models to HuggingFace and load them from there.
Make sure to use the naming scheme helpers
`as described in the readme <https://github.com/huggingface/huggingface_sb3#case-5-i-want-to-automate-uploaddownload-from-the-hub>`_.
Otherwise the loader will not be able to find your model in the repository.

For a convenient high-level interface to train RL models and upload them to HuggingFace,
we recommend using the
`rl-baselines3-zoo <https://github.com/DLR-RM/rl-baselines3-zoo/>`_.
