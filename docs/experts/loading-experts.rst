===============
Loading Experts
===============

Some algorithms, such as DAgger, can learn from an expert policy.

In the :doc:`../getting-started/first-steps` tutorial, we first trained an expert and
then imitate it's behavior using :doc:`../algorithms/bc`.
In practice, you may want to load a pre-trained policy for performance reasons.

The Python interface provides a :func:`~imitation.policies.serialize.load_policy`
function to which you pass a `policy_type` and any extra kwargs to pass to the
corresponding policy loader.

While using the CLI interface, you set the `expert.policy_type` and
`expert.loader_kwargs` parameters.

There are several existing policy types, and you can also define your own in the
registry.

Loading a policy from a file
----------------------------

To load a policy from disk, use either `ppo` or `sac` as the policy type.
The path is specified by `path` in the `loader_kwargs` and it should either point
to a zip file containing the policy or a directory containing a `model.zip` file.

Loading a policy from HuggingFace
---------------------------------

`HuggingFace <https://huggingface.co/>`_ is a popular repository for pre-trained models.

To load a policy from HuggingFace, use either `ppo-huggingface` or `sac-huggingface` as
the policy type.
By default, the policies are loaded from the HumanCompatibleAI repository, but you can
override this by setting the `organization` parameter in the `loader_kwargs`.
When using the Python API, you also have to specify the environment name as `env_name`.

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
