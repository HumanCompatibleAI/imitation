"""Train GAIL or AIRL."""

import functools
import logging
import pathlib
from typing import Any, Mapping, Optional, Type

import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver

from imitation.algorithms.adversarial import airl as airl_algo
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_adversarial_ex
from imitation.scripts.ingredients import demonstrations, environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward, rl

logger = logging.getLogger("imitation.scripts.train_adversarial")


def save(trainer: common.AdversarialTrainer, save_path: pathlib.Path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.reward_train, save_path / "reward_train.pt")
    th.save(trainer.reward_test, save_path / "reward_test.pt")
    serialize.save_stable_model(
        save_path / "gen_policy",
        trainer.gen_algo,
    )


def _add_hook(ingredient: sacred.Ingredient) -> None:
    # This is an ugly hack around Sacred config brokenness.
    # Config hooks only apply to their current ingredient,
    # and cannot update things in nested ingredients.
    # So we have to apply this hook to every ingredient we use.
    @ingredient.config_hook
    def hook(config, command_name, logger):
        del logger
        path = ingredient.path
        if path == "train_adversarial":
            path = ""
        ingredient_config = sacred.utils.get_by_dotted_path(config, path)
        return ingredient_config["algorithm_specific"].get(command_name, {})

    # We add this so Sacred doesn't complain that algorithm_specific is unused
    @ingredient.capture
    def dummy_no_op(algorithm_specific):
        pass

    # But Sacred may then complain it isn't defined in config! So, define it.
    @ingredient.config
    def dummy_config():
        algorithm_specific = {}  # noqa: F841


for ingredient in [train_adversarial_ex, *train_adversarial_ex.ingredients]:
    _add_hook(ingredient)


@train_adversarial_ex.capture
def train_adversarial(
    _run,
    show_config: bool,
    algo_cls: Type[common.AdversarialTrainer],
    algorithm_kwargs: Mapping[str, Any],
    total_timesteps: int,
    checkpoint_interval: int,
    agent_path: Optional[str],
) -> Mapping[str, Mapping[str, float]]:
    """Train an adversarial-network-based imitation learning algorithm.

    Checkpoints:
        - AdversarialTrainer train and test RewardNets are saved to
           `f"{log_dir}/checkpoints/{step}/reward_{train,test}.pt"`
            where step is either the training round or "final".
        - Generator policies are saved to `f"{log_dir}/checkpoints/{step}/gen_policy/"`.

    Args:
        show_config: Print the merged config before starting training. This is
            analogous to the print_config command, but will show config after
            rather than before merging `algorithm_specific` arguments.
        algo_cls: The adversarial imitation learning algorithm to use.
        algorithm_kwargs: Keyword arguments for the `GAIL` or `AIRL` constructor.
        total_timesteps: The number of transitions to sample from the environment
            during training.
        checkpoint_interval: Save the discriminator and generator models every
            `checkpoint_interval` rounds and after training is complete. If 0,
            then only save weights after training is complete. If <0, then don't
            save weights at all.
        agent_path: Path to a directory containing a pre-trained agent. If
            provided, then the agent will be initialized using this stored policy
            (warm start). If not provided, then the agent will be initialized using
            a random policy.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations.
    """
    # This allows to specify total_timesteps and checkpoint_interval in scientific
    # notation, which is interpreted as a float by python.
    total_timesteps = int(total_timesteps)
    checkpoint_interval = int(checkpoint_interval)

    if show_config:
        # Running `train_adversarial print_config` will show unmerged config.
        # So, support showing merged config from `train_adversarial {airl,gail}`.
        sacred.commands.print_config(_run)

    custom_logger, log_dir = logging_ingredient.setup_logging()
    expert_trajs = demonstrations.get_expert_trajectories()

    with environment.make_venv() as venv:  # type: ignore[wrong-arg-count]
        reward_net = reward.make_reward_net(venv)
        relabel_reward_fn = functools.partial(
            reward_net.predict_processed,
            update_stats=False,
        )

        if agent_path is None:
            gen_algo = rl.make_rl_algo(venv, relabel_reward_fn=relabel_reward_fn)
        else:
            gen_algo = rl.load_rl_algo_from_path(
                agent_path=agent_path,
                venv=venv,
                relabel_reward_fn=relabel_reward_fn,
            )

        logger.info(f"Using '{algo_cls}' algorithm")
        algorithm_kwargs = dict(algorithm_kwargs)
        for k in ("shared", "airl", "gail"):
            # Config hook has copied relevant subset of config to top-level.
            # But due to Sacred limitations, cannot delete the rest of it.
            # So do that here to avoid passing in invalid arguments to constructor.
            if k in algorithm_kwargs:
                del algorithm_kwargs[k]
        trainer = algo_cls(
            venv=venv,
            demonstrations=expert_trajs,
            gen_algo=gen_algo,
            log_dir=log_dir,
            reward_net=reward_net,
            custom_logger=custom_logger,
            **algorithm_kwargs,
        )

        def callback(round_num: int, /) -> None:
            if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
                save(trainer, log_dir / "checkpoints" / f"{round_num:05d}")

        trainer.train(total_timesteps, callback)
        imit_stats = policy_evaluation.eval_policy(trainer.policy, trainer.venv_train)

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save(trainer, log_dir / "checkpoints" / "final")

    return {
        "imit_stats": imit_stats,
        "expert_stats": rollout.rollout_stats(expert_trajs),
    }


@train_adversarial_ex.command
def gail():
    return train_adversarial(algo_cls=gail_algo.GAIL)


@train_adversarial_ex.command
def airl():
    return train_adversarial(algo_cls=airl_algo.AIRL)


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_adversarial"
    observer = FileStorageObserver(observer_path)
    train_adversarial_ex.observers.append(observer)
    train_adversarial_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
