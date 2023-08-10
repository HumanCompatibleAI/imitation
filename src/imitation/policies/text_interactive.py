import numpy as np

from imitation.data.rollout import generate_trajectories, make_min_episodes
from imitation.policies.base import InteractivePolicy


class TextInteractivePolicy(InteractivePolicy):
    """Text-based interactive policy.

    This policy is intended to be used with a text-based environment,
    such as the Toy Text environments in OpenAI Gym."""

    ACTION_MAP = {
        "w": 3,  # up
        "a": 0,  # left
        "s": 1,  # down
        "d": 2,  # right
    }

    def _render(self, obs: np.ndarray) -> None:
        """Print the current state of the environment to the console."""
        self.venv.render(mode="human")

    def _query_action(self, obs: np.ndarray) -> np.ndarray:
        """Query human for an action."""
        while True:
            print("Enter an action (w: up, a: left, s: down, d: right):")
            user_input = input().strip().lower()
            if user_input in self.ACTION_MAP:
                return np.array([self.ACTION_MAP[user_input]])
            else:
                print("Invalid input. Try again.")


if __name__ == "__main__":
    import numpy as np
    from gym.envs.toy_text.frozen_lake import generate_random_map

    from imitation.util.util import make_vec_env

    # todo: also test with gym.env
    env = make_vec_env(
        "FrozenLake-v1",
        rng=np.random.default_rng(),
        n_envs=1,  # easier to play :)
        env_make_kwargs=dict(
            desc=generate_random_map(size=4),
            is_slippery=False,
        ),
    )
    expert = TextInteractivePolicy(env)

    # rollout our interactive expert policy
    generate_trajectories(
        expert,
        env,
        sample_until=make_min_episodes(1),
        rng=np.random.default_rng(),
    )

    # import tempfile
    #
    # import numpy as np
    # from gym.envs.toy_text.frozen_lake import generate_random_map
    # from stable_baselines3.common.evaluation import evaluate_policy
    #
    # from imitation.algorithms import bc
    # from imitation.algorithms.dagger import SimpleDAggerTrainer
    # from imitation.util.util import make_vec_env

    # bc_trainer = bc.BC(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     rng=np.random.default_rng(),
    # )
    #
    # tmpdir = tempfile.mkdtemp(prefix="dagger_human_example_")
    # print(tmpdir)
    # dagger_trainer = SimpleDAggerTrainer(
    #     venv=env,
    #     scratch_dir=tmpdir,
    #     expert_policy=expert,
    #     bc_trainer=bc_trainer,
    #     rng=np.random.default_rng(),
    # )
    #
    # reward_before, _ = evaluate_policy(dagger_trainer.policy, env, 20)
    # print(f"{reward_before=}")
    #
    #
    # dagger_trainer.train(
    #     total_timesteps=10,
    #     rollout_round_min_timesteps=10,
    # )
    #
    # reward_after, _ = evaluate_policy(dagger_trainer.policy, env, 20)
    # print(f"{reward_after=}")
