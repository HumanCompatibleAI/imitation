import gym
import numpy as np
import torch

from imitation.rewards.reward_distance.collections import ModelCollection
from imitation.rewards.reward_distance.reward_models import (ZeroRewardModel,
                                          ConstantRewardModel,
                                          GroundTruthRewardModelForHC,
                                          RewardModelWrapperForImitation,
                                          RandomRewardModel)
from imitation.rewards.reward_distance.mujoco_sampler import MujocoTransitionSampler
from imitation.rewards.reward_distance.transition_sampler import UniformlyRandomActionSampler
from imitation.rewards.reward_distance.epic import EPIC
from imitation.rewards.reward_distance.distances import compute_distance_between_reward_pairs
from imitation.rewards.reward_distance.distances import compute_pearson_distance
from imitation.rewards.serialize import load_reward

# Give path to appropriate reward directory
imitation_reward_model_path = "/mnt/models/exp-0607-cheetah_train_pc_then_rl_sac/2022-06-07T17:04:38+00:00/EXP-cheetah_train_pc_then_rl_sac/inner_ef0e9_00000_0_normalize_output_layer=<class 'imitation.util.networks.RunningNorm'>,pc_seed=0,named_configs=['dmc_cheetah_run_2022-06-07_17-04-46/output/dmc_cheetah_run/seed/0/checkpoints/final/reward_net.pt"


def get_action_dim(make_env_fn):
    """An auxillary function that simply returns action dimension
    of given mujoco environment"""
    env = make_env_fn()
    n = env.action_space.shape[0]
    del env
    return n

def get_data_for_epic(make_env_fn, states_to_sample=5000):
    single_action_sampler = UniformlyRandomActionSampler(num_actions=1,
                                                  max_magnitude=1,
                                                  action_dim=get_action_dim(make_env_fn))
    transition_sampler = MujocoTransitionSampler(make_env_fn,
                                                 single_action_sampler,
                                                 num_workers=2)
    env = make_env_fn()
    # Lets sample some transitions for computing rewards
    init_states = torch.from_numpy(np.concatenate(
            [env.observation_space.sample()[None,...] for _ in range(states_to_sample)],
            axis=0))
    actions, next_states, _ = transition_sampler.sample(init_states)
    actions, next_states = torch.squeeze(actions), torch.squeeze(next_states)
    return init_states, actions, next_states


def main():
    # Choose the enviroment to test on
    env_name = "HalfCheetah-v3"
    # We may need to make this environment several time, so lets make this
    # into a function call
    make_env_fn = lambda :gym.make(env_name, exclude_current_positions_from_observation=False)
    env = make_env_fn()
    reward_model1 = RandomRewardModel()

    # TODO: Verify that load_reward function works correctly
    # Specifically, I am worried about
    # This function does not throw an error for non-default gym
    # environment even though I expect that this was trained on a
    # default gym environment
    # I also do not really understand what reward_type argument does here
    imitation_reward_fn = load_reward(reward_type="RewardNet_unshaped",
                                      reward_path=imitation_reward_model_path,
                                      venv = make_env_fn())
    reward_model2 = RewardModelWrapperForImitation(imitation_reward_fn)
    reward_model3 = RewardModelWrapperForImitation(imitation_reward_fn)
    ground_truth_rm = GroundTruthRewardModelForHC(make_env_fn)
    reward_models = ModelCollection(dict(random_model=reward_model1,
                                         IL_model1=reward_model2,
                                         IL_model2=reward_model3,
                                         ground_truth_rm=ground_truth_rm))

    # Define how the epic should sample transitions
    action_sampler = UniformlyRandomActionSampler(num_actions=20,
                                                  max_magnitude=1,
                                                  action_dim=get_action_dim(make_env_fn))
    transition_sampler = MujocoTransitionSampler(make_env_fn,
                                                 action_sampler,
                                                 num_workers=2)

    # Get data for epic
    # Currently I am doing this on a randomly sampled dataset
    # However, an already sampled dataset like demonstrations could
    # also be used here
    init_states, actions, next_states = get_data_for_epic(make_env_fn)

    rewards = EPIC().compute_canonical_rewards(models=reward_models,
                                           states=init_states,
                                           actions=actions,
                                           next_states=next_states,
                                           terminals=torch.zeros_like(init_states),
                                           transition_sampler=transition_sampler,
                                           discount=0.99)

    distances = compute_distance_between_reward_pairs(rewards, compute_pearson_distance)
    print(np.round(distances.distances, 4))

    # Optionally you can save a heatmap visualization of distance matrix
    distances.visualize("distances.png", title="Epic Sample")

if __name__ == '__main__':
    main()