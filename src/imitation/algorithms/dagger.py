"""DAgger. Interactively trains policy by collecting some demonstrations, doing
BC, collecting more demonstrations, doing BC again, etc. Initially the
demonstrations just come from the expert's policy; over time, they shift to be
drawn more and more from the imitator's policy."""

import datetime
import os

import cloudpickle
import numpy as np
import tensorflow as tf

from imitation.algorithms.bc import BCTrainer, set_tf_vars
from imitation.util import rollout


def linear_beta_schedule(rampdown_rounds):
  """Schedule that linearly anneals beta from 1 to 0 over `rampdown_rounds`
  rounds of interaction (numbered 0, 1, 2, ...)."""
  assert rampdown_rounds > 0

  def linear_beta(i):
    assert i >= 0
    return min(1, max(0, (rampdown_rounds - i) / rampdown_rounds))

  return linear_beta


def save_trajectory(npz_path, trajectory):
  save_dir = os.path.dirname(npz_path)
  if save_dir:
    os.makedirs(save_dir, exist_ok=True)
  assert isinstance(trajectory, rollout.Trajectory)
  np.savez_compressed(npz_path, **trajectory._asdict())


def load_trajectory(npz_path):
  np_data = np.load(npz_path, allow_pickle=True)
  return rollout.Trajectory(**dict(np_data.items()))


class InteractiveTrajectoryCollector:
  """Wrapper around the .step()/.reset() of an env that allows DAgger to inject
  its own actions and save trajectories."""
  def __init__(self, env, get_robot_act, beta, save_dir, save_suffix):
    self.get_robot_act = get_robot_act
    assert 0 <= beta <= 1
    self.beta = beta
    self.env = env
    self.traj_accum = rollout._TrajectoryAccumulator()
    self.done_before = False
    self.save_suffix = save_suffix
    self.save_dir = save_dir
    self._last_obs = None
    self._done_before = True
    self._is_reset = False

  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)

  def reset(self):
    self.traj_accum.reset(0)
    obs = self.env.reset()
    self._last_obs = obs
    self.traj_accum.add_step(0, {'obs': obs})
    self._done_before = False
    self._is_reset = True
    return obs

  def step(self, user_action):
    assert self._is_reset, "call .reset() before .step()"

    # Replace the given action with a robot action 100*(1-beta)% of the time.
    if np.random.uniform(0, 1) > self.beta:
      actual_act = self.get_robot_act(self._last_obs)
    else:
      actual_act = user_action

    # actually step the env & record data as appropriate
    next_obs, reward, done, info = self.env.step(actual_act)
    self._last_obs = next_obs
    self.traj_accum.add_step(0, {
        'acts': user_action,
        'obs': next_obs,
        'rews': reward,
        'infos': info,
    })

    # if we're finished, then save the trajectory & print a message
    if done and not self._done_before:
      trajectory = self.traj_accum.finish_trajectory(0)
      now = datetime.datetime.now()
      date_str = now.strftime('%FT%k:%M:%S')
      trajectory_path = os.path.join(
          self.save_dir, 'dagger-demo-' + date_str + self.save_suffix)
      print(f"Saving demo at '{trajectory_path}'")
      save_trajectory(trajectory_path, trajectory)

    if done:
      # record the fact that we're already done to avoid saving demo over and
      # over until the user resets
      self._done_before = True

    return next_obs, reward, done, info


class NeedDemosException(Exception):
  """Raised when demonstrations need to be collected for the current round
  before continuing."""
  pass


class DAggerTrainer:
  """Helper class for interactively training with DAgger. Essentially just
  BCTrainer with some helpers for incrementally resuming training and
  interpolating between demonstrator/learnt policies. Stores files in a scratch
  directory with the following structure:

      scratch-dir-name/
          checkpoint-001.pkl
          checkpoint-002.pkl
          …
          checkpoint-XYZ.pkl
          checkpoint-latest.pkl
          demos/
              round-000/
                  demos_round_000_000.pkl.gz
                  demos_round_000_001.pkl.gz
                  …
              round-001/
                  demos_round_001_000.pkl.gz
                  …
              …
              round-XYZ/
                  …

  Args:
    ???"""
  SAVE_ATTRS = ('round_num', )
  DEMO_SUFFIX = '.npz'

  def __init__(self, env, scratch_dir, beta_schedule=None, **bc_kwargs):
    # for pickling
    self.__init_args = locals()
    self.__init_args.update(bc_kwargs)
    del self.__init_args['self']
    del self.__init_args['bc_kwargs']

    if beta_schedule is None:
      beta_schedule = linear_beta_schedule(15)
    self.beta_schedule = beta_schedule
    self.scratch_dir = scratch_dir
    self.env = env
    self.round_num = 0
    self.bc_kwargs = bc_kwargs
    self._loaded_demos = False

    self._build_graph()

  def _build_graph(self):
    with tf.variable_scope('dagger'):
      self.bc_trainer = BCTrainer(self.env,
                                  expert_demos=None,
                                  **self.bc_kwargs)
      self.__vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=tf.get_variable_scope().name)

  def _load_all_demos(self):
    num_demos_by_round = []
    all_demos = []
    for round_num in range(self.round_num + 1):
      round_dir = self._demo_dir_path_for_round(round_num)
      demo_paths = self._get_demo_paths(round_dir)
      all_demos.extend(load_trajectory(p) for p in demo_paths)
      num_demos_by_round.append(len(demo_paths))
    demo_transitions = rollout.flatten_trajectories(all_demos)
    return demo_transitions, num_demos_by_round

  def _get_demo_paths(self, round_dir):
    return [
        os.path.join(round_dir, p) for p in os.listdir(round_dir)
        if p.endswith(self.DEMO_SUFFIX)
    ]

  def _demo_dir_path_for_round(self, round_num=None):
    if round_num is None:
      round_num = self.round_num
    return os.path.join(self.scratch_dir, 'demos', f'round-{round_num:03d}')

  def _check_has_latest_demos(self):
    demo_dir = self._demo_dir_path_for_round()
    demo_paths = self._get_demo_paths(demo_dir) \
        if os.path.isdir(demo_dir) else []
    if len(demo_paths) == 0:
      raise NeedDemosException(
          f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
          f"Maybe you need to collect some demos? See "
          f".get_trajectory_collector()")

  def _try_load_demos(self):
    self._check_has_latest_demos()
    if not self._loaded_demos:
      transitions, num_demos = self._load_all_demos()
      print(f"Loaded {sum(num_demos)} demos from {len(num_demos)} rounds")
      self.bc_trainer.set_expert_dataset(transitions)
      self._loaded_demos = True

  @property
  def _sess(self):
    return self.bc_trainer.sess

  def extend_and_update(self, **train_kwargs):
    """Load new transitions (if necessary), train the model for a while, and
    advance the round counter."""
    print("Loading demonstrations")
    self._try_load_demos()
    print(f"Training at round {self.round_num}")
    self.bc_trainer.train(**train_kwargs)
    self.round_num += 1
    print(f"New round number is {self.round_num}")
    return self.round_num

  def get_trajectory_collector(self):
    save_dir = self._demo_dir_path_for_round()
    beta = self.beta_schedule(self.round_num)

    def get_robot_act(obs):
      (act, ), _, _, _ = self.bc_trainer.policy.step(obs[None])
      return act

    return InteractiveTrajectoryCollector(env=self.env,
                                          get_robot_act=get_robot_act,
                                          beta=beta,
                                          save_dir=save_dir,
                                          save_suffix=self.DEMO_SUFFIX)

  def save_trainer(self):
    os.makedirs(self.scratch_dir, exist_ok=True)
    # save full trainer checkpoints
    checkpoint_paths = [
        os.path.join(self.scratch_dir, f'checkpoint-{self.round_num:03d}.pkl'),
        os.path.join(self.scratch_dir, 'checkpoint-latest.pkl'),
    ]
    saved_attrs = {
        attr_name: getattr(self, attr_name)
        for attr_name in self.SAVE_ATTRS
    }
    snapshot_dict = {
        'init_args': self.__init_args,
        'variable_values': self._sess.run(self.__vars),
        'saved_attrs': saved_attrs,
    }
    for checkpoint_path in checkpoint_paths:
      with open(checkpoint_path, 'wb') as fp:
        cloudpickle.dump(snapshot_dict, fp)

    # save policies separately for convenience
    policy_paths = [
        os.path.join(self.scratch_dir, f'policy-{self.round_num:03d}.pkl'),
        os.path.join(self.scratch_dir, 'policy-latest.pkl'),
    ]
    for policy_path in policy_paths:
      self.save_policy(policy_path)

    return checkpoint_paths[-1]

  @classmethod
  def reconstruct_trainer(cls, scratch_dir):
    checkpoint_path = os.path.join(scratch_dir, 'checkpoint-latest.pkl')
    with open(checkpoint_path, 'rb') as fp:
      saved_trainer = cloudpickle.load(fp)
    # reconstruct from old init args
    rv = cls(**saved_trainer['init_args'])
    # set TF variables
    set_tf_vars(values=saved_trainer['variable_values'],
                tf_vars=rv.__vars,
                sess=rv.bc_trainer.sess)
    for attr_name in cls.SAVE_ATTRS:
      attr_value = saved_trainer['saved_attrs'][attr_name]
      setattr(rv, attr_name, attr_value)
    return rv

  def save_policy(self, *args, **kwargs):
    """Save the current policy only, and not the rest of the trainer
      parameters. Refer to docs for `BCTrainer.save_policy`."""
    return self.bc_trainer.save_policy(*args, **kwargs)

  @staticmethod
  def reconstruct_policy(*args, **kwargs):
    """Reconstruct a policy saved with `save_policy()`. Alias for
      `BCTrainer.reconstruct_policy()`."""
    return BCTrainer.reconstruct_policy(*args, **kwargs)
