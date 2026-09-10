"""PufferLib-native interface for headless 22-player self-play."""

from collections import deque
from functools import partial

import gymnasium
import numpy as np
import psutil
import pufferlib
import pufferlib.vector

import gfootball.env as football_env
from gfootball.env import football_action_set
from gfootball.curriculum import (
    ADVANTAGE_ENV_NAME, ATTACKER_ORDER, DEFENDER_ORDER, TOTAL_LEVELS,
    advantage_for_level, curriculum_state)


# Engine units converted to simple115v2's per-step coordinate system.
_RELATIVE_POSITION_MAX = np.array([
    2 * (55.0 + 2.55) / 54.4,
    2 * 36.0 / 83.6,
], dtype=np.float32)
_RELATIVE_PLAYER_STEP_MAX = np.array([
    2 * 8.0 / 10.0 / 54.4,
    2 * 8.0 / 10.0 / 83.6,
], dtype=np.float32)
_BALL_STEP_SCALE = np.array([
    45.0 / 10.0 / 54.4,
    45.0 / 10.0 / 83.6,
    45.0 / 10.0,
], dtype=np.float32)
_GOAL_HEIGHT = 2.5


def _soft_scale(values, scale):
  """Bound values without clipping when the simulator has no hard maximum."""
  values /= scale + np.abs(values)


def normalize_egocentric(observations):
  """Center simple115v2 physical features on each controlled player."""
  frames = observations.reshape(-1, 115)
  own_positions = frames[:, :22].reshape(-1, 11, 2)
  own_directions = frames[:, 22:44].reshape(-1, 11, 2)
  opponent_positions = frames[:, 44:66].reshape(-1, 11, 2)
  opponent_directions = frames[:, 66:88].reshape(-1, 11, 2)
  active = frames[:, 97:108].argmax(axis=1)
  rows = np.arange(frames.shape[0])
  ego_position = own_positions[rows, active].copy()
  ego_direction = own_directions[rows, active].copy()

  for positions, directions in (
      (own_positions, own_directions),
      (opponent_positions, opponent_directions)):
    missing = np.all(positions == -1, axis=-1)
    positions -= ego_position[:, None, :]
    positions /= _RELATIVE_POSITION_MAX
    directions -= ego_direction[:, None, :]
    directions /= _RELATIVE_PLAYER_STEP_MAX
    positions[missing] = -1
    directions[missing] = -1

  frames[:, 88:90] -= ego_position
  frames[:, 88:90] /= _RELATIVE_POSITION_MAX
  _soft_scale(frames[:, 90], _GOAL_HEIGHT)
  frames[:, 91:93] -= ego_direction
  _soft_scale(frames[:, 91:94], _BALL_STEP_SCALE)
  return observations


def sort_players_by_distance(observations):
  """Order teammates and opponents by distance from the controlled player.

  simple115v2 lists all 22 players in a fixed slot order, so one policy shared
  across 22 agents has to learn permutation invariance from data: the same
  teammate appears in a different slot depending on who is being controlled.
  Sorting each block by distance makes slot 0 always "me", slot 1 always "my
  nearest teammate", and so on, which is the same information in a form the
  network does not have to spend capacity untangling.

  Must run after normalize_egocentric, which is what makes positions relative.
  """
  frames = observations.reshape(-1, 115)
  rows = np.arange(frames.shape[0])[:, None]
  for base in (0, 44):
    positions = frames[:, base:base + 22].reshape(-1, 11, 2)
    directions = frames[:, base + 22:base + 44].reshape(-1, 11, 2)
    distance = np.linalg.norm(positions, axis=-1)
    # Absent players are the -1 sentinel; keep them last so they never
    # displace a real player from a near slot.
    distance[np.all(positions == -1, axis=-1)] = np.inf
    order = np.argsort(distance, axis=1)
    frames[:, base:base + 22] = positions[rows, order].reshape(-1, 22)
    frames[:, base + 22:base + 44] = directions[rows, order].reshape(-1, 22)
  return observations


def centralized_score_rewards(score_reward, active_mask, out=None):
  """Share the zero-sum match score with every active player on each team."""
  active_mask = np.asarray(active_mask, dtype=bool)
  if active_mask.shape != (22,):
    raise ValueError('active_mask must have shape (22,)')
  score_reward = float(score_reward)
  rewards = np.empty(22, dtype=np.float32) if out is None else out
  rewards[:11] = score_reward
  rewards[11:] = -score_reward
  rewards[~active_mask] = 0
  return rewards


class FootballPufferEnv(pufferlib.PufferEnv):
  """One GRF match exposed as 22 PufferLib agents."""

  def __init__(self, env_name='11_vs_11_curriculum', render=False, buf=None,
               seed=0, frame_stack=4, curriculum_levels=TOTAL_LEVELS,
               curriculum_window=20, curriculum_success_threshold=0.6,
               attacker_only_levels=0, curriculum_level_value=None,
               curriculum_evaluation=False, sort_players=True):
    if frame_stack not in (1, 4):
      raise ValueError('frame_stack must be 1 or 4')
    if curriculum_levels < 2:
      raise ValueError('curriculum_levels must be at least 2')
    if curriculum_window < 1:
      raise ValueError('curriculum_window must be positive')
    if not 0 < curriculum_success_threshold <= 1:
      raise ValueError('curriculum_success_threshold must be in (0, 1]')
    if not 0 <= attacker_only_levels <= curriculum_levels:
      raise ValueError('attacker_only_levels must be within curriculum')
    self.num_envs = 1
    self.num_agents = 22
    self.agents_per_batch = self.num_agents
    self.single_observation_space = gymnasium.spaces.Box(
        low=-1, high=1, shape=(115 * frame_stack,), dtype=np.float32)
    self.single_action_space = gymnasium.spaces.Discrete(
        len(football_action_set.action_set_dict['default']))
    super().__init__(buf)

    self._env_name = env_name
    self._render = render
    self._seed = int(seed)
    self._frame_stack = frame_stack
    self._sort_players = bool(sort_players)
    self._curriculum_levels = int(curriculum_levels)
    self._curriculum_level = 0
    self._curriculum_results = deque(maxlen=int(curriculum_window))
    self._curriculum_success_threshold = float(curriculum_success_threshold)
    self._attacker_only_levels = int(attacker_only_levels)
    self._curriculum_level_value = curriculum_level_value
    self._curriculum_evaluation = bool(curriculum_evaluation)
    self._advantage_mode = env_name == ADVANTAGE_ENV_NAME
    self._curriculum_enabled = env_name in (
        '11_vs_11_curriculum', ADVANTAGE_ENV_NAME)
    self._episode_level = 0
    self._episode_attackers = 1
    self._episode_template = 0
    self._attacking_left = True
    self._active_mask = np.ones(self.num_agents, dtype=bool)
    self._episode_active_mask = np.ones(self.num_agents, dtype=bool)
    self._step_actions = np.zeros(self.num_agents, dtype=np.int64)
    self._step_rewards = np.zeros(self.num_agents, dtype=np.float32)
    self._env = self._make_env()
    self._episode_return = np.zeros(2, dtype=np.float32)
    self._episode_length = 0
    # Goals are far too rare to read progress from: a whole evaluation yields
    # a handful. These two move every step, so they show whether play is
    # improving before it starts converting.
    self._possession_steps = 0
    self._advance_sum = 0.0

  def _make_env(self):
    return football_env.create_environment(
        env_name=self._env_name,
        representation='simple115v2',
        rewards='scoring',
        render=self._render,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        write_video=False,
        stacked=self._frame_stack == 4,
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=11,
        extra_players=None,
        other_config_options={
            'action_set': 'default',
            'advantage': advantage_for_level(
                self._curriculum_level, self._curriculum_levels),
            'curriculum_level': self._curriculum_level,
            'curriculum_levels': self._curriculum_levels,
            'curriculum_evaluation': self._curriculum_evaluation,
            'fast_mode': not self._render,
            'game_engine_random_seed': self._seed,
            # simple115v2 does not encode sticky actions, and producing them
            # costs ten engine queries per controlled player per step.
            'needs_sticky_actions': False,
            'real_time': False,
        })

  def _reset_match(self):
    if self._curriculum_level_value is not None:
      shared_level = self._curriculum_level_value.value
      if shared_level != self._curriculum_level:
        self._curriculum_results.clear()
        self._curriculum_level = shared_level
    self._env.unwrapped._config['curriculum_level'] = self._curriculum_level
    if self._advantage_mode:
      self._env.unwrapped._config['advantage'] = advantage_for_level(
          self._curriculum_level, self._curriculum_levels)
    observations = self._env.reset()
    raw_config = self._env.unwrapped._config
    ball_x = raw_config.ScenarioConfig().ball_position[0]
    self._attacking_left = ball_x > 0
    self._episode_level = self._curriculum_level
    if self._curriculum_enabled:
      self._episode_attackers = int(
          raw_config._values['curriculum_episode_attackers'])
      self._episode_template = int(
          raw_config._values['curriculum_episode_template'])
    self._set_active_players()
    return observations

  def _set_active_players(self):
    self._active_mask.fill(True)
    if not self._curriculum_enabled or self._advantage_mode:
      # The advantage schedule keeps every player on the pitch at every level.
      return
    _, defenders, _ = curriculum_state(self._episode_level)
    self._active_mask.fill(False)
    attacking_offset = 0 if self._attacking_left else 11
    defending_offset = 11 - attacking_offset
    self._active_mask[
        attacking_offset + np.asarray(
            ATTACKER_ORDER[:self._episode_attackers], dtype=np.intp)] = True
    if self._episode_level >= self._attacker_only_levels:
      self._active_mask[defending_offset] = True
      self._active_mask[
          defending_offset + np.asarray(
              DEFENDER_ORDER[:defenders], dtype=np.intp)] = True

  def _record_curriculum_result(self, success):
    self._curriculum_results.append(float(success))
    success_rate = float(np.mean(self._curriculum_results))
    advanced = (
        len(self._curriculum_results) == self._curriculum_results.maxlen and
        success_rate >= self._curriculum_success_threshold and
        self._curriculum_level_value is None and
        self._curriculum_level < self._curriculum_levels - 1)
    if advanced:
      self._curriculum_level += 1
      self._curriculum_results.clear()
    return success_rate, advanced

  def _write_observations(self, observations):
    observations = np.asarray(observations, dtype=np.float32)
    if observations.shape != self.observations.shape:
      raise ValueError('Expected observations with shape {}, got {}'.format(
          self.observations.shape, observations.shape))
    # Inactive rows are zeroed either way, so normalizing and sorting them is
    # pure waste: the early curriculum levels control 1 of 22 players, and
    # sort_players_by_distance is the most expensive numpy in the step loop.
    active = self._active_mask
    self.observations[~active] = 0
    rows = observations[active]
    if rows.size:
      normalize_egocentric(rows)
      if self._sort_players:
        sort_players_by_distance(rows)
      self.observations[active] = rows

  def reset(self, seed=None):
    if seed is not None and int(seed) != self._seed:
      self._env.close()
      self._seed = int(seed)
      self._env = self._make_env()
    self._write_observations(self._reset_match())
    self.rewards.fill(0)
    self.terminals.fill(False)
    self.truncations.fill(False)
    self._episode_return.fill(0)
    self._episode_length = 0
    self._possession_steps = 0
    self._advance_sum = 0.0
    return self.observations, []

  def step(self, actions):
    # _reset_match rewrites _active_mask mid-step, so the mask that governs
    # this transition has to be snapshotted first.  Reuse the buffers rather
    # than allocating three arrays on every step.
    episode_active_mask = self._episode_active_mask
    np.copyto(episode_active_mask, self._active_mask)
    np.copyto(self._step_actions, np.asarray(actions).reshape(self.num_agents),
              casting='unsafe')
    actions = self._step_actions
    actions[~episode_active_mask] = 0
    observations, _, done, info = self._env.step(actions)
    rewards = centralized_score_rewards(
        info['score_reward'], episode_active_mask, out=self._step_rewards)
    for team, team_slice in enumerate((slice(0, 11), slice(11, 22))):
      team_active = episode_active_mask[team_slice]
      if team_active.any():
        self._episode_return[team] += rewards[team_slice][team_active].mean()
    self._episode_length += 1
    frame = np.asarray(observations, dtype=np.float32).reshape(
        self.num_agents, -1)[0, -115:]
    # simple115v2: 88 is ball x, 94:97 is a none/left/right ownership one-hot.
    owner = int(np.argmax(frame[94:97]))
    attacking_owner = 1 if self._attacking_left else 2
    if owner == attacking_owner:
      self._possession_steps += 1
    # +1 means the ball is on the goal being attacked, -1 the other end.
    self._advance_sum += float(
        frame[88] if self._attacking_left else -frame[88])
    self.rewards[:] = rewards
    self.terminals.fill(done)
    self.truncations.fill(False)

    infos = []
    if done:
      attacking_return = self._episode_return[
          0 if self._attacking_left else 1]
      _, active_defenders, distance_progress = curriculum_state(
          self._episode_level)
      curriculum_success = attacking_return > 0
      success_rate, advanced = self._record_curriculum_result(
          curriculum_success) if self._curriculum_enabled else (0.0, False)
      infos.append({
          'curriculum_advanced': float(advanced),
          'curriculum_level': float(self._episode_level),
          'curriculum_success': float(curriculum_success),
          'curriculum_success_rate': success_rate,
          'curriculum_active_attackers': float(self._episode_attackers),
          'curriculum_active_defenders': float(active_defenders + 1),
          'curriculum_learning_goalkeeper': float(
              self._episode_level >= self._attacker_only_levels),
          'curriculum_distance_progress': distance_progress,
          'curriculum_template': float(self._episode_template),
          'curriculum_evaluation': float(self._curriculum_evaluation),
          'episode_length': self._episode_length,
          'possession_fraction': (
              self._possession_steps / max(1, self._episode_length)),
          'mean_ball_advance': (
              self._advance_sum / max(1, self._episode_length)),
          'left_episode_return': float(self._episode_return[0]),
          'right_episode_return': float(self._episode_return[1]),
          'score_reward': float(info['score_reward']),
      })
      observations = self._reset_match()
      self._episode_return.fill(0)
      self._episode_length = 0
      self._possession_steps = 0
      self._advance_sum = 0.0
    self._write_observations(observations)
    return (self.observations, self.rewards, self.terminals,
            self.truncations, infos)

  def close(self):
    if self._env is not None:
      self._env.close()
      self._env = None


def make_vector_env(num_envs=None, num_workers=None, batch_size=None,
                    reserved_cpus=2, seed=0, centralized_curriculum=False,
                    envs_per_worker=1, **env_kwargs):
  """Create headless matches across PufferLib multiprocessing workers.

  With ``envs_per_worker`` above 1 the workers hold more matches than any one
  batch needs, which lets PufferLib collect the next batch while the policy is
  still consuming the current one.  That matters here because a GRF reset costs
  far more than a step and the early curriculum levels end episodes every ~119
  steps: with one env per worker and a full-width batch, every reset stalls
  every other worker.
  """
  if num_workers is None:
    available_cpus = (len(psutil.Process().cpu_affinity())
                      if hasattr(psutil.Process(), 'cpu_affinity') else
                      psutil.cpu_count(logical=False) or 1)
    num_workers = max(1, available_cpus - reserved_cpus)
  envs_per_worker = max(1, int(envs_per_worker))
  num_envs = num_workers * envs_per_worker if num_envs is None else num_envs
  batch_size = num_workers if batch_size is None else batch_size
  if num_envs % num_workers:
    raise ValueError('num_envs must be a multiple of num_workers')
  if batch_size % envs_per_worker:
    raise ValueError('batch_size must be a multiple of envs_per_worker')
  if centralized_curriculum:
    if env_kwargs.get('curriculum_level_value') is not None:
      raise ValueError('centralized curriculum already has a shared level')
    from multiprocessing import RawValue
    env_kwargs['curriculum_level_value'] = RawValue('i', 0)
  vecenv = pufferlib.vector.make(
      partial(FootballPufferEnv, **env_kwargs),
      backend=pufferlib.vector.Multiprocessing,
      num_envs=num_envs,
      num_workers=num_workers,
      batch_size=batch_size,
      zero_copy=True,
      seed=seed)
  vecenv.curriculum_level_value = env_kwargs.get('curriculum_level_value')
  return vecenv
