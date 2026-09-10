# coding=utf-8
"""The observation pipeline was rewritten for speed; prove it did not move.

Each test here holds a copy of the implementation as it stood before the
optimization and asserts the current code reproduces it exactly.  These are
pure-Python/numpy comparisons -- the engine-level check that a whole rollout is
unchanged lives in gfootball/examples/observation_trace.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from gfootball.env import config as cfg
from gfootball.env import football_action_set
from gfootball.env import observation_rotation
from gfootball.env import puffer_env
from gfootball.env.wrappers import Simple115StateWrapper
from gfootball import curriculum


def reference_simple115(observation, fixed_positions):
  """Simple115StateWrapper.convert_observation before it was vectorized."""

  def do_flatten(obj):
    if type(obj) == list:
      return np.array(obj).flatten()
    return obj.flatten()

  final_obs = []
  for obs in observation:
    o = []
    if fixed_positions:
      for i, name in enumerate(['left_team', 'left_team_direction',
                                'right_team', 'right_team_direction']):
        o.extend(do_flatten(obs[name]))
        if len(o) < (i + 1) * 22:
          o.extend([-1] * ((i + 1) * 22 - len(o)))
    else:
      o.extend(do_flatten(obs['left_team']))
      o.extend(do_flatten(obs['left_team_direction']))
      o.extend(do_flatten(obs['right_team']))
      o.extend(do_flatten(obs['right_team_direction']))
    if len(o) < 88:
      o.extend([-1] * (88 - len(o)))
    o.extend(obs['ball'])
    o.extend(obs['ball_direction'])
    if obs['ball_owned_team'] == -1:
      o.extend([1, 0, 0])
    if obs['ball_owned_team'] == 0:
      o.extend([0, 1, 0])
    if obs['ball_owned_team'] == 1:
      o.extend([0, 0, 1])
    active = [0] * 11
    if obs['active'] != -1:
      active[obs['active']] = 1
    o.extend(active)
    game_mode = [0] * 7
    game_mode[obs['game_mode']] = 1
    o.extend(game_mode)
    final_obs.append(o)
  return np.array(final_obs, dtype=np.float32)


def reference_rotate_sticky_actions(sticky_actions_state, config):
  """observation_rotation.rotate_sticky_actions before it was memoized."""
  sticky_actions = football_action_set.get_sticky_actions(config)
  assert len(sticky_actions) == len(sticky_actions_state)
  action_to_state = {}
  for i in range(len(sticky_actions)):
    action_to_state[sticky_actions[i]] = sticky_actions_state[i]
  rotated = []
  for i in range(len(sticky_actions)):
    rotated.append(action_to_state[observation_rotation.flip_single_action(
        sticky_actions[i], config)])
  return rotated


def reference_episode_duration(level):
  """The game_duration formula as it was inlined in the curriculum scenario."""
  attackers, defenders, progress = curriculum.curriculum_state(level)
  near_goal = min(599, 119 + 40 * (attackers - 1 + defenders))
  return int(near_goal + (3000 - near_goal) * progress)


def make_player_observations(rng, n_left, n_right, views=2, per_view=11):
  """Observations shaped the way FootballEnv hands them to the wrapper.

  Several controlled players share one view of the pitch and differ only in
  which player each one controls; the right-hand side gets a second, flipped
  view.  That sharing is what the vectorized converter exploits.
  """
  observations = []
  for _ in range(views):
    shared = {
        'left_team': rng.uniform(-1, 1, (n_left, 2)),
        'left_team_direction': rng.uniform(-0.1, 0.1, (n_left, 2)),
        'right_team': rng.uniform(-1, 1, (n_right, 2)),
        'right_team_direction': rng.uniform(-0.1, 0.1, (n_right, 2)),
        'ball': rng.uniform(-1, 1, 3),
        'ball_direction': rng.uniform(-0.1, 0.1, 3),
        'ball_owned_team': int(rng.integers(-1, 2)),
        'game_mode': int(rng.integers(0, 7)),
    }
    for _ in range(per_view):
      view = dict(shared)
      view['active'] = int(rng.integers(-1, n_left))
      observations.append(view)
  return observations


class ObservationEquivalenceTest(absltest.TestCase):

  def test_simple115_matches_the_reference_implementation(self):
    rng = np.random.default_rng(0)
    for _ in range(200):
      observations = make_player_observations(
          rng, int(rng.integers(1, 12)), int(rng.integers(1, 12)),
          per_view=int(rng.integers(1, 12)))
      for fixed_positions in (True, False):
        actual = Simple115StateWrapper.convert_observation(
            observations, fixed_positions)
        expected = reference_simple115(observations, fixed_positions)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)

  def test_simple115_handles_the_full_22_agent_match(self):
    rng = np.random.default_rng(1)
    observations = make_player_observations(rng, 11, 11, views=2, per_view=11)
    self.assertLen(observations, 22)
    np.testing.assert_array_equal(
        Simple115StateWrapper.convert_observation(observations, True),
        reference_simple115(observations, True))

  def test_sticky_action_rotation_matches_the_reference(self):
    config = cfg.Config()
    for action_set in ('default', 'v2', 'full'):
      config['action_set'] = action_set
      width = len(football_action_set.get_sticky_actions(config))
      rng = np.random.default_rng(2)
      for _ in range(64):
        state = rng.integers(0, 2, width).astype(np.uint8)
        self.assertEqual(
            list(observation_rotation.rotate_sticky_actions(state, config)),
            list(reference_rotate_sticky_actions(state, config)))

  def test_rotating_twice_is_the_identity(self):
    config = cfg.Config()
    width = len(football_action_set.get_sticky_actions(config))
    rng = np.random.default_rng(3)
    for _ in range(64):
      state = list(rng.integers(0, 2, width).astype(np.uint8))
      once = observation_rotation.rotate_sticky_actions(state, config)
      twice = observation_rotation.rotate_sticky_actions(once, config)
      self.assertEqual(list(twice), state)

  def test_episode_duration_matches_the_inlined_formula(self):
    for level in range(curriculum.TOTAL_LEVELS):
      self.assertEqual(curriculum.curriculum_episode_duration(level),
                       reference_episode_duration(level),
                       'level {}'.format(level))

  def test_normalizing_only_active_rows_matches_normalizing_all(self):
    """puffer_env skips inactive rows; they are zeroed either way."""
    rng = np.random.default_rng(4)
    for frame_stack in (1, 4):
      for _ in range(50):
        observations = rng.uniform(
            -1, 1, (22, 115 * frame_stack)).astype(np.float32)
        for frame in range(frame_stack):
          base = frame * 115
          for row in range(22):
            for block in (0, 44):
              for player in range(11):
                if rng.random() < 0.25:
                  start = base + block + 2 * player
                  observations[row, start:start + 2] = -1
            observations[row, base + 97:base + 108] = 0
            observations[row, base + 97 + int(rng.integers(0, 11))] = 1

        mask = rng.random(22) < 0.4
        mask[int(rng.integers(0, 22))] = True

        expected = observations.copy()
        puffer_env.normalize_egocentric(expected)
        puffer_env.sort_players_by_distance(expected)
        expected[~mask] = 0

        actual = np.zeros_like(observations)
        rows = observations[mask].copy()
        puffer_env.normalize_egocentric(rows)
        puffer_env.sort_players_by_distance(rows)
        actual[mask] = rows

        np.testing.assert_array_equal(actual, expected)

  def test_centralized_rewards_accept_an_output_buffer(self):
    mask = np.zeros(22, dtype=bool)
    mask[[0, 3, 11]] = True
    out = np.empty(22, dtype=np.float32)
    np.testing.assert_array_equal(
        puffer_env.centralized_score_rewards(1.0, mask, out=out),
        puffer_env.centralized_score_rewards(1.0, mask))
    self.assertIs(puffer_env.centralized_score_rewards(1.0, mask, out=out), out)


if __name__ == '__main__':
  absltest.main()
