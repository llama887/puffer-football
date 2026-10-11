"""Tests for the defending side (gfootball/env/defence.py); no game engine.

Run with `python -m gfootball.env.defence_test` (or pytest).
"""

import os
import tempfile
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

from gfootball.env import defence
from gfootball.env.puffer_env import (
    normalize_egocentric, sort_players_by_distance)
from gfootball.env.puffer_policy import FootballPolicy, save_policy_snapshot

ACTIONS = 19


def _spaces(features):
  return SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-1, high=1, shape=(features,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(ACTIONS))


def _save(path, features=115, always=None, seed=0):
  """A snapshot reading `features`; with `always`, it picks that action."""
  torch.manual_seed(seed)
  policy = FootballPolicy(_spaces(features), hidden_size=16)
  if always is not None:
    with torch.no_grad():
      policy.actor.weight.zero_()
      policy.actor.bias.fill_(-50.0)
      policy.actor.bias[always] = 50.0
  save_policy_snapshot(policy, path)


def _frames(seed=0):
  """Eleven plausible simple115v2 frames (one active player each)."""
  rng = np.random.default_rng(seed)
  frames = rng.uniform(-0.4, 0.4, (11, 115)).astype(np.float32)
  frames[:, 97:108] = np.eye(11, dtype=np.float32)
  return frames


def test_random_and_idle_need_no_snapshot():
  """Random draws uniform actions from a seeded stream; idle does nothing."""
  rows, frames = np.zeros((11, 300), np.float32), _frames()
  random = defence.Defence(defence.RANDOM, ACTIONS, seed=3)
  drawn = np.concatenate([random.act(rows, frames) for _ in range(50)])
  assert drawn.shape == (550,)
  assert len(set(drawn.tolist())) > 15
  assert np.all((drawn >= 0) & (drawn < ACTIONS))
  random.reseed(3)
  replay = np.concatenate([random.act(rows, frames) for _ in range(50)])
  np.testing.assert_array_equal(drawn, replay)
  idle = defence.Defence(defence.IDLE, ACTIONS, seed=3)
  idle.start_episode()
  np.testing.assert_array_equal(idle.act(rows, frames), np.zeros(11))


def test_snapshot_is_reloaded_only_at_an_episode_start_after_replacement():
  """The trainer replaces the probe defence on disk; the defence picks the
  new weights up at its next episode start, never mid-episode."""
  with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, 'defence.pt')
    _save(path, always=5)
    side = defence.Defence(path, ACTIONS, seed=0)
    side.start_episode()
    rows, frames = np.zeros((11, 115), np.float32), _frames()
    np.testing.assert_array_equal(side.act(rows, frames), np.full(11, 5))
    _save(path, always=9)
    np.testing.assert_array_equal(side.act(rows, frames), np.full(11, 5))
    side.start_episode()
    np.testing.assert_array_equal(side.act(rows, frames), np.full(11, 9))


def test_a_simple115_snapshot_reads_simple115_in_any_match():
  """A snapshot trained on simple115 gets egocentric simple115 rows even when
  the match's own rows are entity rows, exactly what a simple115 match would
  hand it; a snapshot reading neither width is refused."""
  with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, 'simple.pt')
    _save(path, features=115, seed=1)
    frames = _frames(seed=2)
    egocentric = frames.copy()
    normalize_egocentric(egocentric)
    sort_players_by_distance(egocentric)
    in_entity_match = defence.Defence(path, ACTIONS, seed=4)
    in_simple_match = defence.Defence(path, ACTIONS, seed=4)
    for side in (in_entity_match, in_simple_match):
      side.start_episode()
    for _ in range(6):
      np.testing.assert_array_equal(
          in_entity_match.act(np.ones((11, 300), np.float32), frames),
          in_simple_match.act(egocentric, frames))
    wide = os.path.join(directory, 'wide.pt')
    _save(wide, features=300)
    side = defence.Defence(wide, ACTIONS, seed=0)
    side.start_episode()
    try:
      side.act(np.zeros((11, 115), np.float32), frames)
    except ValueError:
      pass
    else:
      raise AssertionError('a snapshot reading 300 features must be refused')


def test_same_seed_same_rows_same_actions_and_reseed_replays():
  """Sampling is seeded: twins agree, and reseed restarts the stream (an
  evaluation pool replays its episodes on every reset(seed))."""
  with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, 'defence.pt')
    _save(path, seed=5)
    rows = [np.random.default_rng(i).normal(size=(11, 115)).astype(np.float32)
            for i in range(8)]
    sides = [defence.Defence(path, ACTIONS, seed=7, memory_window=3)
             for _ in range(2)]
    played = []
    for side in sides:
      side.start_episode()
      played.append([side.act(r, _frames()) for r in rows])
    np.testing.assert_array_equal(played[0], played[1])
    sides[0].reseed(7)
    sides[0].start_episode()
    np.testing.assert_array_equal(
        [sides[0].act(r, _frames()) for r in rows], played[0])


def test_memory_restarts_every_window_and_at_every_episode_start():
  """Training restarts recurrent memory every rollout segment, regardless of
  episode ends, and the defence replays a snapshot the same way: the clock
  runs on across episodes, and a new episode also starts from zero."""
  with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, 'defence.pt')
    _save(path)
    side = defence.Defence(path, ACTIONS, seed=0, memory_window=3)
    side.start_episode()
    # All-zero rows are absent players, whose memory stays zero.
    rows, frames = np.full((11, 115), 0.1, np.float32), _frames()
    fresh = []
    policy = side._load().policy
    forward = policy.forward_eval

    def recording_forward(observations, state):
      fresh.append(not bool(state['lstm_h'].abs().any()))
      return forward(observations, state)

    policy.forward_eval = recording_forward
    for _ in range(4):
      side.act(rows, frames)
    side.start_episode()  # an episode end on the clock's step 4
    for _ in range(4):
      side.act(rows, frames)
    assert fresh == [True, False, False, True,
                     True, False, True, False], fresh
    carried = defence.Defence(path, ACTIONS, seed=0, memory_window=None)
    carried.start_episode()
    fresh.clear()
    carried._load().policy.forward_eval = recording_forward
    for _ in range(5):
      carried.act(rows, frames)
    assert fresh == [True, False, False, False, False], fresh


if __name__ == '__main__':
  for test_name, test in sorted(globals().items()):
    if test_name.startswith('test_'):
      test()
      print('ok', test_name)
