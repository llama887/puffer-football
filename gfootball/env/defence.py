"""The defending side of a match when the learner does not play it.

A defence is named by a spec: RANDOM (uniform random actions, the base
random policy), IDLE (no action at all) or the path of a policy snapshot.
BOT, the engine's built-in AI, is a spec too, but no Defence plays it: its
team must have no agents at all, so the env plays bot episodes in an engine
of their own (puffer_env.FootballPufferEnv).
It turns the defending side's rows of the latest observation into 11
actions inside the env worker; the env hides those rows from the learner, so
they are never trained.

A snapshot is loaded on first use and reloaded at the next episode start
whenever the file on disk has been replaced (writers replace it atomically,
see save_policy_snapshot, so new weights always come with a new inode or
mtime): that is how the trainer refreshes the ADR probe defence.  It reads
the observation it was trained on: the match's own rows when their width
matches, otherwise simple115 from the newest frame (a fixed simple115
opponent in an entity-observation or frame-stacked match).  It plays
through a PolicyPlayer (gfootball/env/puffer_policy.py): memory restarts
every `memory_window` of its own steps (None: whole episodes) and at every
episode start, and sampling is seeded, so the same seed and the same
observations give the same actions.

A Defence can play either side: attack_vs_defence plays its attack with
one too.
"""

import os
from types import SimpleNamespace

import numpy as np

RANDOM = 'random'
IDLE = 'idle'
BOT = 'bot'
_SIMPLE115 = 115
# The PolicyPlayer noise stream of a snapshot side: an evaluation pool's
# match 0 has the evaluation seed, so its defence and the evaluated policy
# would otherwise draw the same noise.
_NOISE_STREAM = 0xdef


class Defence:
  """One side of one match played by a fixed policy (or random or idle
  actions): `act` maps that side's 11 rows to 11 actions."""

  def __init__(self, spec, action_count, seed, memory_window=None,
               sort_players=True):
    if memory_window is not None and memory_window < 1:
      raise ValueError('memory_window must be positive or None')
    if spec == BOT:
      raise ValueError('the bot is played by the env, not by a Defence')
    self._spec = spec
    self._action_count = int(action_count)
    self._memory_window = memory_window
    self._sort_players = bool(sort_players)
    # The snapshot's player, and the (path, inode, mtime) it was loaded from.
    self._player = None
    self._source = None
    self.reseed(seed)

  def reseed(self, seed):
    """Restart every random stream and the memory clock from `seed`."""
    self._seed = int(seed)
    self._rng = np.random.default_rng([self._seed, 0xdef])
    if self._player is not None:
      self._player.reset(self._seed)

  def start_episode(self):
    """Clear the memory; pick up a snapshot replaced on disk since loading.

    Checked once per episode, not per step, to spare the file system.
    """
    if self._spec in (RANDOM, IDLE):
      return
    stat = os.stat(self._spec)
    source = (self._spec, stat.st_ino, stat.st_mtime_ns)
    if source != self._source:
      self._player = None
      self._source = source
    elif self._player is not None:
      self._player.clear(0)

  def act(self, rows, frames):
    """Actions for the side's 11 agents.

    `rows` are their rows of the match's own observation (normalized, before
    hiding); `frames` their raw simple115v2 frames, newest frame last.
    """
    if self._spec == IDLE:
      return np.zeros(11, dtype=np.int64)
    if self._spec == RANDOM:
      return self._rng.integers(0, self._action_count, size=11)
    from gfootball.env.puffer_env import (
        normalize_egocentric, sort_players_by_distance)
    player = self._load()
    reads = player.policy.normalizer.mean.numel()
    if reads == rows.shape[1]:
      observations = rows
    elif reads == _SIMPLE115:
      observations = np.array(frames[:, -_SIMPLE115:], dtype=np.float32)
      normalize_egocentric(observations)
      if self._sort_players:
        sort_players_by_distance(observations)
    else:
      raise ValueError('defence snapshot reads {} features; this match gives '
                       '{}'.format(reads, rows.shape[1]))
    actions, _ = player.act(observations, [0])
    return actions.numpy()

  def _load(self):
    """The snapshot's player, single threaded on the CPU, built on first use
    and again after a reload (its clock and stream restart from the seed)."""
    if self._player is None:
      import torch
      from gfootball.env.puffer_policy import PolicyPlayer, load_frozen_policy
      torch.set_num_threads(1)
      policy = load_frozen_policy(self._spec, SimpleNamespace(
          single_action_space=SimpleNamespace(n=self._action_count)))
      self._player = PolicyPlayer(policy, 1, 11, self._memory_window,
                                  self._seed, stream=_NOISE_STREAM)
    return self._player
