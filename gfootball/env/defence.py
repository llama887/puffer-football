"""The defending side of a match when the learner does not play it.

A defence is named by a spec: RANDOM (uniform random actions, the base
random policy), IDLE (no action at all) or the path of a policy snapshot.
It turns the defending side's rows of the latest observation into 11
actions inside the env worker; the env hides those rows from the learner, so
they are never trained.

A snapshot is loaded on first use and reloaded at the next episode start
whenever the file on disk has been replaced (writers replace it atomically,
see save_policy_snapshot, so new weights always come with a new inode or
mtime): that is how the trainer refreshes the ADR probe defence.  It reads
the observation it was trained on: the match's own rows when their width
matches, otherwise simple115 (a fixed simple115 opponent in an
entity-observation match).  Its recurrent memory restarts every
`memory_window` of its own steps, the way training restarts memory at every
rollout segment regardless of episode ends (None keeps it for whole
episodes), and at every episode start.  Sampling is seeded, so the same seed
and the same observations give the same actions.
"""

import os

import numpy as np

RANDOM = 'random'
IDLE = 'idle'
_SIMPLE115 = 115


class Defence:
  """One match's defending side: `act` maps its rows to 11 actions."""

  def __init__(self, spec, action_count, seed, memory_window=None,
               sort_players=True):
    if memory_window is not None and memory_window < 1:
      raise ValueError('memory_window must be positive or None')
    self.spec = spec
    self._action_count = int(action_count)
    self._memory_window = memory_window
    self._sort_players = bool(sort_players)
    self._policy = None
    # (path, inode, mtime) of the file _policy was loaded from.
    self._source = None
    self.reseed(seed)

  def reseed(self, seed):
    """Restart every random stream and the memory clock from `seed`."""
    self._seed = int(seed)
    self._rng = np.random.default_rng([self._seed, 0xdef])
    self._generator = None
    self._steps = 0
    self._state = {'lstm_h': None, 'lstm_c': None, 'done': None}

  def start_episode(self):
    """Clear the memory; pick up a snapshot replaced on disk since loading.

    Checked once per episode, not per step, to spare the file system.
    """
    self._state = {'lstm_h': None, 'lstm_c': None, 'done': None}
    if self.spec in (RANDOM, IDLE):
      return
    stat = os.stat(self.spec)
    source = (self.spec, stat.st_ino, stat.st_mtime_ns)
    if source != self._source:
      self._policy = None
      self._source = source

  def act(self, rows, frames):
    """Actions for the 11 defending agents.

    `rows` are their rows of the match's own observation (normalized, before
    hiding); `frames` their raw simple115v2 frames, newest frame last.
    """
    if self.spec == IDLE:
      return np.zeros(11, dtype=np.int64)
    if self.spec == RANDOM:
      return self._rng.integers(0, self._action_count, size=11)
    import torch
    from gfootball.env.puffer_env import (
        normalize_egocentric, sort_players_by_distance)
    policy = self._load()
    reads = policy.normalizer.mean.numel()
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
    if (self._memory_window is not None and
        self._steps % self._memory_window == 0):
      self._state['lstm_h'] = self._state['lstm_c'] = None
    self._steps += 1
    if self._generator is None:
      self._generator = torch.Generator().manual_seed(self._seed)
    with torch.no_grad():
      logits, _ = policy.forward_eval(
          torch.as_tensor(observations), self._state)
      actions = torch.multinomial(
          torch.softmax(logits.float(), dim=-1), 1,
          generator=self._generator).squeeze(-1)
    return actions.numpy()

  def _load(self):
    """The snapshot, single threaded on the CPU, loaded on first use."""
    if self._policy is None:
      import torch
      from types import SimpleNamespace
      from gfootball.env.puffer_policy import load_frozen_policy
      torch.set_num_threads(1)
      self._policy = load_frozen_policy(self.spec, SimpleNamespace(
          single_action_space=SimpleNamespace(n=self._action_count)))
    return self._policy
