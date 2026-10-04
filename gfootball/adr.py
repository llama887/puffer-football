"""Automatic domain randomization (ADR) for the advantage scenario.

OpenAI, "Solving Rubik's Cube with a Robot Hand" (2019), Section 5: every
randomized setting has its own range [low, high].  Training episodes draw
each setting uniformly within its current range, so what has already been
learned stays in the mix.  Probe episodes pin one randomly chosen setting at
one of its two edges and record whether the episode succeeded; once an edge
has `buffer_size` results, an average at or above `high_threshold` moves that
edge outward one step and one at or below `low_threshold` moves it inward.
Ranges therefore grow where the policy copes and shrink where it does not,
each setting on its own.

Here the ranges replace the single curriculum level.  They all start as the
single level-0 value and are clamped to each setting's valid limits.  The
ranges live in a shared-memory array (`shared_ranges`): the trainer's
randomizer records probe results and moves edges, and every env worker's
randomizer, built on the same array, samples from the current ranges with no
message passing.  The scenario (11_vs_11_advantage) turns a sampled set of
values into a scene.
"""

import collections
import multiprocessing

import numpy as np

LOWER, UPPER = 0, 1

ADRParameter = collections.namedtuple(
    'ADRParameter', ('name', 'minimum', 'maximum', 'start', 'step'))

# Valid limits, the level-0 starting value and the step per edge move.
PARAMETERS = (
    # Distance from the attacked goal line to the ball (level 0: 0.10).
    ADRParameter('ball_distance', 0.05, 1.0, 0.10, 0.045),
    # Ball y across the pitch.
    ADRParameter('ball_lateral', -0.36, 0.36, 0.0, 0.02),
    # Goal-side blockers; fractional values add one more by chance.
    ADRParameter('blockers', 0.0, 5.0, 1.0, 0.2),
    # Distance from the still ball to the nearest attacker (0.03: the
    # carrier at the ball, as at level 0).
    ADRParameter('fetch_distance', 0.03, 0.9, 0.03, 0.05),
    # Nearest defender's distance minus the nearest attacker's.
    ADRParameter('defender_head_start', -0.5, 0.5, 0.05, 0.05),
    # Attackers placed between the ball and the goal, onside and clear of
    # the shooting lane; fractional values add one more by chance.
    ADRParameter('attackers_ahead', 0.0, 3.0, 0.0, 0.25),
    # 0: both teams packed around the ball; 1: real match formations.
    ADRParameter('formation', 0.0, 1.0, 0.0, 0.05),
    # Share of outfield players that start holding a random direction.
    ADRParameter('held_share', 0.0, 1.0, 0.0, 0.1),
)
PARAMETER_INDEX = {parameter.name: i for i, parameter in enumerate(PARAMETERS)}
# Widths below this count as this for the entropy, so a point is finite.
_ENTROPY_FLOOR = 1e-3


def shared_ranges():
  """A shared-memory array holding every range, initialised to the starts."""
  array = multiprocessing.RawArray('d', 2 * len(PARAMETERS))
  view = np.frombuffer(array, dtype=np.float64).reshape(len(PARAMETERS), 2)
  for i, parameter in enumerate(PARAMETERS):
    view[i] = parameter.start
  return array


class AutomaticDomainRandomization:
  """The ADR ranges, how episodes sample them, and how probes move them.

  Build one in the trainer (it records probe results) and one per env worker
  on the same `shared` array (they only sample).  Without `shared` it keeps
  its own private ranges, which is what tests use.
  """

  def __init__(self, shared=None, buffer_size=64, low_threshold=0.3,
               high_threshold=0.6):
    if not 0 <= low_threshold < high_threshold <= 1:
      raise ValueError('need 0 <= low_threshold < high_threshold <= 1')
    if buffer_size < 1:
      raise ValueError('buffer_size must be positive')
    self._shared = shared if shared is not None else shared_ranges()
    self._ranges = np.frombuffer(self._shared, dtype=np.float64).reshape(
        len(PARAMETERS), 2)
    self._buffer_size = int(buffer_size)
    self._low_threshold = float(low_threshold)
    self._high_threshold = float(high_threshold)
    self._buffers = [[[], []] for _ in PARAMETERS]

  @property
  def shared(self):
    return self._shared

  def ranges(self):
    """A copy of the current [low, high] per setting, in PARAMETERS order."""
    return self._ranges.copy()

  def set_range(self, name, low, high):
    """Set one range directly (tests and restarts), clamped to its limits."""
    parameter = PARAMETERS[PARAMETER_INDEX[name]]
    low = max(parameter.minimum, float(low))
    high = min(parameter.maximum, float(high))
    if low > high:
      raise ValueError('low must not exceed high')
    self._ranges[PARAMETER_INDEX[name]] = (low, high)

  def sample(self, rng):
    """Values for a training episode: each setting uniform in its range."""
    ranges = self.ranges()
    return {parameter.name: float(rng.uniform(low, high)) if high > low
            else float(low)
            for parameter, (low, high) in zip(PARAMETERS, ranges)}

  def sample_probe(self, rng):
    """Values for a probe episode, and which setting and edge it tests.

    One setting, chosen uniformly, is pinned to its lower or upper edge
    (also chosen uniformly); the rest are sampled as for training.
    """
    ranges = self.ranges()
    values = {parameter.name: float(rng.uniform(low, high)) if high > low
              else float(low)
              for parameter, (low, high) in zip(PARAMETERS, ranges)}
    index = int(rng.integers(len(PARAMETERS)))
    edge = int(rng.integers(2))
    values[PARAMETERS[index].name] = float(ranges[index][edge])
    return values, index, edge

  def record(self, index, edge, success):
    """Add one probe result at a setting's edge; move the edge when due.

    Returns True when this result triggered a move.
    """
    buffer = self._buffers[index][edge]
    buffer.append(float(success))
    if len(buffer) < self._buffer_size:
      return False
    average = sum(buffer) / len(buffer)
    buffer.clear()
    if average >= self._high_threshold:
      outward = True
    elif average <= self._low_threshold:
      outward = False
    else:
      return False
    parameter = PARAMETERS[index]
    low, high = self._ranges[index]
    if edge == UPPER:
      high = (min(parameter.maximum, high + parameter.step) if outward
              else max(low, high - parameter.step))
    else:
      low = (max(parameter.minimum, low - parameter.step) if outward
             else min(high, low + parameter.step))
    moved = (low, high) != tuple(self._ranges[index])
    self._ranges[index] = (low, high)
    return moved

  def entropy(self):
    """ADR entropy: mean log range width (nats per setting)."""
    widths = np.maximum(self._ranges[:, 1] - self._ranges[:, 0],
                        _ENTROPY_FLOOR)
    return float(np.mean(np.log(widths)))
