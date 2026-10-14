"""Automatic domain randomization (ADR): what every training scene is.

OpenAI, "Solving Rubik's Cube with a Robot Hand" (2019), Section 5: every
randomized setting has its own range [low, high].  Training episodes draw
each setting uniformly within its current range, so what has already been
learned stays in the mix.  Probe episodes pin one randomly chosen setting at
one of its two edges and record whether the episode succeeded; once an edge
has `buffer_size` results, an average at or above `high_threshold` moves that
edge outward one step and one at or below `low_threshold` moves it inward.
Ranges therefore grow where the policy copes and shrink where it does not,
each setting on its own.

The ranges all start at the start scene (a carrier at the ball 0.1 from the
goal line, one blocker): single points, except ball_lateral, which starts
wider than the goal posts so the carrier has to angle in from the first
episode.  An edge moves outward no further than the setting's valid limits
and inward no further than its start range.  The ranges live in a
shared-memory array (`shared_ranges`): the trainer's randomizer records
probe results and moves edges, and every env worker's randomizer, built on
the same array, samples from the current ranges with no message passing.  The scenario (SCENARIO) turns a sampled set
of values into a scene.  Evaluation scenes use fixed ranges instead
(`set_difficulty`), the same for every run.
"""

import collections
import multiprocessing

import numpy as np

LOWER, UPPER = 0, 1
# The gfootball scenario that builds a scene from a set of ADR values.  It
# reads them from the config value VALUES and writes what it built for the
# episode into HELD_DIRECTIONS and LOOSE_BALL (see read_scene).
SCENARIO = '11_vs_11_adr'
VALUES = 'adr_values'
HELD_DIRECTIONS = 'held_directions'
LOOSE_BALL = 'loose_ball'

# What the scenario built for an episode (read_scene).
Scene = collections.namedtuple(
    'Scene', ('attacking_left', 'held_directions', 'loose_ball'))


class ADRParameter(collections.namedtuple(
    'ADRParameter',
    ('name', 'minimum', 'maximum', 'start', 'step', 'start_spread'),
    defaults=(0.0,))):
  """One randomized setting: valid limits, start value, step per edge move.

  `start_spread` widens the start from a point to start +- start_spread.
  """

  @property
  def start_range(self):
    """(low, high) every run starts from; edges never shrink inside it."""
    return (self.start - self.start_spread, self.start + self.start_spread)


# Valid limits, the starting value and the step per edge move.
PARAMETERS = (
    # Distance from the attacked goal line to the ball.
    ADRParameter('ball_distance', 0.05, 1.0, 0.10, 0.045),
    # Ball y across the pitch.  Starts at +-0.08, beyond the posts (|y|
    # 0.044): from a point at 0, policies learned only a straight shot and
    # the edge never got past the posts (2026-10-14 lateral analysis).
    ADRParameter('ball_lateral', -0.36, 0.36, 0.0, 0.02, start_spread=0.08),
    # Goal-side blockers; fractional values add one more by chance.
    ADRParameter('blockers', 0.0, 5.0, 1.0, 0.2),
    # Distance from the still ball to the nearest attacker (0.03: the
    # carrier at the ball).
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


def _uniform_values(ranges, rng):
  """Each setting uniform in its row of `ranges` (a point gives itself)."""
  return {parameter.name: float(rng.uniform(low, high)) if high > low
          else float(low)
          for parameter, (low, high) in zip(PARAMETERS, ranges)}


def shared_ranges():
  """A shared-memory array holding every range, initialised to the start
  ranges."""
  array = multiprocessing.RawArray('d', 2 * len(PARAMETERS))
  view = np.frombuffer(array, dtype=np.float64).reshape(len(PARAMETERS), 2)
  for i, parameter in enumerate(PARAMETERS):
    view[i] = parameter.start_range
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

  def set_difficulty(self, difficulty):
    """Fix every range at a test difficulty, for evaluation scenes.

    Difficulty d spans each setting from its start range out to d of the way
    to each valid limit: 0 is the start scene, 1 every limit.  Unlike the
    ranges training reaches, these depend on d alone, so a score at a given
    d compares across runs and over time.
    """
    if not 0 <= difficulty <= 1:
      raise ValueError('difficulty must be in [0, 1]')
    for i, parameter in enumerate(PARAMETERS):
      start_low, start_high = parameter.start_range
      self._ranges[i] = (
          start_low - difficulty * (start_low - parameter.minimum),
          start_high + difficulty * (parameter.maximum - start_high))

  def sample(self, rng):
    """Values for a training episode: each setting uniform in its range."""
    return _uniform_values(self.ranges(), rng)

  def sample_probe(self, rng):
    """Values for a probe episode, and which setting and edge it tests.

    One setting, chosen uniformly, is pinned to its lower or upper edge
    (also chosen uniformly); the rest are sampled as for training.
    """
    index = int(rng.integers(len(PARAMETERS)))
    edge = int(rng.integers(2))
    return self.sample_pinned(rng, index, edge), index, edge

  def sample_pinned(self, rng, index, edge):
    """Values with setting `index` at `edge` (LOWER or UPPER) of its range
    and the rest sampled as for training: a probe's scene, with the
    setting and edge picked at random by sample_probe or by hand
    (scripts/attack_vs_defence.py --scene adr:pin:...).

    Every value comes from one copy of the ranges: the trainer may move an
    edge while a worker samples, and the probe must test the edge it pins.
    """
    ranges = self.ranges()
    values = _uniform_values(ranges, rng)
    values[PARAMETERS[index].name] = float(ranges[index][edge])
    return values

  def record(self, index, edge, success):
    """Add one probe result at a setting's edge; move the edge when due.

    An edge moves outward no further than the setting's limit and inward no
    further than its start range, so every run keeps practising the start
    scene.  Returns True when this result triggered a move.
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
    start_low, start_high = parameter.start_range
    low, high = self._ranges[index]
    # Moving in stops at the start range; an edge already inside it (set by
    # hand or loaded from an older run) stays put rather than jump out.
    if edge == UPPER:
      high = (min(parameter.maximum, high + parameter.step) if outward
              else max(low, min(high, start_high), high - parameter.step))
    else:
      low = (max(parameter.minimum, low - parameter.step) if outward
             else min(high, max(low, start_low), low + parameter.step))
    moved = (low, high) != tuple(self._ranges[index])
    self._ranges[index] = (low, high)
    return moved

  def entropy(self):
    """ADR entropy: mean log range width (nats per setting)."""
    widths = np.maximum(self._ranges[:, 1] - self._ranges[:, 0],
                        _ENTROPY_FLOOR)
    return float(np.mean(np.log(widths)))

  def metrics(self):
    """Both edges of every range and the entropy, as the trainer logs them
    (`adr/<setting>/low`, `adr/<setting>/high`, `adr/entropy`)."""
    metrics = {'adr/entropy': self.entropy()}
    for parameter, (low, high) in zip(PARAMETERS, self._ranges):
      metrics['adr/{}/low'.format(parameter.name)] = float(low)
      metrics['adr/{}/high'.format(parameter.name)] = float(high)
    return metrics

  def load_metrics(self, metrics):
    """Set every range from logged metrics (a training log's ADR line)."""
    for parameter in PARAMETERS:
      self.set_range(parameter.name,
                     metrics['adr/{}/low'.format(parameter.name)],
                     metrics['adr/{}/high'.format(parameter.name)])


def read_scene(config):
  """What the scenario built for the episode just reset into `config`: which
  side attacks, the direction each agent holds on the first step (0: none)
  and whether the opening is a race for a still ball."""
  values = config._values
  return Scene(
      attacking_left=config.ScenarioConfig().ball_position[0] > 0,
      held_directions=np.asarray(values.get(HELD_DIRECTIONS, [0] * 22),
                                 dtype=np.int64),
      loose_ball=bool(values.get(LOOSE_BALL, False)))
