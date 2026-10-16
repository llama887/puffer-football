"""Tests for automatic domain randomization (gfootball/adr.py).

Pure numpy; run with `python -m gfootball.adr_test` (or pytest).
"""

import multiprocessing

import numpy as np

from gfootball import adr

# The goal posts' |y| in the pitch frame.
POST_Y = 0.044


def _fresh(**kwargs):
  return adr.AutomaticDomainRandomization(**kwargs)


def test_ranges_start_at_the_start_scene():
  """Every range starts as its start range, inside its limits: a single
  point for every setting except ball_lateral."""
  randomizer = _fresh()
  ranges = randomizer.ranges()
  assert ranges.shape == (len(adr.PARAMETERS), 2)
  for (low, high), parameter in zip(ranges, adr.PARAMETERS):
    assert (low, high) == parameter.start_range
    assert parameter.minimum <= low <= parameter.start <= high
    assert high <= parameter.maximum
    if parameter.name != 'ball_lateral':
      assert low == high == parameter.start
  values = randomizer.sample(np.random.default_rng(0))
  assert {name: value for name, value in values.items()
          if name != 'ball_lateral'} == {
              p.name: p.start for p in adr.PARAMETERS
              if p.name != 'ball_lateral'}


def test_ball_lateral_starts_wider_than_the_posts():
  """From the first episode the ball starts beside the goal mouth as well as
  in it, so a straight shot is not enough and the carrier has to angle in a
  little.  Started at a point, ADR policies learned only to go straight and
  the edge never got past the posts (2026-10-14 lateral analysis)."""
  low, high = _fresh().ranges()[adr.PARAMETER_INDEX['ball_lateral']]
  assert low < -POST_Y and high > POST_Y
  assert high - POST_Y >= 0.02 and -POST_Y - low >= 0.02
  assert high <= 0.12 and low >= -0.12  # wide, but not far out on the wing
  rng = np.random.default_rng(3)
  draws = np.array([_fresh().sample(rng)['ball_lateral']
                    for _ in range(2000)])
  outside = np.mean(np.abs(draws) > POST_Y)
  assert 0.25 < outside < 0.75, outside


def test_training_samples_are_uniform_over_every_current_range():
  """Non-probe episodes draw each setting uniformly within its range, so the
  easy settings already learned stay in the training mix."""
  randomizer = _fresh()
  index = adr.PARAMETER_INDEX['ball_distance']
  randomizer.set_range('ball_distance', 0.1, 0.5)
  rng = np.random.default_rng(1)
  draws = np.array([randomizer.sample(rng)['ball_distance']
                    for _ in range(4000)])
  assert draws.min() >= 0.1 and draws.max() <= 0.5
  assert abs(np.mean(draws < 0.3) - 0.5) < 0.05
  assert draws.min() < 0.12  # past (easy) tasks still drawn
  assert randomizer.ranges()[index].tolist() == [0.1, 0.5]


def test_probe_pins_one_setting_to_one_edge():
  """A probe fixes one randomly chosen setting at its lower or upper edge
  and samples the rest as usual; settings and edges are chosen evenly."""
  randomizer = _fresh()
  for name in adr.PARAMETER_INDEX:
    randomizer.set_range(name, *_middle_half(name))
  rng = np.random.default_rng(2)
  chosen = np.zeros((len(adr.PARAMETERS), 2))
  for _ in range(3200):
    values, index, edge = randomizer.sample_probe(rng)
    name = adr.PARAMETERS[index].name
    assert values[name] == randomizer.ranges()[index][edge]
    for other, value in values.items():
      low, high = randomizer.ranges()[adr.PARAMETER_INDEX[other]]
      assert low <= value <= high
    chosen[index, edge] += 1
  expected = 3200 / chosen.size
  assert np.all(np.abs(chosen - expected) < 0.35 * expected)


def test_edges_grow_on_success_and_shrink_on_failure():
  """After `buffer_size` probe results at an edge: an average at or above the
  high threshold moves that edge outward one step, at or below the low
  threshold moves it inward, in between leaves it; the buffer then clears."""
  randomizer = _fresh(buffer_size=4, low_threshold=0.3, high_threshold=0.6)
  name = 'ball_distance'
  index = adr.PARAMETER_INDEX[name]
  step = adr.PARAMETERS[index].step
  randomizer.set_range(name, 0.3, 0.5)
  for success in (1, 1, 1):
    assert not randomizer.record(index, adr.UPPER, success)
  assert randomizer.ranges()[index].tolist() == [0.3, 0.5]
  assert randomizer.record(index, adr.UPPER, 0)  # 3/4 = 0.75 >= 0.6
  np.testing.assert_allclose(randomizer.ranges()[index], [0.3, 0.5 + step])
  for success in (1, 1, 1, 1):
    randomizer.record(index, adr.LOWER, success)
  np.testing.assert_allclose(randomizer.ranges()[index],
                             [0.3 - step, 0.5 + step])
  for success in (0, 0, 0, 1):  # 0.25 <= 0.3: the upper edge comes back in
    randomizer.record(index, adr.UPPER, success)
  np.testing.assert_allclose(randomizer.ranges()[index], [0.3 - step, 0.5])
  for success in (1, 0, 1, 0):  # 0.5: no change
    randomizer.record(index, adr.UPPER, success)
  np.testing.assert_allclose(randomizer.ranges()[index], [0.3 - step, 0.5])


def test_edges_respect_the_limits_and_never_cross():
  """Growing stops at each setting's valid limits, and shrinking never moves
  an edge past the other one, so a point range stays a point."""
  randomizer = _fresh(buffer_size=1)
  for index, parameter in enumerate(adr.PARAMETERS):
    for _ in range(400):
      randomizer.record(index, adr.UPPER, 1)
      randomizer.record(index, adr.LOWER, 1)
    np.testing.assert_allclose(randomizer.ranges()[index],
                               [parameter.minimum, parameter.maximum])
  start = _fresh(buffer_size=1)
  for index, parameter in enumerate(adr.PARAMETERS):
    start.record(index, adr.UPPER, 0)
    start.record(index, adr.LOWER, 0)
    assert tuple(start.ranges()[index]) == parameter.start_range


def test_edges_never_shrink_inside_the_start_range():
  """Failing edges come back in, but no further than the start range: every
  run keeps practising the start scene, and ball_lateral keeps the balls
  beside the posts even while the policy cannot score from there yet."""
  randomizer = _fresh(buffer_size=1)
  index = adr.PARAMETER_INDEX['ball_lateral']
  parameter = adr.PARAMETERS[index]
  for _ in range(3):
    randomizer.record(index, adr.UPPER, 1)
    randomizer.record(index, adr.LOWER, 1)
  low, high = randomizer.ranges()[index]
  np.testing.assert_allclose(
      [low, high], [parameter.start_range[0] - 3 * parameter.step,
                    parameter.start_range[1] + 3 * parameter.step])
  for _ in range(10):
    randomizer.record(index, adr.UPPER, 0)
    randomizer.record(index, adr.LOWER, 0)
  np.testing.assert_allclose(randomizer.ranges()[index],
                             parameter.start_range)
  # A range set (or loaded from an older run) clear of the start range
  # never jumps outward to it on a failure; it just stops moving in.
  randomizer.set_range('ball_distance', 0.3, 0.5)
  distance = adr.PARAMETER_INDEX['ball_distance']
  for _ in range(3):
    randomizer.record(distance, adr.LOWER, 0)
  np.testing.assert_allclose(randomizer.ranges()[distance], [0.3, 0.5])


def test_workers_share_the_ranges_through_shared_memory():
  """The trainer's randomizer writes ranges that a worker's randomizer built
  on the same shared array reads, without any message passing."""
  shared = adr.shared_ranges()
  # A ctypes array of doubles in shared memory, one slot per range edge.
  assert shared._type_ is type(multiprocessing.RawArray('d', 1))._type_
  assert len(shared) == 2 * len(adr.PARAMETERS)
  trainer = _fresh(shared=shared, buffer_size=1)
  worker = _fresh(shared=shared)
  index = adr.PARAMETER_INDEX['blockers']
  trainer.record(index, adr.UPPER, 1)
  np.testing.assert_allclose(worker.ranges()[index],
                             trainer.ranges()[index])
  assert worker.ranges()[index][1] > adr.PARAMETERS[index].start


def test_entropy_rises_as_ranges_widen():
  """ADR entropy (mean log width, widths floored so points are finite)
  tracks progress."""
  randomizer = _fresh(buffer_size=1)
  before = randomizer.entropy()
  assert np.isfinite(before)
  randomizer.record(adr.PARAMETER_INDEX['ball_lateral'], adr.UPPER, 1)
  assert randomizer.entropy() > before


def test_difficulty_spans_a_fixed_share_of_the_way_to_the_limits():
  """A test difficulty d fixes every range from its start range out to d of
  the way to each valid limit: 0 is the start scene, 1 every limit.  The
  same d always gives the same ranges, so scores compare across runs."""
  randomizer = _fresh()
  randomizer.set_difficulty(0.0)
  for (low, high), parameter in zip(randomizer.ranges(), adr.PARAMETERS):
    assert (low, high) == parameter.start_range
  randomizer.set_difficulty(1.0)
  for (low, high), parameter in zip(randomizer.ranges(), adr.PARAMETERS):
    assert (low, high) == (parameter.minimum, parameter.maximum)
  randomizer.set_difficulty(0.25)
  for (low, high), parameter in zip(randomizer.ranges(), adr.PARAMETERS):
    np.testing.assert_allclose(
        [low, high],
        [parameter.start_range[0] -
         0.25 * (parameter.start_range[0] - parameter.minimum),
         parameter.start_range[1] +
         0.25 * (parameter.maximum - parameter.start_range[1])])
  for bad in (-0.1, 1.5):
    try:
      randomizer.set_difficulty(bad)
    except ValueError:
      continue
    raise AssertionError('difficulty {} accepted'.format(bad))


def test_logged_metrics_restore_the_ranges():
  """The ADR log line (metrics) is the one format for ranges: training
  writes it, scoring scripts load a run's ranges back from it."""
  trainer = _fresh()
  trainer.set_range('blockers', 0.4, 3.2)
  trainer.set_range('ball_lateral', -0.06, 0.04)
  logged = trainer.metrics()
  assert logged['adr/blockers/low'] == 0.4
  assert logged['adr/entropy'] == trainer.entropy()
  reader = _fresh()
  reader.load_metrics(logged)
  np.testing.assert_array_equal(reader.ranges(), trainer.ranges())


def test_a_pinned_probe_reads_one_snapshot_of_the_ranges():
  """The trainer moves edges while env workers sample: a probe's pinned
  edge and its other values must come from the same ranges."""
  randomizer = _fresh()
  randomizer.set_range('blockers', 1.0, 2.0)
  live = adr.AutomaticDomainRandomization(shared=randomizer.shared)

  class MovingRng:
    """Draws like numpy, but the trainer moves an edge mid-sample."""

    def __init__(self):
      self._rng = np.random.default_rng(0)

    def uniform(self, low, high):
      randomizer.set_range('blockers', 1.0, 4.0)
      return self._rng.uniform(low, high)

    def integers(self, high):
      return self._rng.integers(high)

  values = live.sample_pinned(MovingRng(), adr.PARAMETER_INDEX['blockers'],
                              adr.UPPER)
  assert values['blockers'] == 2.0, values['blockers']


def test_clearing_probe_results_starts_every_edge_afresh():
  """Results recorded against an old probe defence are dropped, so the next
  edge move is decided only by results against the new one."""
  randomizer = _fresh(buffer_size=4)
  index = adr.PARAMETER_INDEX['blockers']
  for _ in range(3):
    randomizer.record(index, adr.UPPER, 1)
  randomizer.clear_probe_results()
  assert not randomizer.record(index, adr.UPPER, 1)
  assert randomizer.ranges()[index][1] == adr.PARAMETERS[index].start


def test_bot_ladder_starts_easy_and_shares_its_difficulty():
  """The ladder starts at the easy bot (0.05) and writes the current
  difficulty to shared memory, which every env worker reads."""
  shared = adr.shared_bot_difficulty()
  ladder = adr.BotLadder(shared=shared)
  assert ladder.difficulty == adr.BOT_LADDER[0] == 0.05
  assert shared.value == 0.05
  assert list(adr.BOT_LADDER) == sorted(adr.BOT_LADDER)
  assert adr.BOT_LADDER[-1] <= 1.0


def test_bot_ladder_steps_up_after_consecutive_toothless_checks():
  """Bite = 1 - goals vs bot / goals vs idle defenders on the same scenes.
  Two checks in a row with bite below 0.2 mean the bot barely costs the
  attack anything, so the ladder moves to the next bot and restarts its
  count; a biting check in between restarts the count."""
  shared = adr.shared_bot_difficulty()
  ladder = adr.BotLadder(shared=shared)
  metrics = ladder.check(vs_bot=0.45, vs_idle=0.5)  # bite 0.1
  assert np.isclose(metrics['bot/bite'], 0.1)
  assert metrics['bot/raised'] == 0.0 and metrics['bot/toothless_checks'] == 1
  ladder.check(vs_bot=0.2, vs_idle=0.5)  # bite 0.6: count restarts
  ladder.check(vs_bot=0.45, vs_idle=0.5)
  assert ladder.difficulty == 0.05
  metrics = ladder.check(vs_bot=0.48, vs_idle=0.5)
  assert metrics['bot/raised'] == 1.0
  assert ladder.difficulty == adr.BOT_LADDER[1] == shared.value
  assert metrics['bot/difficulty'] == adr.BOT_LADDER[1]
  assert metrics['bot/toothless_checks'] == 0
  assert ladder.check(vs_bot=0.45, vs_idle=0.5)['bot/raised'] == 0.0


def test_bot_ladder_skips_checks_too_noisy_to_judge():
  """With few goals even against idle defenders the bite ratio is noise:
  such a check neither counts toward stepping up nor restarts the count."""
  ladder = adr.BotLadder(shared=adr.shared_bot_difficulty())
  ladder.check(vs_bot=0.45, vs_idle=0.5)
  metrics = ladder.check(vs_bot=0.05, vs_idle=0.1)
  assert np.isnan(metrics['bot/bite'])
  assert metrics['bot/toothless_checks'] == 1
  assert ladder.check(vs_bot=0.45, vs_idle=0.5)['bot/raised'] == 1.0
  # No goals at all against idle defenders: nothing to divide by.
  lenient = adr.BotLadder(shared=adr.shared_bot_difficulty(),
                          minimum_idle_rate=0.0)
  assert np.isnan(lenient.check(vs_bot=0.0, vs_idle=0.0)['bot/bite'])


def test_bot_ladder_stops_at_the_hardest_bot():
  ladder = adr.BotLadder(shared=adr.shared_bot_difficulty())
  for _ in range(4 * len(adr.BOT_LADDER)):
    ladder.check(vs_bot=0.5, vs_idle=0.5)
  assert ladder.difficulty == adr.BOT_LADDER[-1]


def _middle_half(name):
  parameter = adr.PARAMETERS[adr.PARAMETER_INDEX[name]]
  span = parameter.maximum - parameter.minimum
  return (parameter.minimum + 0.25 * span, parameter.minimum + 0.75 * span)


if __name__ == '__main__':
  for test_name, test in sorted(globals().items()):
    if test_name.startswith('test_'):
      test()
      print('ok', test_name)
