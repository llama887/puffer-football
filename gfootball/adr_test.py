"""Tests for automatic domain randomization (gfootball/adr.py).

Pure numpy; run with `python -m gfootball.adr_test` (or pytest).
"""

import multiprocessing

import numpy as np

from gfootball import adr


def _fresh(**kwargs):
  return adr.AutomaticDomainRandomization(**kwargs)


def test_ranges_start_as_points_at_the_start_scene():
  """Every range starts as the single start value, inside its limits."""
  randomizer = _fresh()
  ranges = randomizer.ranges()
  assert ranges.shape == (len(adr.PARAMETERS), 2)
  for (low, high), parameter in zip(ranges, adr.PARAMETERS):
    assert low == high == parameter.start
    assert parameter.minimum <= parameter.start <= parameter.maximum
  values = randomizer.sample(np.random.default_rng(0))
  assert values == {p.name: p.start for p in adr.PARAMETERS}


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
  point = _fresh(buffer_size=1)
  for index, parameter in enumerate(adr.PARAMETERS):
    point.record(index, adr.UPPER, 0)
    point.record(index, adr.LOWER, 0)
    low, high = point.ranges()[index]
    assert low == high == parameter.start


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
  """A test difficulty d fixes every range from its start value out to d of
  the way to each valid limit: 0 is the start scene, 1 every limit.  The
  same d always gives the same ranges, so scores compare across runs."""
  randomizer = _fresh()
  randomizer.set_difficulty(0.0)
  for (low, high), parameter in zip(randomizer.ranges(), adr.PARAMETERS):
    assert low == high == parameter.start
  randomizer.set_difficulty(1.0)
  for (low, high), parameter in zip(randomizer.ranges(), adr.PARAMETERS):
    assert (low, high) == (parameter.minimum, parameter.maximum)
  randomizer.set_difficulty(0.25)
  for (low, high), parameter in zip(randomizer.ranges(), adr.PARAMETERS):
    np.testing.assert_allclose(
        [low, high],
        [parameter.start - 0.25 * (parameter.start - parameter.minimum),
         parameter.start + 0.25 * (parameter.maximum - parameter.start)])
  for bad in (-0.1, 1.5):
    try:
      randomizer.set_difficulty(bad)
    except ValueError:
      continue
    raise AssertionError('difficulty {} accepted'.format(bad))


def _middle_half(name):
  parameter = adr.PARAMETERS[adr.PARAMETER_INDEX[name]]
  span = parameter.maximum - parameter.minimum
  return (parameter.minimum + 0.25 * span, parameter.minimum + 0.75 * span)


if __name__ == '__main__':
  for test_name, test in sorted(globals().items()):
    if test_name.startswith('test_'):
      test()
      print('ok', test_name)
