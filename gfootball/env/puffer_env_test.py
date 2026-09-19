"""Smoke test for the native PufferLib interface."""

import math
from types import SimpleNamespace

from absl.testing import absltest
import gymnasium
import numpy as np
import torch

from gfootball.env import puffer_env
from gfootball.env import config
from gfootball.env.puffer_policy import FootballPolicy, save_policy_snapshot
from gfootball.curriculum import (
    ADVANTAGE_ENV_NAME, ADVANTAGE_LEVELS, ALIGNMENT_SCHEDULE,
    ATTACKER_ONLY_LEVELS, FULL_LATERAL_HALF_WIDTH, KEEPER_LEVELS,
    MAX_GOALSIDE_BLOCKERS, NARROW_LATERAL_HALF_WIDTH, SPAWN_TEMPLATE_COUNT,
    TOTAL_LEVELS, advantage_for_level, curriculum_episode,
    curriculum_geometry, curriculum_state, expected_goalside_blockers,
    goalside_blockers, keeper_spawn_offset, lateral_half_width)


def _advantage_config(advantage, evaluation=True, seed=0):
  return config.Config({
      'level': ADVANTAGE_ENV_NAME,
      'advantage': advantage,
      'curriculum_level': 0,
      'curriculum_levels': ADVANTAGE_LEVELS,
      'curriculum_evaluation': evaluation,
      'game_engine_random_seed': seed,
      'players': ['agent:left_players=11,right_players=11'],
  })


def _blocker_world_offsets(cfg, count):
  """Lateral world offsets (defender minus ball) of the first blockers."""
  scenario = cfg.ScenarioConfig()
  attack_right = scenario.ball_position[0] > 0
  defence = scenario.right_team if attack_right else scenario.left_team
  # AddPlayer flips both axes for the right team, so undo that for world y.
  side = -1.0 if attack_right else 1.0
  return [side * defence[1 + rank].position[1] - scenario.ball_position[1]
          for rank in range(count)]


class PufferEnvTest(absltest.TestCase):

  def test_score_reward_is_centralized_per_team(self):
    active = np.zeros(22, dtype=bool)
    active[[1, 4, 12, 18]] = True
    rewards = puffer_env.centralized_score_rewards(1, active)
    np.testing.assert_array_equal(
        rewards, [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0])

  def test_observations_are_normalized_and_egocentric(self):
    observations = np.zeros((1, 115), dtype=np.float32)
    own_positions = observations[:, :22].reshape(1, 11, 2)
    own_directions = observations[:, 22:44].reshape(1, 11, 2)
    opponent_positions = observations[:, 44:66].reshape(1, 11, 2)
    own_positions[0, 1] = (0.5, 0.21)
    own_positions[0, 2] = (-0.5, -0.21)
    own_directions[0, 1] = (0.02, -0.01)
    opponent_positions[0, 0] = (-1, -0.42)
    opponent_positions[0, 10] = -1
    observations[0, 88:91] = (1, 0.42, 3)
    observations[0, 91:94] = (0.04, -0.02, 2)
    observations[0, 98] = 1

    puffer_env.normalize_egocentric(observations)

    np.testing.assert_array_equal(own_positions[0, 1], (0, 0))
    np.testing.assert_allclose(
        own_positions[0, 2],
        np.array((-1, -0.42)) / puffer_env._RELATIVE_POSITION_MAX)
    np.testing.assert_allclose(own_directions[0, 1], (0, 0))
    np.testing.assert_allclose(
        observations[0, 88:90],
        np.array((0.5, 0.21)) / puffer_env._RELATIVE_POSITION_MAX)
    self.assertAlmostEqual(observations[0, 90], 3 / 5.5)
    np.testing.assert_array_equal(opponent_positions[0, 10], (-1, -1))
    self.assertGreaterEqual(observations.min(), -1)
    self.assertLessEqual(observations.max(), 1)

  def test_reset_and_step_use_fixed_buffers(self):
    env = puffer_env.FootballPufferEnv(
        env_name='tests.symmetric', seed=7, frame_stack=4)
    try:
      observations, infos = env.reset()
      self.assertEqual(observations.shape, (22, 460))
      self.assertGreaterEqual(observations.min(), -1)
      self.assertLessEqual(observations.max(), 1)
      frames = observations.reshape(22, 4, 115)
      active = frames[:, :, 97:108].argmax(axis=-1)
      np.testing.assert_array_equal(
          active[:, -1], np.tile(np.arange(11), 2))
      own_positions = frames[:, :, :22].reshape(22, 4, 11, 2)
      # Players are sorted by distance from the controlled player, so the
      # controlled player is always slot 0 rather than the slot named by the
      # active-player one-hot.
      np.testing.assert_allclose(own_positions[:, :, 0], 0, atol=1e-6)
      distances = np.linalg.norm(own_positions, axis=-1)
      finite = np.where(np.isclose(distances, np.sqrt(2)), np.inf, distances)
      self.assertTrue(bool(np.all(np.diff(finite, axis=-1) >= -1e-6)))
      np.testing.assert_array_equal(observations[:, :115],
                                    observations[:, 115:230])
      self.assertEqual(infos, [])
      observations, rewards, terminals, truncations, _ = env.step(
          np.zeros(22, dtype=np.int32))
      self.assertEqual(observations.shape, (22, 460))
      self.assertEqual(rewards.shape, (22,))
      self.assertEqual(terminals.shape, (22,))
      self.assertEqual(truncations.shape, (22,))
    finally:
      env.close()

  def test_curriculum_adds_players_before_moving_from_goal(self):
    cfg = config.Config({
        'level': '11_vs_11_curriculum',
        'curriculum_level': 0,
        'curriculum_levels': TOTAL_LEVELS,
        'game_engine_random_seed': 7,
        'players': ['agent:left_players=11,right_players=11'],
    })
    cfg.NewScenario(0)
    initial = cfg.ScenarioConfig()
    self.assertFalse(initial.use_magnet)
    self.assertLen(initial.left_team, 11)
    self.assertLen(initial.right_team, 11)
    self.assertGreaterEqual(abs(initial.ball_position[0]), 0.88)
    self.assertLessEqual(abs(initial.ball_position[0]), 0.92)
    self.assertEqual(initial.game_duration, 119)
    ball = tuple(initial.ball_position[i] for i in range(2))
    if ball[0] > 0:
      attackers, attacker_side = initial.left_team[1:], 1
      defenders, defender_side = initial.right_team[1:], -1
    else:
      attackers, attacker_side = initial.right_team[1:], -1
      defenders, defender_side = initial.left_team[1:], 1
    attacker_distances = [
        math.hypot(attacker_side * player.position[0] - ball[0],
                   attacker_side * player.position[1] - ball[1])
        for player in attackers]
    defender_distances = [
        math.hypot(defender_side * player.position[0] - ball[0],
                   defender_side * player.position[1] - ball[1])
        for player in defenders]
    expected_attackers = curriculum_episode(0, 7, 1)[0]
    self.assertEqual(
        sum(distance < 0.25 for distance in attacker_distances),
        expected_attackers)
    self.assertEqual(sum(distance < 0.25 for distance in defender_distances), 0)
    self.assertGreater(np.median(defender_distances), 0.4)
    defending_team = initial.right_team if ball[0] > 0 else initial.left_team
    self.assertAlmostEqual(abs(defending_team[0].position[1]), 0.36)
    self.assertEqual(initial.reverse_team_processing, ball[0] < 0)
    # The keeper walks into the goal one level at a time.
    for level, offset in enumerate((0.36, 0.0)):
      cfg['curriculum_level'] = level
      cfg.NewScenario(0)
      keeper_level = cfg.ScenarioConfig()
      defending = (keeper_level.right_team
                   if keeper_level.ball_position[0] > 0
                   else keeper_level.left_team)
      self.assertAlmostEqual(abs(defending[0].position[1]), offset)
      self.assertEqual(keeper_level.game_duration, 119)
    cfg['curriculum_level'] = 13
    cfg.NewScenario(0)
    misaligned = cfg.ScenarioConfig()
    self.assertFalse(misaligned.use_magnet)
    self.assertAlmostEqual(abs(misaligned.ball_position[0]), 0.90)
    self.assertEqual(misaligned.game_duration, 119)
    cfg['curriculum_level'] = 25
    cfg.NewScenario(0)
    all_attackers = cfg.ScenarioConfig()
    self.assertAlmostEqual(abs(all_attackers.ball_position[0]), 0.90)
    self.assertEqual(all_attackers.game_duration, 519)
    cfg['curriculum_level'] = 35
    cfg.NewScenario(0)
    full_near_goal = cfg.ScenarioConfig()
    self.assertAlmostEqual(abs(full_near_goal.ball_position[0]), 0.90)
    self.assertEqual(full_near_goal.game_duration, 599)
    cfg['curriculum_level'] = 40
    cfg.NewScenario(0)
    middle = cfg.ScenarioConfig()
    self.assertAlmostEqual(abs(middle.ball_position[0]), 0.45)
    self.assertEqual(middle.game_duration, 1799)
    cfg['curriculum_level'] = TOTAL_LEVELS - 1
    cfg.NewScenario(0)
    mature = cfg.ScenarioConfig()
    self.assertAlmostEqual(mature.ball_position[0], 0.0)
    self.assertEqual(mature.game_duration, 3000)

  def test_unsorted_mode_keeps_the_raw_slot_order(self):
    env = puffer_env.FootballPufferEnv(
        env_name='tests.symmetric', seed=7, frame_stack=1,
        sort_players=False)
    try:
      observations, _ = env.reset()
      frames = observations.reshape(22, 115)
      active = frames[:, 97:108].argmax(axis=-1)
      own = frames[:, :22].reshape(22, 11, 2)
      # Without sorting the controlled player sits at the slot its one-hot
      # names, which is what simple115v2 does natively.
      np.testing.assert_allclose(
          own[np.arange(22), active], 0, atol=1e-6)
    finally:
      env.close()

  def test_curriculum_player_counts(self):
    self.assertEqual(curriculum_state(0), (1, 0, 0.0))
    self.assertEqual(curriculum_state(13), (1, 0, 0.0))
    self.assertEqual(curriculum_state(14), (2, 0, 0.0))
    self.assertEqual(curriculum_state(17), (3, 0, 0.0))
    self.assertEqual(curriculum_state(25), (11, 0, 0.0))
    self.assertEqual(curriculum_state(26), (11, 1, 0.0))
    self.assertEqual(curriculum_state(35), (11, 10, 0.0))
    self.assertAlmostEqual(curriculum_state(40)[2], 0.5)
    self.assertEqual(curriculum_state(TOTAL_LEVELS - 1), (11, 10, 1.0))
    self.assertEqual(ATTACKER_ONLY_LEVELS, 26)

  def test_alignment_levels_sit_on_the_measured_transition(self):
    # A constant-shot reference scores 1.000 at alignment 0.0, 0.432 at 0.2
    # and 0.000 at 0.4, so the schedule has to be dense inside [0, 0.4]
    # rather than spread evenly to 1.0.
    self.assertEqual(list(ALIGNMENT_SCHEDULE), sorted(ALIGNMENT_SCHEDULE))
    self.assertEqual(ALIGNMENT_SCHEDULE[-1], 1.0)
    inside = [value for value in ALIGNMENT_SCHEDULE if value <= 0.4]
    outside = len(ALIGNMENT_SCHEDULE) - len(inside)
    self.assertGreaterEqual(len(inside), 2 * outside)
    self.assertLessEqual(
        max(b - a for a, b in zip(inside, inside[1:])), 0.0501)

  def test_early_levels_change_exactly_one_thing(self):
    keeper = [curriculum_geometry(level)[0] for level in range(TOTAL_LEVELS)]
    alignment = [
        curriculum_geometry(level)[1] for level in range(TOTAL_LEVELS)]
    self.assertEqual((keeper[0], alignment[0]), (0.0, 0.0))
    self.assertEqual((keeper[-1], alignment[-1]), (1.0, 1.0))
    self.assertEqual(keeper, sorted(keeper))
    self.assertEqual(alignment, sorted(alignment))
    # The keeper finishes closing the goal before the carrier is ever
    # misaligned, so no early level moves two knobs at once.
    for level in range(TOTAL_LEVELS):
      if alignment[level] > 0:
        self.assertEqual(keeper[level], 1.0)
    # Level 0 stays the open-goal, aligned-carrier anchor.
    self.assertAlmostEqual(keeper_spawn_offset(0), 0.36)
    self.assertAlmostEqual(keeper_spawn_offset(TOTAL_LEVELS - 1), 0.0)
    offsets = [keeper_spawn_offset(level) for level in range(KEEPER_LEVELS)]
    self.assertEqual(len(set(offsets)), KEEPER_LEVELS)

  def test_second_attacker_fades_in_over_several_levels(self):
    for level, expected in ((13, 0.0), (14, 1 / 3), (15, 2 / 3), (16, 1.0)):
      attackers = [
          curriculum_episode(level, 7, episode)[0]
          for episode in range(2000)
      ]
      self.assertAlmostEqual(
          sum(count == 2 for count in attackers) / len(attackers),
          expected, delta=0.04)
    # The mix phase ends on two attackers, so the step to three is one player.
    self.assertEqual(
        {curriculum_episode(16, 7, episode)[0] for episode in range(200)},
        {2})
    self.assertEqual(
        {curriculum_episode(17, 7, episode)[0] for episode in range(200)},
        {3})

  def test_training_spawns_cover_heldout_template_angles(self):
    cfg = config.Config({
        'level': '11_vs_11_curriculum',
        'curriculum_level': 0,
        'curriculum_levels': TOTAL_LEVELS,
        'game_engine_random_seed': 0,
        'players': ['agent:left_players=11,right_players=11'],
    })
    depths = []
    for episode in range(8):
      cfg.NewScenario(episode)
      scenario = cfg.ScenarioConfig()
      attack_right = scenario.ball_position[0] > 0
      team = scenario.left_team if attack_right else scenario.right_team
      side = 1 if attack_right else -1
      carrier = team[2].position
      offset = side * carrier[1] - scenario.ball_position[1]
      gap = side * (scenario.ball_position[0] - side * carrier[0])
      phase = (scenario.ball_position[1] / 0.03 + 1) / 2
      template = cfg._values['curriculum_episode_template']
      depths.append(abs(scenario.ball_position[0]))
      self.assertGreaterEqual(phase, 1 / 6 + 2 / 3 * template / 8)
      self.assertLess(phase, 1 / 6 + 2 / 3 * (template + 1) / 8)
      self.assertAlmostEqual(gap, 0.03)
      self.assertLess(abs(offset), 0.007)
      self.assertGreaterEqual(depths[-1], 0.88)
      self.assertLessEqual(depths[-1], 0.92)

    self.assertGreater(max(depths) - min(depths), 0.025)

    cfg['curriculum_evaluation'] = True
    for episode in range(8):
      cfg.NewScenario(episode)
      phase = (cfg.ScenarioConfig().ball_position[1] / 0.03 + 1) / 2
      template = cfg._values['curriculum_episode_template']
      self.assertAlmostEqual(
          phase, 1 / 6 + 2 / 3 * (template + 0.5) / 8)

  def test_inactive_curriculum_players_are_hidden_and_forced_idle(self):
    env = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
    try:
      observations, _ = env.reset()
      expected_active = env._episode_attackers + 1
      self.assertEqual(env._active_mask.sum(), expected_active)
      self.assertEqual(np.any(observations, axis=1).sum(), expected_active)
      raw_env = env._env.unwrapped._env
      initial = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      class RecordingEnv:
        def __init__(self, wrapped):
          self.wrapped = wrapped
          self.actions = None

        def step(self, actions):
          self.actions = np.asarray(actions).copy()
          return self.wrapped.step(actions)

        def __getattr__(self, name):
          return getattr(self.wrapped, name)

      env._env = RecordingEnv(env._env)
      requested = np.arange(22, dtype=np.int32) % 19
      expected = requested.copy()
      expected[~env._active_mask] = 0
      env.step(requested)
      np.testing.assert_array_equal(env._env.actions, expected)
      for _ in range(4):
        env.step(np.full(22, 5, dtype=np.int32))
      after = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      movement = np.linalg.norm(after - initial, axis=1)
      self.assertLess(movement[~env._active_mask].max(), 0.002)
      self.assertGreater(movement[env._active_mask].max(), 0.002)
    finally:
      env.close()

  def test_attacker_only_levels_keep_physical_goalkeeper_out_of_learning(self):
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=7, attacker_only_levels=11)
    try:
      observations, _ = env.reset()
      self.assertEqual(env._active_mask.sum(), env._episode_attackers)
      self.assertEqual(
          np.any(observations, axis=1).sum(), env._episode_attackers)
      raw = env._env.unwrapped._env.observation()
      self.assertLen(raw['left_team'], 11)
      self.assertLen(raw['right_team'], 11)
      defending_goalkeeper = 0 if env._attacking_left else 11
      self.assertFalse(env._active_mask[defending_goalkeeper])
      env.step(np.zeros(22, dtype=np.int32))
    finally:
      env.close()

  def test_later_levels_without_magnet_require_direction_to_move(self):
    env = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
    env._curriculum_level = 1
    try:
      env.reset()
      raw_env = env._env.unwrapped._env
      initial = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      for _ in range(5):
        env.step(np.full(22, 10, dtype=np.int32))
      after_pass = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      self.assertLess(np.median(np.linalg.norm(
          after_pass - initial, axis=1)[env._active_mask]), 0.002)

      env.reset()
      initial = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      for _ in range(5):
        env.step(np.full(22, 5, dtype=np.int32))
      after_move = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      self.assertGreater(
          np.median(np.linalg.norm(
              after_move - initial, axis=1)[env._active_mask]), 0.002)
    finally:
      env.close()

  def test_first_curriculum_attempt_uses_short_credit_horizon(self):
    env = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
    try:
      env.reset()
      for episode_length in range(1, 602):
        _, _, terminals, _, _ = env.step(np.zeros(22, dtype=np.int32))
        if terminals.all():
          break
      self.assertEqual(episode_length, 120)
    finally:
      env.close()

  def test_curriculum_advances_only_after_mastery_window(self):
    env = puffer_env.FootballPufferEnv(
        env_name='tests.symmetric', curriculum_levels=3,
        curriculum_window=3, curriculum_success_threshold=2 / 3)
    try:
      self.assertFalse(env._record_curriculum_result(True)[1])
      self.assertFalse(env._record_curriculum_result(False)[1])
      success_rate, advanced = env._record_curriculum_result(True)
      self.assertTrue(advanced)
      self.assertAlmostEqual(success_rate, 2 / 3)
      self.assertEqual(env._curriculum_level, 1)
      self.assertEmpty(env._curriculum_results)
    finally:
      env.close()

  def test_shared_curriculum_moves_all_matches_together(self):
    from multiprocessing import RawValue
    level = RawValue('i', 0)
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=7, attacker_only_levels=ATTACKER_ONLY_LEVELS,
        curriculum_level_value=level)
    try:
      env.reset()
      level.value = 1
      env.reset()
      self.assertEqual(env._episode_level, 1)
      self.assertEqual(env._active_mask.sum(), 1)
      level.value = 25
      env.reset()
      self.assertEqual(env._episode_level, 25)
      self.assertEqual(env._active_mask.sum(), 11)
      level.value = 26
      env.reset()
      self.assertEqual(env._episode_level, 26)
      # Eleven attackers, plus the now-controllable keeper and one defender.
      self.assertEqual(env._active_mask.sum(), 13)
    finally:
      env.close()

  def test_two_matches_run_in_parallel(self):
    env = puffer_env.make_vector_env(
        num_envs=2, num_workers=2, batch_size=2, reserved_cpus=0,
        env_name='tests.symmetric', frame_stack=4)
    try:
      observations, _ = env.reset(seed=7)
      self.assertEqual(observations.shape, (44, 460))
      active = observations.reshape(44, 4, 115)[:, -1, 97:108].argmax(-1)
      np.testing.assert_array_equal(active, np.tile(np.arange(11), 4))
      observations, rewards, terminals, truncations, _ = env.step(
          np.zeros(44, dtype=np.int32))
      self.assertEqual(observations.shape, (44, 460))
      self.assertEqual(rewards.shape, (44,))
      self.assertEqual(terminals.shape, (44,))
      self.assertEqual(truncations.shape, (44,))
    finally:
      env.close()

  def test_level_zero_lateral_band_stays_inside_the_scoreable_region(self):
    """Level 0 must not spawn the ball where nothing can score from.

    Measured at advantage 1.00: success is 0.98 at |ball_y| 0.088 but 0.13 by
    0.113, and always-shot is zero beyond 0.079.  The anchor level has to sit
    inside that, and the band has to widen monotonically after it.
    """
    self.assertAlmostEqual(
        lateral_half_width(1.0), NARROW_LATERAL_HALF_WIDTH)
    self.assertAlmostEqual(
        lateral_half_width(0.4), FULL_LATERAL_HALF_WIDTH)
    self.assertAlmostEqual(
        lateral_half_width(0.0), FULL_LATERAL_HALF_WIDTH)
    widths = [lateral_half_width(advantage_for_level(level))
              for level in range(ADVANTAGE_LEVELS)]
    self.assertEqual(widths, sorted(widths))
    self.assertLess(widths[0], 0.10)
    # The band holds at level-0 width until the blocker fade is complete
    # (level 5), so the two ramps never move on the same level.
    for level in range(6):
      self.assertAlmostEqual(widths[level], NARROW_LATERAL_HALF_WIDTH)
      self.assertAlmostEqual(
          expected_goalside_blockers(advantage_for_level(level)),
          1.0 + 0.2 * level)
    self.assertGreater(widths[6], NARROW_LATERAL_HALF_WIDTH)
    self.assertAlmostEqual(
        expected_goalside_blockers(advantage_for_level(6)), 2.2)

    cfg = config.Config({
        'level': ADVANTAGE_ENV_NAME,
        'advantage': 1.0,
        'curriculum_level': 0,
        'curriculum_levels': ADVANTAGE_LEVELS,
        'curriculum_evaluation': True,
        'game_engine_random_seed': 0,
        'players': ['agent:left_players=11,right_players=11'],
    })
    templates = {}
    for episode in range(2 * SPAWN_TEMPLATE_COUNT):
      cfg.NewScenario(episode)
      ball_y = cfg.ScenarioConfig().ball_position[1]
      self.assertLessEqual(abs(ball_y), NARROW_LATERAL_HALF_WIDTH + 1e-6)
      templates[int(cfg._values['curriculum_episode_template'])] = ball_y
    self.assertEqual(len(templates), SPAWN_TEMPLATE_COUNT)
    # Every template still names a distinct lateral spawn inside the band.
    self.assertEqual(len(set(round(y, 6) for y in templates.values())),
                     SPAWN_TEMPLATE_COUNT)

  def test_goalside_blockers_fade_in_by_at_most_a_fifth_per_level(self):
    """No level may add a whole blocker at once; level 0 keeps exactly one.

    Three seeds cleared levels 0-2 and collapsed on level 3, which is where
    the rounded schedule dropped the second blocker in one step.
    """
    draws = [(k + 0.5) / 1000 for k in range(1000)]
    for draw in (0.0, 0.5, 0.999):
      self.assertEqual(goalside_blockers(1.0, draw), 1)
      self.assertEqual(goalside_blockers(0.0, draw), MAX_GOALSIDE_BLOCKERS)
    means = []
    for level in range(ADVANTAGE_LEVELS):
      advantage = advantage_for_level(level)
      counts = [goalside_blockers(advantage, draw) for draw in draws]
      self.assertLessEqual(max(counts) - min(counts), 1)
      means.append(sum(counts) / len(counts))
      self.assertAlmostEqual(
          means[-1], expected_goalside_blockers(advantage), places=2)
    self.assertEqual(means, sorted(means))
    for earlier, later in zip(means, means[1:]):
      self.assertLessEqual(later - earlier, 0.2 + 1e-6)
    # Level 3 used to be a hard two-blocker level; now it is 60% likely.
    self.assertAlmostEqual(means[3], 1.6, places=2)

  def test_level_zero_blocker_geometry_is_the_measured_anchor(self):
    cfg = _advantage_config(1.0)
    for _ in range(2 * SPAWN_TEMPLATE_COUNT):
      cfg.NewScenario()
      self.assertEqual(cfg._values['curriculum_goalside_defenders'], 1)
      (offset,) = _blocker_world_offsets(cfg, 1)
      self.assertAlmostEqual(offset, -0.085, delta=0.02)

  def test_second_blocker_mirrors_the_first_and_the_third_fills_the_middle(self):
    """Two blockers must leave the shot line open; only the third closes it."""
    cfg = _advantage_config(0.75)  # exactly two blockers, no fade
    for _ in range(2 * SPAWN_TEMPLATE_COUNT):
      cfg.NewScenario()
      self.assertEqual(cfg._values['curriculum_goalside_defenders'], 2)
      first, second = _blocker_world_offsets(cfg, 2)
      self.assertAlmostEqual(first, -0.085, delta=0.02)
      self.assertAlmostEqual(second, 0.085, delta=0.02)
    cfg = _advantage_config(0.5)  # exactly three
    cfg.NewScenario()
    self.assertEqual(cfg._values['curriculum_goalside_defenders'], 3)
    self.assertAlmostEqual(_blocker_world_offsets(cfg, 3)[2], 0.0, delta=0.02)

  def test_blocker_fade_is_deterministic_per_episode_and_hits_its_rate(self):
    # NewScenario(inc) ADVANCES the episode counter by inc, so a fresh config
    # per pass is what makes the two passes see the same episode numbers.
    def counts_for(passes):
      cfg = _advantage_config(advantage_for_level(3), evaluation=False)
      counts = []
      for _ in range(passes):
        cfg.NewScenario()
        counts.append(cfg._values['curriculum_goalside_defenders'])
      return counts

    counts = counts_for(200)
    self.assertEqual(counts, counts_for(200))
    self.assertEqual(set(counts), {1, 2})
    self.assertAlmostEqual(sum(counts) / len(counts), 1.6, delta=0.1)

  def test_potential_shaping_telescopes_to_minus_the_start_potential(self):
    """Ng et al. (1999): F = gamma*Phi(s') - Phi(s), Phi(absorbing) = 0.

    Summed over any trajectory the shaping is -Phi(s_0) plus the (1-gamma)
    drift, whatever happened in between and however it ended, which is what
    makes it unable to change the optimal policy.
    """
    gamma, scale = 0.9, 2.0
    self.assertEqual(puffer_env.ball_potential(1.0, scale), 0.0)
    self.assertEqual(puffer_env.ball_potential(-1.0, scale), -4.0)
    for advances in ([0.9, 0.95, 1.0], [0.9, 0.5, 0.2, 0.7], [0.3]):
      potentials = [puffer_env.ball_potential(a, scale) for a in advances]
      total = 0.0
      drift = 0.0
      for index in range(1, len(potentials)):
        total += puffer_env.potential_shaping(
            potentials[index - 1], potentials[index], gamma, terminal=False)
        drift += (gamma - 1) * potentials[index]
      # The episode ends: the absorbing state has zero potential.
      total += puffer_env.potential_shaping(
          potentials[-1], potentials[-1], gamma, terminal=True)
      self.assertAlmostEqual(total, -potentials[0] + drift)
    # Progress toward the goal is paid for as it happens ...
    self.assertGreater(puffer_env.potential_shaping(
        puffer_env.ball_potential(0.9, 1.0),
        puffer_env.ball_potential(0.95, 1.0), 1.0, False), 0)
    # ... and moving away costs the same amount back.
    self.assertLess(puffer_env.potential_shaping(
        puffer_env.ball_potential(0.95, 1.0),
        puffer_env.ball_potential(0.9, 1.0), 1.0, False), 0)

  def test_ball_shaping_rewards_both_sides_and_leaves_success_alone(self):
    scale, gamma = 1.0, 0.99
    env = puffer_env.FootballPufferEnv(
        env_name=ADVANTAGE_ENV_NAME, frame_stack=1, seed=7,
        curriculum_levels=ADVANTAGE_LEVELS, ball_potential_scale=scale,
        potential_gamma=gamma)
    plain = puffer_env.FootballPufferEnv(
        env_name=ADVANTAGE_ENV_NAME, frame_stack=1, seed=7,
        curriculum_levels=ADVANTAGE_LEVELS)
    try:
      env.reset()
      plain.reset()
      attacking = slice(0, 11) if env._attacking_left else slice(11, 22)
      defending = slice(11, 22) if env._attacking_left else slice(0, 11)
      start_potential = env._attack_potential
      self.assertLess(start_potential, 0.0)
      # Sprint toward the goal with everyone: the ball carrier advances it.
      shaped_total = 0.0
      infos = []
      for _ in range(400):
        previous_attack = env._attack_potential
        previous_defence = env._defence_potential
        _, rewards, _, _, infos = env.step(np.full(22, 5, dtype=np.int32))
        _, plain_rewards, _, _, _ = plain.step(np.full(22, 5, dtype=np.int32))
        attack = float(rewards[attacking][0])
        defence = float(rewards[defending][0])
        # Every row on a side sees the same shaping ...
        np.testing.assert_allclose(rewards[attacking], attack, atol=1e-6)
        np.testing.assert_allclose(rewards[defending], defence, atol=1e-6)
        score = float(plain_rewards[attacking][0])
        shaped_total += attack - score
        if infos:
          # The absorbing state has zero potential, so the last step refunds
          # each side exactly its own potential, whatever the final frame.
          self.assertAlmostEqual(attack - score, -previous_attack, places=5)
          self.assertAlmostEqual(defence + score, -previous_defence, places=5)
          break
        # ... and until then the two sides are shaped in opposite directions,
        # up to the (1 - gamma) constant, on top of the zero-sum score.
        self.assertAlmostEqual(
            (attack - score) + (defence + score), 2 * scale * (1 - gamma),
            places=5)
      self.assertTrue(infos)
      info = infos[0]
      # Success and the score returns are untouched by shaping.
      self.assertIn(info['curriculum_success'], (0.0, 1.0))
      self.assertEqual(
          info['curriculum_success'],
          float(info['left_episode_return' if env._attacking_left
                     else 'right_episode_return'] > 0))
      self.assertAlmostEqual(
          info['attacking_shaping_return'], shaped_total, places=4)
      # Telescoped: the episode's shaping is -Phi(s_0) plus the tiny drift,
      # which for a spawn near the goal is a small number, not a goal's worth.
      self.assertLess(abs(shaped_total + start_potential), 0.05 + 0.02)
      self.assertLess(abs(shaped_total), 0.2)
    finally:
      env.close()
      plain.close()

  def test_frozen_defence_plays_the_defending_side_from_a_snapshot(self):
    """The frozen side is hidden from the caller and acted in the worker."""
    torch.manual_seed(0)
    spaces = SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(115,), dtype=np.float32),
        single_action_space=gymnasium.spaces.Discrete(19))
    snapshot = self.create_tempfile('frozen_defence.pt').full_path
    save_policy_snapshot(FootballPolicy(spaces, hidden_size=16), snapshot)

    class RecordingEnv:
      def __init__(self, wrapped):
        self.wrapped = wrapped
        self.sent = []

      def step(self, actions):
        self.sent.append(np.asarray(actions).copy())
        return self.wrapped.step(actions)

      def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def make():
      env = puffer_env.FootballPufferEnv(
          env_name=ADVANTAGE_ENV_NAME, frame_stack=1, seed=7,
          curriculum_levels=ADVANTAGE_LEVELS, frozen_defence_path=snapshot,
          frozen_defence_horizon=4)
      env._env = RecordingEnv(env._env)
      return env

    env = make()
    twin = make()
    try:
      observations, _ = env.reset()
      twin.reset()
      defending = env._defending_rows()
      attacking = slice(0, 11) if env._attacking_left else slice(11, 22)
      self.assertFalse(env._active_mask[defending].any())
      self.assertTrue(env._active_mask[attacking].all())
      self.assertFalse(np.any(observations[defending]))
      self.assertTrue(np.all(np.any(observations[attacking], axis=1)))
      # The rows the snapshot acts on are the real, un-hidden observations.
      self.assertTrue(np.all(np.any(
          env._full_observations[defending], axis=1)))

      raw_env = env._env.unwrapped._env
      before = np.concatenate([raw_env.observation()['left_team'],
                               raw_env.observation()['right_team']])
      requested = np.full(22, 5, dtype=np.int32)
      infos = []
      for _ in range(12):
        env.step(requested)
        twin.step(requested)
      sent = np.stack(env._env.sent)
      np.testing.assert_array_equal(sent[:, attacking], 5)
      defender_actions = sent[:, defending]
      self.assertTrue(np.any(defender_actions != 5))
      self.assertGreater(len(np.unique(defender_actions)), 1)
      # Same seed, same snapshot, same caller actions: the same defence.
      np.testing.assert_array_equal(
          defender_actions, np.stack(twin._env.sent)[:, defending])
      after = np.concatenate([raw_env.observation()['left_team'],
                              raw_env.observation()['right_team']])
      movement = np.linalg.norm(after - before, axis=1)
      self.assertGreater(movement[defending].max(), 0.002)

      for _ in range(400):
        _, _, _, _, infos = env.step(requested)
        if infos:
          break
      self.assertTrue(infos)
      self.assertEqual(infos[0]['curriculum_frozen_defence'], 1.0)
      self.assertIn(infos[0]['curriculum_attacking_left'], (0.0, 1.0))
      # A new episode starts the frozen memory from zero again.
      self.assertIsNone(env._frozen_state['lstm_h'])
    finally:
      env.close()
      twin.close()


if __name__ == '__main__':
  absltest.main()
