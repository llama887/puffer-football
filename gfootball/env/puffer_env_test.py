"""Smoke test for the native PufferLib interface."""

import math
from types import SimpleNamespace

from absl.testing import absltest
import gymnasium
import numpy as np
import torch

from gfootball import adr
from gfootball.env import puffer_env
from gfootball.env import config
from gfootball.env import entity_observation
from gfootball.env import observation_rotation
from gfootball.env.puffer_policy import FootballPolicy, save_policy_snapshot


START = {parameter.name: parameter.start for parameter in adr.PARAMETERS}


def _scene_config(seed=0, values=None):
  """A config whose scenario builds ADR scenes from `values` (default: the
  start values)."""
  return config.Config({
      'level': adr.SCENARIO,
      'adr_values': dict(values or START),
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

  def test_entity_observations_describe_one_world_from_both_sides(self):
    """Entity rows come from the live engine, one view per team."""
    env = puffer_env.FootballPufferEnv(
        seed=3, frame_stack=1,
        observation='entities')
    try:
      self.assertEqual(env.single_observation_space.shape,
                       (entity_observation.SIZE,))
      observations, _ = env.reset()
      height = (entity_observation.CONTEXT_FEATURES +
                entity_observation.BALL_FEATURE_INDEX['height'])
      steps_left = entity_observation.CONTEXT_FEATURE_INDEX['steps_left']
      is_self = (entity_observation.PLAYERS_START +
                 entity_observation.PLAYER_FEATURE_INDEX['is_self'])
      for _ in range(3):
        active = np.flatnonzero(observations.any(axis=1))
        self.assertGreater(len(active), 0)
        np.testing.assert_array_equal(observations[active, is_self], 1)
        left = active[active < 11]
        right = active[active >= 11]
        if len(left) and len(right):
          np.testing.assert_allclose(observations[left[0], height],
                                     observations[right[0], height], atol=1e-6)
        before = observations[active[0], steps_left]
        observations, _, terminals, _, _ = env.step(
            np.zeros(22, dtype=np.int32))
        if not terminals[0]:
          self.assertAlmostEqual(
              before - observations[active[0], steps_left],
              1 / entity_observation.STEPS_LEFT_SCALE, places=5)
    finally:
      env.close()
    with self.assertRaises(ValueError):
      puffer_env.FootballPufferEnv(
          frame_stack=4, observation='entities')

  def test_tracked_sticky_buttons_match_the_engine(self):
    """The buttons the observation reports are the ones the engine holds."""
    env = puffer_env.FootballPufferEnv(
        seed=5, frame_stack=1,
        observation='entities')
    rng = np.random.default_rng(0)
    sticky = entity_observation.CONTEXT_FEATURE_INDEX['sticky']
    try:
      observations, _ = env.reset()
      core = env._env.unwrapped._env
      checked = 0
      for _ in range(60):
        actions = rng.integers(0, 19, size=22)
        observations, _, _, _, _ = env.step(actions)
        for agent in range(22):
          left_team = agent < 11
          engine = core.sticky_actions_state(left_team, agent % 11)
          if not left_team:
            engine = observation_rotation.rotate_sticky_actions(
                engine, env._env.unwrapped._config)
          tracked = env._sticky[agent]
          np.testing.assert_array_equal(tracked, engine, err_msg=str(agent))
          if observations[agent].any():
            np.testing.assert_array_equal(
                observations[agent, sticky:sticky +
                             entity_observation.STICKY_ACTIONS], tracked)
          checked += int(np.asarray(engine).any())
      self.assertGreater(checked, 100)
    finally:
      env.close()

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

  def test_scenes_differ_for_every_seed_and_episode(self):
    """Consecutive worker seeds through consecutive episodes are new scenes.

    Evaluation runs worker seeds s, s+1, ... through episodes 0, 1, ...  When
    a scene's randomness depended only on seed + episode, worker i's episode
    e and worker i+1's episode e-1 were the same scene (an old 256-episode
    evaluation replayed 38 scenes), so scenes are keyed on the (seed,
    episode) pair.
    """
    scenes = set()
    for seed, episode in ((100, 5), (101, 4), (102, 3), (105, 0)):
      cfg = _scene_config(seed=seed)
      cfg.NewScenario(episode)
      scenario = cfg.ScenarioConfig()
      scenes.add(tuple(round(p.position[axis], 6)
                       for team in (scenario.left_team, scenario.right_team)
                       for p in team
                       for axis in (0, 1)))
    self.assertLen(scenes, 4)

  def test_episodes_end_when_the_ball_goes_out_of_play(self):
    """With the magnet off no set-piece taker is ever designated.

    A goal kick, corner or throw-in then never restarts and the match sits
    frozen until the timeout, so the episode ends as soon as play stops.
    """
    limits = {p.name: p.maximum for p in adr.PARAMETERS}
    for values in (START, limits):
      cfg = _scene_config(values=values)
      cfg.NewScenario(0)
      self.assertTrue(cfg.ScenarioConfig().end_episode_on_out_of_play)

  def test_attackers_ahead_start_onside_and_out_of_the_shooting_lane(self):
    """attackers_ahead puts that many attackers between the ball and the goal.

    They are onside (no nearer the goal line than the second-last defender)
    and out of the shooting lane, so they never block the carrier's shot.
    """
    for seed in range(25):
      for count in (1, 2, 3):
        for distance in (0.3, 0.55):
          cfg = _scene_config(seed=seed, values=dict(
              START, ball_distance=distance, attackers_ahead=float(count)))
          cfg.NewScenario(seed % 7)
          scenario = cfg.ScenarioConfig()
          attack_right = scenario.ball_position[0] > 0
          flip = 1.0 if attack_right else -1.0
          left = [(flip * p.position[0], flip * p.position[1])
                  for p in scenario.left_team]
          right = [(-flip * p.position[0], -flip * p.position[1])
                   for p in scenario.right_team]
          attackers, defenders = (left, right) if attack_right else (
              right, left)
          ball = (flip * scenario.ball_position[0],
                  flip * scenario.ball_position[1])
          ahead = [(x, y) for x, y in attackers[1:] if x > ball[0] + 0.02]
          self.assertGreaterEqual(len(ahead), count, (count, seed))
          line = max(ball[0], sorted(x for x, _ in defenders)[-2])
          for x, y in attackers:
            self.assertLessEqual(x, line + 1e-6, (count, seed))
          for x, y in ahead:
            # The lane narrows from the posts (|y| = 0.044) to the ball.
            along = (x - ball[0]) / (1.0 - ball[0])
            centre = ball[1] * (1.0 - along)
            self.assertGreater(abs(y - centre), 0.044 * along + 0.015,
                               (count, seed))

  def test_held_directions_are_pressed_on_the_first_step_outside_training(
      self):
    """The first step of a scene with held players (ADR held_share) presses
    the held directions.

    Every row is hidden on that step, so the trainer (which skips all-zero
    rows) never learns from actions the policy did not choose and the step
    pays nothing; from the next step the policy acts on players that are
    already running, with those buttons held.
    """
    shared = adr.shared_ranges()
    adr.AutomaticDomainRandomization(shared=shared).set_range(
        'held_share', 1.0, 1.0)
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=5, observation='entities', adr_shared=shared)
    try:
      observations, _ = env.reset()
      held = np.asarray(env._env.unwrapped._config['held_directions'])
      self.assertGreater((held > 0).sum(), 0)
      self.assertFalse(np.any(observations))
      observations, rewards, _, _, _ = env.step(np.zeros(22, dtype=np.int32))
      self.assertTrue(np.all(np.any(observations, axis=1)))
      self.assertFalse(np.any(rewards))
      sticky_names = entity_observation._STICKY_NAMES
      action_names = entity_observation._ACTION_NAMES
      for agent in np.flatnonzero(held):
        pressed = action_names[held[agent]]
        self.assertEqual(env._sticky[agent, sticky_names.index(pressed)], 1.0)
    finally:
      env.close()

  def test_adr_values_build_the_scene_they_describe(self):
    """With 'adr_values' the scene comes from explicit settings.

    Each value is honoured: the ball's distance from the attacked goal line
    and its y, the number of goal-side blockers (fractions add one by
    chance), the nearest attacker's distance from the still ball (fetch
    distance), the nearest outfield defender's extra distance (head start),
    attackers ahead of the ball (onside, lane clear), the share of players
    holding a direction, and offside once the formation is a real match
    shape.  The start values give the start scene.
    """

    def scene(values, seed):
      cfg = _scene_config(seed=seed, values=values)
      cfg.NewScenario(seed % 7)
      scenario = cfg.ScenarioConfig()
      attack_right = scenario.ball_position[0] > 0
      flip = 1.0 if attack_right else -1.0
      left = [(flip * p.position[0], flip * p.position[1])
              for p in scenario.left_team]
      right = [(-flip * p.position[0], -flip * p.position[1])
               for p in scenario.right_team]
      attackers, defenders = (left, right) if attack_right else (right, left)
      # Everything in the attacking team's frame (x toward its goal).
      ball = (flip * scenario.ball_position[0],
              flip * scenario.ball_position[1])
      return cfg, scenario, ball, attackers, defenders

    def nearest(team, ball):
      return min(math.hypot(x - ball[0], y - ball[1]) for x, y in team)

    start = {p.name: p.start for p in adr.PARAMETERS}
    for seed in range(20):
      cfg, scenario, ball, attackers, defenders = scene(start, seed)
      self.assertAlmostEqual(ball[0], 0.9, places=6)
      self.assertAlmostEqual(nearest(attackers[1:], ball), 0.03, delta=0.01)
      self.assertEqual(cfg['goalside_blockers'], 1)
      # The start ring is squeezed up to 0.04 (the room in front of the
      # ball) ahead against the goal line;
      # nobody is placed ahead of it.
      self.assertEqual(sum(x > ball[0] + 0.05 for x, _ in attackers[1:]), 0)
      self.assertEqual(list(cfg['held_directions']), [0] * 22)
      self.assertFalse(scenario.offsides)

    values = dict(start, ball_distance=0.55, ball_lateral=-0.2, blockers=2.0,
                  fetch_distance=0.4, defender_head_start=0.1,
                  attackers_ahead=2.0, formation=0.5, held_share=1.0)
    for seed in range(20):
      cfg, scenario, ball, attackers, defenders = scene(values, seed)
      self.assertAlmostEqual(ball[0], 0.45, places=6)
      # ball_lateral is a pitch y, the same for both attack directions.
      self.assertAlmostEqual(scenario.ball_position[1], -0.2, places=6)
      self.assertEqual(cfg['goalside_blockers'], 2)
      self.assertAlmostEqual(nearest(attackers[1:], ball), 0.4, delta=0.02)
      self.assertAlmostEqual(nearest(defenders[1:], ball), 0.5, delta=0.03)
      held = list(cfg['held_directions'])
      self.assertEqual(held[0], 0)
      self.assertEqual(held[11], 0)
      self.assertEqual(sum(h > 0 for h in held), 20)
      self.assertEqual(scenario.game_duration, 500)

    counts = []
    for seed in range(200):
      cfg, scenario, ball, attackers, defenders = scene(
          dict(start, blockers=1.5, attackers_ahead=1.5), seed)
      counts.append(cfg['goalside_blockers'])
    self.assertBetween(np.mean(counts), 1.35, 1.65)

    for seed in range(20):
      cfg, scenario, ball, attackers, defenders = scene(
          dict(start, ball_distance=0.9, formation=1.0, attackers_ahead=2.0),
          seed)
      self.assertTrue(scenario.offsides)
      second_last = sorted(x for x, _ in defenders)[-2]
      for x, _ in attackers:
        self.assertLessEqual(x, max(ball[0], second_last) + 1e-6)

  def test_adr_env_draws_each_episode_from_the_live_shared_ranges(self):
    """An ADR env samples every episode from the shared ranges, so a range
    the trainer moves takes effect at the next episode in every worker."""
    shared = adr.shared_ranges()
    randomizer = adr.AutomaticDomainRandomization(shared=shared)
    randomizer.set_range('ball_distance', 0.3, 0.3)
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=3,
        adr_shared=shared)
    try:
      env.reset()
      ball = env._env.unwrapped._config.ScenarioConfig().ball_position[0]
      # The engine rewrites the scenario in metres at reset (x * 54.4).
      self.assertAlmostEqual(abs(ball) / 54.4, 0.7, places=4)
      randomizer.set_range('ball_distance', 0.5, 0.5)
      env.reset()
      ball = env._env.unwrapped._config.ScenarioConfig().ball_position[0]
      self.assertAlmostEqual(abs(ball) / 54.4, 0.5, places=4)
    finally:
      env.close()
    with self.assertRaisesRegex(ValueError, 'adr_probe_fraction'):
      puffer_env.FootballPufferEnv(adr_shared=shared, adr_probe_fraction=1.5)

  def test_evaluation_scenes_replay_exactly_after_every_reseed(self):
    """An env on fixed test ranges (no probes) draws the same scenes after
    every reset(seed): an evaluation pool is reused between checks, and each
    check has to replay the episodes a freshly built pool would."""
    shared = adr.shared_ranges()
    adr.AutomaticDomainRandomization(shared=shared).set_difficulty(0.5)
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=3, adr_shared=shared)

    def scenes():
      drawn = []
      for episode in range(3):
        env.reset(seed=100) if episode == 0 else env.reset()
        ball = env._env.unwrapped._config.ScenarioConfig().ball_position
        drawn.append((dict(env._adr_values), (ball[0], ball[1])))
      return drawn

    try:
      first = scenes()
      self.assertNotEqual(first[0][0], first[1][0])
      self.assertEqual(scenes(), first)
    finally:
      env.close()

  def test_adr_probe_episodes_play_random_defenders_outside_training(self):
    """A probe episode pins one setting's edge, the defending side plays
    uniform random actions with its rows hidden (so it is never trained),
    and the episode end reports the setting, the edge and the result."""
    shared = adr.shared_ranges()
    for fraction, probe in ((1.0, True), (0.0, False)):
      env = puffer_env.FootballPufferEnv(
          frame_stack=1, seed=11,
          adr_shared=shared,
          adr_probe_fraction=fraction)
      try:
        observations, _ = env.reset()
        defending = env._defending_rows()
        attacking = slice(0, 11) if defending.start == 11 else slice(11, 22)
        self.assertEqual(not np.any(observations[defending]), probe)
        self.assertTrue(np.all(np.any(observations[attacking], axis=1)))
        infos = []
        for _ in range(800):
          _, _, terminals, _, infos = env.step(np.zeros(22, dtype=np.int32))
          if infos:
            break
        self.assertTrue(infos, 'episode must end')
        info = infos[0]
        self.assertEqual(info['adr_probe'], float(probe))
        if probe:
          self.assertIn(int(info['adr_parameter']), range(len(adr.PARAMETERS)))
          self.assertIn(int(info['adr_edge']), (adr.LOWER, adr.UPPER))
        else:
          self.assertEqual(info['adr_edge'], -1.0)
        self.assertIn('success', info)
        self.assertAlmostEqual(info['adr_ball_distance'], 0.10, places=6)
      finally:
        env.close()

  def test_adr_probe_defence_snapshot_plays_probes_and_reloads_when_replaced(
      self):
    """With adr_probe_defence set to a snapshot, probe episodes are defended
    by that snapshot instead of random actions (its rows hidden, so it is
    never trained), training episodes stay self-play, and a snapshot replaced
    on disk takes over at the next episode: that is how the trainer
    refreshes the probe opponent."""
    spaces = SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(115,), dtype=np.float32),
        single_action_space=gymnasium.spaces.Discrete(19))

    def save_always(action, path):
      policy = FootballPolicy(spaces, hidden_size=16)
      with torch.no_grad():
        policy.actor.weight.zero_()
        policy.actor.bias.fill_(-50.0)
        policy.actor.bias[action] = 50.0
      save_policy_snapshot(policy, path)

    snapshot = self.create_tempfile('probe_defence.pt').full_path
    save_always(5, snapshot)
    shared = adr.shared_ranges()
    probing = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=11,
        adr_shared=shared,
        adr_probe_fraction=1.0, adr_probe_defence=snapshot)
    try:
      observations, _ = probing.reset()
      self.assertFalse(np.any(observations[probing._defending_rows()]))
      np.testing.assert_array_equal(probing._frozen_actions(), np.full(11, 5))
      save_always(9, snapshot)
      probing.reset()
      np.testing.assert_array_equal(probing._frozen_actions(), np.full(11, 9))
      for _ in range(5):
        probing.step(np.zeros(22, dtype=np.int32))
    finally:
      probing.close()
    training = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=11,
        adr_shared=shared,
        adr_probe_fraction=0.0, adr_probe_defence=snapshot)
    try:
      observations, _ = training.reset()
      self.assertTrue(np.all(np.any(observations, axis=1)))
    finally:
      training.close()

  def test_random_frozen_defence_needs_no_snapshot(self):
    """frozen_defence_path='random' plays the defending side with uniform
    random actions (the base random policy) and loads no file."""
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=4,
        frozen_defence_path='random')
    try:
      observations, _ = env.reset()
      self.assertFalse(np.any(observations[env._defending_rows()]))
      drawn = np.concatenate([env._frozen_actions() for _ in range(50)])
      self.assertEqual(drawn.shape, (550,))
      self.assertGreater(len(set(drawn.tolist())), 15)
      self.assertTrue(np.all((drawn >= 0) & (drawn < 19)))
      for _ in range(20):
        env.step(np.zeros(22, dtype=np.int32))
    finally:
      env.close()

  def test_without_magnet_players_need_a_direction_to_move(self):
    env = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
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

  def test_lone_blocker_side_is_mirrored_at_random(self):
    """A lone blocker takes either lane, so mirror scenes are equally hard.

    It used to sit on the world -y side of the ball every time, which put it
    inside the goal mouth only for balls on the +y side: one side of the
    pitch scored 0.50 while its mirror scored 0.64-0.97.
    """
    sides = {True: set(), False: set()}
    offsets = []
    for seed in range(40):
      cfg = _scene_config(seed=seed)
      for _ in range(16):
        cfg.NewScenario()
        self.assertEqual(cfg._values['goalside_blockers'], 1)
        (offset,) = _blocker_world_offsets(cfg, 1)
        self.assertAlmostEqual(abs(offset), 0.085, delta=0.02)
        offsets.append(offset)
        sides[cfg.ScenarioConfig().ball_position[0] > 0].add(offset > 0)
    self.assertAlmostEqual(
        sum(offset > 0 for offset in offsets) / len(offsets), 0.5, delta=0.1)
    for attack_right, seen in sides.items():
      self.assertEqual(seen, {True, False}, attack_right)

  def test_second_blocker_mirrors_the_first_and_the_third_fills_the_middle(self):
    """Two blockers must leave the shot line open; only the third closes it."""
    cfg = _scene_config(values=dict(START, ball_distance=0.55, blockers=2.0))
    for _ in range(16):
      cfg.NewScenario()
      self.assertEqual(cfg._values['goalside_blockers'], 2)
      # Mirroring may swap which of the pair takes which lane.
      low, high = sorted(_blocker_world_offsets(cfg, 2))
      self.assertAlmostEqual(low, -0.085, delta=0.02)
      self.assertAlmostEqual(high, 0.085, delta=0.02)
    cfg = _scene_config(values=dict(START, ball_distance=0.55, blockers=3.0))
    cfg.NewScenario()
    self.assertEqual(cfg._values['goalside_blockers'], 3)
    self.assertAlmostEqual(_blocker_world_offsets(cfg, 3)[2], 0.0, delta=0.02)

  def test_potential_shaping_telescopes_to_minus_the_start_potential(self):
    """Ng et al. (1999): F = gamma*Phi(s') - Phi(s), Phi(absorbing) = 0.

    Discounted over any complete trajectory the shaping is -Phi(s_0),
    independent of its length, intermediate states, or terminal outcome.
    This is the return identity required for policy invariance.
    """
    gamma, scale = 0.9, 2.0
    self.assertEqual(puffer_env.ball_potential(1.0, scale), 2.0)
    self.assertEqual(puffer_env.ball_potential(-1.0, scale), -2.0)
    for advances in ([0.9, 0.95, 1.0], [0.9, 0.5, 0.2, 0.7], [0.3]):
      potentials = [puffer_env.ball_potential(a, scale) for a in advances]
      total = 0.0
      for index in range(1, len(potentials)):
        total += gamma ** (index - 1) * puffer_env.potential_shaping(
            potentials[index - 1], potentials[index], gamma, terminal=False)
      # The episode ends: the absorbing state has zero potential.
      total += gamma ** (len(potentials) - 1) * puffer_env.potential_shaping(
          potentials[-1], potentials[-1], gamma, terminal=True)
      self.assertAlmostEqual(total, -potentials[0])
    # Progress toward the goal is paid for as it happens ...
    self.assertGreater(puffer_env.potential_shaping(
        puffer_env.ball_potential(0.9, 1.0),
        puffer_env.ball_potential(0.95, 1.0), 1.0, False), 0)
    # ... and moving away costs the same amount back.
    self.assertLess(puffer_env.potential_shaping(
        puffer_env.ball_potential(0.95, 1.0),
        puffer_env.ball_potential(0.9, 1.0), 1.0, False), 0)

  def test_ball_shaping_is_zero_sum_between_the_teams(self):
    """Each team's ball potential is the other's negative.

    The score is zero-sum across the 22 players, so the shaping must be too:
    on every step, including the terminal correction and across an episode
    reset, all 22 rewards sum to zero.
    """
    for advance in (-1.0, -0.3, 0.0, 0.9, 1.0):
      self.assertEqual(puffer_env.ball_potential(advance, 2.0) +
                       puffer_env.ball_potential(-advance, 2.0), 0.0)
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=7,
        ball_potential_scale=1.0,
        potential_gamma=0.997)
    try:
      env.reset()
      episodes = 0
      for _ in range(1500):
        _, rewards, terminals, _, _ = env.step(np.full(22, 5, dtype=np.int32))
        self.assertAlmostEqual(float(np.sum(rewards)), 0.0, places=4)
        episodes += int(np.asarray(terminals).any())
        if episodes == 2:
          break
      self.assertEqual(episodes, 2, 'test must cross an episode reset')
    finally:
      env.close()

  def test_closest_player_potential_geometry_and_identity_changes(self):
    """Use the ball position, physical distance, and only present players."""
    positions = np.full((11, 2), -1.0)
    ball = np.array((0.6, 0.1))
    positions[0] = (0.4, 0.1)
    positions[1] = (0.5, 0.3)  # Nearer in x, farther in physical distance.
    self.assertAlmostEqual(
        puffer_env.closest_player_potential(positions, ball, 2), -0.4)
    mirrored = -positions
    mirrored[2:] = -1  # Absent sentinel does not get mirrored.
    self.assertAlmostEqual(
        puffer_env.closest_player_potential(mirrored, -ball, 2), -0.4)
    positions[[0, 1]] = positions[[1, 0]]
    self.assertAlmostEqual(
        puffer_env.closest_player_potential(positions, ball, 2), -0.4)
    # Moving the ball onto a stationary teammate changes the potential.
    self.assertEqual(
        puffer_env.closest_player_potential(positions, positions[1], 2), 0)
    positions[0] = ball
    self.assertEqual(puffer_env.closest_player_potential(positions, ball, 2), 0)
    self.assertEqual(
        puffer_env.closest_player_potential(np.full((11, 2), -1), ball, 2), 0)

  def test_both_teams_measure_distance_to_the_same_ball(self):
    """Opponent positions and ball share row zero's absolute coordinate frame."""
    env = object.__new__(puffer_env.FootballPufferEnv)
    env._ball_potential_scale = 0.0
    env._player_potential_scale = 1.0
    env._player_potential_fade_speed = 0.0
    raw = np.zeros((22, 115), dtype=np.float32)
    raw[0, :22] = raw[0, 44:66] = -1
    raw[0, :2] = (0.4, 0.1)
    raw[0, 44:46] = (0.9, 0.1)
    raw[0, 88:90] = (0.6, 0.1)
    env._attacking_left = True
    np.testing.assert_allclose(env._potentials(raw), (-0.2, -0.3), atol=1e-6)
    env._attacking_left = False
    np.testing.assert_allclose(env._potentials(raw), (-0.3, -0.2), atol=1e-6)
    raw[0, 88:90] = (0.5, 0.1)
    np.testing.assert_allclose(env._potentials(raw), (-0.4, -0.1), atol=1e-6)

  def test_player_potential_fades_while_the_ball_is_struck(self):
    """A struck ball carries no player potential; a settled loose ball does.

    The player term is weighted by how settled the ball is, read from the
    observed ball direction, so it stays a function of the state.  Kicking
    from the feet then costs nothing (the kicker was at distance zero and the
    flying ball weighs zero), while a ball left lying far from the team still
    lowers that team's potential until someone walks to it.  A fade speed of
    zero switches the weighting off, which is the old potential.
    """
    self.assertEqual(puffer_env.settled_ball_weight(0.3, 0.0), 1.0)
    self.assertEqual(puffer_env.settled_ball_weight(0.0, 0.5), 1.0)
    self.assertAlmostEqual(puffer_env.settled_ball_weight(0.25, 0.5), 0.5)
    self.assertEqual(puffer_env.settled_ball_weight(0.8, 0.5), 0.0)
    env = object.__new__(puffer_env.FootballPufferEnv)
    env._ball_potential_scale = 0.5
    env._player_potential_scale = 1.0
    env._player_potential_fade_speed = 0.5
    env._attacking_left = True
    raw = np.zeros((22, 115), dtype=np.float32)
    raw[0, :22] = raw[0, 44:66] = -1
    raw[0, :2] = (0.4, 0.1)
    raw[0, 44:46] = (0.9, 0.1)
    raw[0, 88:90] = (0.6, 0.1)
    ball = (0.5 * 0.6, -0.5 * 0.6)
    np.testing.assert_allclose(
        env._potentials(raw), (ball[0] - 0.2, ball[1] - 0.3), atol=1e-6)
    raw[0, 91] = 1.0  # struck along the pitch: the player term vanishes
    np.testing.assert_allclose(env._potentials(raw), ball, atol=1e-6)
    raw[0, 91] = 0.25  # half the fade speed: half the player term
    np.testing.assert_allclose(
        env._potentials(raw), (ball[0] - 0.1, ball[1] - 0.15), atol=1e-6)
    raw[0, 91] = 0.0
    raw[0, 92] = 0.25 * 54.4 / 83.6  # y is measured in x-distance units too
    np.testing.assert_allclose(
        env._potentials(raw), (ball[0] - 0.1, ball[1] - 0.15), atol=1e-6)

  def test_combined_potentials_telescope_across_real_episode_resets(self):
    """Both teams preserve score returns with combined or player-only shaping.

    Replay identical engine actions with shaping on/off through two complete
    episodes. Check the discounted return identity separately for each side,
    including terminal correction and potential initialisation after reset.
    Stacked observations must use their newest absolute frame, and the
    ball-speed fade of the player term must keep the identity exact.
    """
    gamma = 0.997
    for ball_scale, player_scale, stack, fade in ((1.0, 0.3, 1, 0.0),
                                                  (0.0, 1.0, 4, 0.0),
                                                  (0.585, 1.0, 1, 0.5)):
      common = dict(frame_stack=stack, seed=7)
      env = puffer_env.FootballPufferEnv(
          **common, ball_potential_scale=ball_scale,
          player_potential_scale=player_scale, potential_gamma=gamma,
          player_potential_fade_speed=fade)
      plain = puffer_env.FootballPufferEnv(**common)
      try:
        env.reset()
        plain.reset()
        for episode in range(2):
          initial = np.array([env._attack_potential, env._defence_potential])
          attacking_left = env._attacking_left
          discounted = np.zeros(2)
          for step in range(500):
            actions = np.full(22, 5, dtype=np.int32)
            _, rewards, terminals, _, infos = env.step(actions)
            _, scores, plain_terminals, _, plain_infos = plain.step(actions)
            np.testing.assert_array_equal(terminals, plain_terminals)
            difference = (rewards - scores).reshape(2, 11)
            np.testing.assert_allclose(
                difference, np.repeat(difference[:, :1], 11, axis=1), atol=1e-6)
            sides = difference[:, 0] if attacking_left else difference[::-1, 0]
            discounted += gamma ** step * sides
            if infos:
              for key in ('success', 'left_episode_return',
                          'right_episode_return', 'score_reward'):
                self.assertEqual(infos[0][key], plain_infos[0][key])
              break
          self.assertTrue(infos, 'test must reach a terminal state')
          np.testing.assert_allclose(discounted, -initial, atol=2e-6)
      finally:
        env.close()
        plain.close()

  def test_shaping_scales_reject_nonfinite_or_negative_values(self):
    """Invalid potentials must fail before constructing an engine instance."""
    for name in ('ball_potential_scale', 'player_potential_scale',
                 'player_potential_fade_speed'):
      for value in (-1, float('inf'), float('nan')):
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
          puffer_env.FootballPufferEnv(**{name: value})

  def test_ball_shaping_rewards_both_sides_and_leaves_success_alone(self):
    scale, gamma = 1.0, 0.99
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=7,
        ball_potential_scale=scale,
        potential_gamma=gamma)
    plain = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
    try:
      env.reset()
      plain.reset()
      attacking = slice(0, 11) if env._attacking_left else slice(11, 22)
      defending = slice(11, 22) if env._attacking_left else slice(0, 11)
      start_potential = env._attack_potential
      # The ball spawns in the defenders' half: positive for the attackers.
      self.assertGreater(start_potential, 0.0)
      self.assertAlmostEqual(env._defence_potential, -start_potential)
      # Sprint toward the goal with everyone: the ball carrier advances it.
      shaped_total = 0.0
      shaped_discounted = 0.0
      infos = []
      for step in range(400):
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
        shaped_discounted += gamma ** step * (attack - score)
        if infos:
          # The absorbing state has zero potential, so the last step refunds
          # each side exactly its own potential, whatever the final frame.
          self.assertAlmostEqual(attack - score, -previous_attack, places=5)
          self.assertAlmostEqual(defence + score, -previous_defence, places=5)
          break
        # ... and until then the two sides are shaped in exactly opposite
        # directions, so shaping is zero-sum like the score.
        self.assertAlmostEqual(
            (attack - score) + (defence + score), 0.0, places=5)
      self.assertTrue(infos)
      info = infos[0]
      # Success and the score returns are untouched by shaping.
      self.assertIn(info['success'], (0.0, 1.0))
      self.assertEqual(
          info['success'],
          float(info['left_episode_return' if env._attacking_left
                     else 'right_episode_return'] > 0))
      self.assertAlmostEqual(
          info['attacking_shaping_return'], shaped_total, places=4)
      # Telescoped: discounted, the episode's shaping is exactly
      # -Phi(s_0), whatever happened in between.
      self.assertAlmostEqual(shaped_discounted, -start_potential, places=5)
    finally:
      env.close()
      plain.close()

  def test_frozen_defence_reads_the_observation_it_was_trained_on(self):
    """A simple115 snapshot defends an entity-observation match.

    The fixed defender runs are scored against was trained on simple115,
    and every model must face the same opponent, so its rows are simple115
    whatever the learner reads: exactly the rows a simple115 match would
    give it.
    """
    torch.manual_seed(0)
    spaces = SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(115,), dtype=np.float32),
        single_action_space=gymnasium.spaces.Discrete(19))
    snapshot = self.create_tempfile('simple_defence.pt').full_path
    save_policy_snapshot(FootballPolicy(spaces, hidden_size=16), snapshot)
    matches = [puffer_env.FootballPufferEnv(
        frame_stack=1, seed=11,
        frozen_defence_path=snapshot,
        observation=observation) for observation in ('entities', 'simple115')]
    try:
      (entity_rows, _), (simple_rows, _) = [m.reset() for m in matches]
      self.assertEqual(entity_rows.shape[1], entity_observation.SIZE)
      defending = matches[0]._defending_rows()
      self.assertEqual(defending, matches[1]._defending_rows())
      np.testing.assert_array_equal(
          matches[0]._full_observations[defending],
          matches[1]._full_observations[defending])
      for _ in range(5):
        for match in matches:
          match.step(np.zeros(22, dtype=np.int32))
    finally:
      for match in matches:
        match.close()

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
          frame_stack=1, seed=7,
          frozen_defence_path=snapshot,
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
      self.assertIn(infos[0]['attacking_left'], (0.0, 1.0))
      # A new episode starts the frozen memory from zero again.
      self.assertIsNone(env._frozen_state['lstm_h'])
    finally:
      env.close()
      twin.close()


if __name__ == '__main__':
  absltest.main()
