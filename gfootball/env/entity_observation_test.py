"""Checks for the per-entity observation (context, ball, 22 player tokens)."""

from absl.testing import absltest
import numpy as np

from gfootball.env import entity_observation as entity
from gfootball.env import football_action_set


ACTIONS = [str(action) for action in
           football_action_set.action_set_dict['default']]


def _world():
  """A left-team view: own GK at -0.9, own 9 on the ball, own 10 offside."""
  own = np.stack([np.linspace(-0.9, 0.8, 11), np.linspace(-0.2, 0.2, 11)], 1)
  own[9] = (0.5, 0.1)
  own[10] = (0.8, -0.1)
  opponents = np.stack([np.linspace(0.95, -0.3, 11), np.zeros(11)], 1)
  # Deepest opponents: their keeper at 0.95, then 0.7, the offside line.
  opponents[1] = (0.6, 0.05)
  return {
      'left_team': own.astype(np.float32),
      'left_team_direction': np.full((11, 2), 0.01, np.float32),
      'left_team_roles': np.array([0] + [1] * 10),
      'left_team_tired_factor': np.linspace(0, 0.5, 11),
      'right_team': opponents.astype(np.float32),
      'right_team_direction': np.zeros((11, 2), np.float32),
      'right_team_roles': np.array([0] + [1] * 10),
      'right_team_tired_factor': np.zeros(11),
      'ball': np.array([0.5, 0.1, 0.2], np.float32),
      'ball_direction': np.array([0.02, 0.0, 0.0], np.float32),
      'ball_owned_team': 0,
      'ball_owned_player': 9,
      'game_mode': 0,
      'steps_left': 150,
  }


def _flip(view):
  """The same world seen by the right team (the engine's 180-degree turn)."""
  flipped = {
      'ball': view['ball'] * np.array([-1, -1, 1], np.float32),
      'ball_direction': view['ball_direction'] * np.array([-1, -1, 1],
                                                          np.float32),
      'ball_owned_team': (1 - view['ball_owned_team']
                          if view['ball_owned_team'] > -1 else -1),
      'ball_owned_player': view['ball_owned_player'],
      'game_mode': view['game_mode'],
      'steps_left': view['steps_left'],
  }
  for source, target in (('left', 'right'), ('right', 'left')):
    flipped[target + '_team'] = -view[source + '_team']
    flipped[target + '_team_direction'] = -view[source + '_team_direction']
    for suffix in ('_team_roles', '_team_tired_factor'):
      flipped[target + suffix] = view[source + suffix]
  return flipped


def _views(world):
  """Per-agent views as the engine returns them; -1 marks an absent agent."""
  right = _flip(world)
  right_players = len(right['left_team'])
  return ([dict(world, active=index) for index in range(11)] +
          [dict(right, active=index if index < right_players else -1)
           for index in range(11)])


def _players(row):
  start = entity.CONTEXT_FEATURES + entity.BALL_FEATURES
  return row[start:].reshape(entity.PLAYERS, entity.PLAYER_FEATURES)


class EntityObservationTest(absltest.TestCase):

  def test_layout_puts_the_agent_first_and_orders_the_rest_by_distance(self):
    observations = entity.build(_views(_world()),
                                np.zeros((22, entity.STICKY_ACTIONS)))
    self.assertEqual(observations.shape, (22, entity.SIZE))
    players = _players(observations[3])
    feature = entity.PLAYER_FEATURE_INDEX
    self.assertEqual(players[0, feature['is_self']], 1)
    self.assertEqual(players[1:, feature['is_self']].sum(), 0)
    np.testing.assert_allclose(players[0, feature['absolute_x']],
                               _world()['left_team'][3, 0], atol=1e-6)
    np.testing.assert_allclose(players[0, feature['offset_x']], 0, atol=1e-6)
    teammates = players[1:11, feature['distance_to_self']]
    opponents = players[11:, feature['distance_to_self']]
    self.assertTrue(np.all(np.diff(teammates) >= 0))
    self.assertTrue(np.all(np.diff(opponents) >= 0))
    self.assertEqual(players[:11, feature['teammate']].sum(), 11)
    self.assertEqual(players[11:, feature['teammate']].sum(), 0)
    self.assertEqual(players[:, feature['present']].sum(), 22)

  def test_flags_mark_keepers_the_ball_carrier_and_offside_attackers(self):
    observations = entity.build(_views(_world()),
                                np.zeros((22, entity.STICKY_ACTIONS)))
    feature = entity.PLAYER_FEATURE_INDEX
    # Agent 9 carries the ball; agent 10 is beyond the second-last defender
    # and the ball in the opponents' half.
    carrier = _players(observations[9])[0]
    offside = _players(observations[10])[0]
    keeper = _players(observations[0])[0]
    self.assertEqual(carrier[feature['has_ball']], 1)
    self.assertEqual(carrier[feature['offside']], 0)
    self.assertEqual(offside[feature['offside']], 1)
    self.assertEqual(offside[feature['has_ball']], 0)
    self.assertEqual(keeper[feature['goalkeeper']], 1)
    all_players = _players(observations[4])
    self.assertEqual(all_players[:, feature['has_ball']].sum(), 1)
    self.assertEqual(all_players[:, feature['goalkeeper']].sum(), 2)
    self.assertEqual(all_players[:, feature['offside']].sum(), 1)
    ball = observations[4, entity.CONTEXT_FEATURES:
                        entity.CONTEXT_FEATURES + entity.BALL_FEATURES]
    np.testing.assert_array_equal(
        ball[entity.BALL_FEATURE_INDEX['owner_none']:
             entity.BALL_FEATURE_INDEX['owner_opponent'] + 1], [0, 1, 0])

  def test_the_right_team_sees_the_same_world_turned_around(self):
    observations = entity.build(_views(_world()),
                                np.zeros((22, entity.STICKY_ACTIONS)))
    feature = entity.PLAYER_FEATURE_INDEX
    left_carrier = _players(observations[9])[0]
    right_view = _players(observations[11 + 3])
    carrier_as_opponent = right_view[
        right_view[:, feature['has_ball']] == 1][0]
    self.assertEqual(carrier_as_opponent[feature['teammate']], 0)
    np.testing.assert_allclose(
        carrier_as_opponent[[feature['absolute_x'], feature['absolute_y']]],
        -left_carrier[[feature['absolute_x'], feature['absolute_y']]],
        atol=1e-6)
    ball = observations[11 + 3, entity.CONTEXT_FEATURES:]
    np.testing.assert_array_equal(
        ball[entity.BALL_FEATURE_INDEX['owner_none']:
             entity.BALL_FEATURE_INDEX['owner_opponent'] + 1], [0, 0, 1])
    # The same player is offside only from the team it attacks for.
    self.assertEqual(right_view[:, feature['offside']].sum(), 1)

  def test_missing_players_are_marked_absent_and_zeroed(self):
    world = _world()
    world['right_team'] = world['right_team'][:8]
    world['right_team_direction'] = world['right_team_direction'][:8]
    world['right_team_roles'] = world['right_team_roles'][:8]
    world['right_team_tired_factor'] = world['right_team_tired_factor'][:8]
    observations = entity.build(_views(world),
                                np.zeros((22, entity.STICKY_ACTIONS)))
    players = _players(observations[2])
    present = players[:, entity.PLAYER_FEATURE_INDEX['present']]
    np.testing.assert_array_equal(present, [1] * 19 + [0] * 3)
    np.testing.assert_array_equal(players[19:], 0)

  def test_context_holds_sticky_buttons_game_mode_and_time_left(self):
    sticky = np.zeros((22, entity.STICKY_ACTIONS))
    sticky[5, 2] = sticky[5, 8] = 1
    world = _world()
    world['game_mode'] = 3
    observations = entity.build(_views(world), sticky)
    context = observations[5, :entity.CONTEXT_FEATURES]
    index = entity.CONTEXT_FEATURE_INDEX
    np.testing.assert_array_equal(
        context[index['sticky']:index['sticky'] + entity.STICKY_ACTIONS],
        sticky[5])
    game_mode = context[index['game_mode']:index['game_mode'] + 7]
    np.testing.assert_array_equal(game_mode, np.eye(7)[3])
    self.assertAlmostEqual(context[index['steps_left']],
                           150 / entity.STEPS_LEFT_SCALE)

  def test_sticky_buttons_follow_the_actions_pressed(self):
    sticky = np.zeros((1, entity.STICKY_ACTIONS), np.float32)
    sticky_names = [str(action) for action in
                    football_action_set.get_sticky_actions(
                        {'action_set': 'default'})]

    def press(name):
      entity.press_buttons(sticky, np.array([ACTIONS.index(name)]))
      return {sticky_names[i] for i in np.flatnonzero(sticky[0])}

    self.assertEqual(press('top_left'), {'top_left'})
    self.assertEqual(press('sprint'), {'top_left', 'sprint'})
    self.assertEqual(press('right'), {'right', 'sprint'})
    self.assertEqual(press('shot'), {'right', 'sprint'})
    self.assertEqual(press('dribble'), {'right', 'sprint', 'dribble'})
    self.assertEqual(press('release_direction'), {'sprint', 'dribble'})
    self.assertEqual(press('release_sprint'), {'dribble'})
    self.assertEqual(press('release_dribble'), set())
    self.assertEqual(press('idle'), set())


if __name__ == '__main__':
  absltest.main()
