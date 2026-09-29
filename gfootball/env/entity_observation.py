"""Per-entity observation for every agent: context, ball, and 22 players.

The default observation (simple115v2, made egocentric by puffer_env) gives
each player only a position and a velocity, relative to the agent.  It leaves
out what the engine knows and the policy has to guess: where the goals and
the touchlines are from the agent (an episode ends when the ball goes out),
who has the ball, which players are keepers, who is in
an offside position, how tired each player is, the buttons the agent is
still holding (direction, sprint and dribble are sticky), and how many steps
are left before the episode times out, which the value of every state
depends on (timeouts are stored as terminal).  This module builds all of it
from the engine's raw per-agent views, and is the one place that knows the
layout: puffer_env writes rows with `build`, the policy reads them with the
index tables below.

One agent's row (SIZE floats) is three blocks:
  context  CONTEXT_FEATURES   sticky buttons, game mode one-hot, steps left,
                              offsets to both goals and both touchlines
  ball     BALL_FEATURES      see BALL_FEATURE_NAMES
  players  PLAYERS x PLAYER_FEATURES, see PLAYER_FEATURE_NAMES: the agent
           itself first, then its teammates nearest first, then the
           opponents nearest first; absent players are all zero.
The frame is egocentric: nothing is an absolute pitch position.  Players and
ball are offsets from the agent, and the goals and touchlines are offsets
from it too, so the same situation anywhere on the pitch gives the same
tokens (better generalization) and only the landmark offsets say where it
is.  Offsets are divided by the pitch's length and width (PITCH_SIZE), so
they lie in [-1, 1].  The axes stay aligned with the pitch, +x toward the
goal the agent attacks (the engine turns the pitch for the right team):
the eight movement actions are pitch directions, so a frame that turned
with the player would no longer match what an action does.
"""

import numpy as np

from gfootball.env import football_action_set

TEAM = 11
PLAYERS = 2 * TEAM
GAME_MODES = 7
# Steps left is divided by this so it is O(1) at the curriculum's lengths.
STEPS_LEFT_SCALE = 100.0
# Pitch units per step: players move up to ~0.015, the ball ~0.1.
_PLAYER_VELOCITY_SCALE = 0.02
_BALL_VELOCITY_SCALE = 0.05
# Pitch length and width in observation units (x in [-1, 1], y in
# [-0.42, 0.42]); offsets are divided by these.
PITCH_SIZE = np.array([2.0, 0.84], dtype=np.float32)

_ACTION_NAMES = [str(action) for action in
                 football_action_set.action_set_dict['default']]
_STICKY_NAMES = [str(action) for action in
                 football_action_set.get_sticky_actions(
                     {'action_set': 'default'})]
STICKY_ACTIONS = len(_STICKY_NAMES)

CONTEXT_FEATURE_INDEX = {
    'sticky': 0,
    'game_mode': STICKY_ACTIONS,
    'steps_left': STICKY_ACTIONS + GAME_MODES,
    # x, y from the agent to the centre of the goal it attacks / defends.
    'attacked_goal': STICKY_ACTIONS + GAME_MODES + 1,
    'own_goal': STICKY_ACTIONS + GAME_MODES + 3,
    # y from the agent to the +y and to the -y touchline.
    'touchlines': STICKY_ACTIONS + GAME_MODES + 5,
}
CONTEXT_FEATURES = STICKY_ACTIONS + GAME_MODES + 7

BALL_FEATURE_NAMES = (
    'height', 'velocity_x', 'velocity_y', 'velocity_z',
    'offset_x', 'offset_y', 'distance_to_self',
    'owner_none', 'owner_team', 'owner_opponent')
BALL_FEATURE_INDEX = {name: i for i, name in enumerate(BALL_FEATURE_NAMES)}
BALL_FEATURES = len(BALL_FEATURE_NAMES)

PLAYER_FEATURE_NAMES = (
    'velocity_x', 'velocity_y', 'offset_x', 'offset_y',
    'ball_offset_x', 'ball_offset_y',
    'distance_to_self', 'distance_to_ball',
    'teammate', 'is_self', 'goalkeeper', 'has_ball', 'offside', 'tired',
    'present')
PLAYER_FEATURE_INDEX = {
    name: i for i, name in enumerate(PLAYER_FEATURE_NAMES)}
PLAYER_FEATURES = len(PLAYER_FEATURE_NAMES)

PLAYERS_START = CONTEXT_FEATURES + BALL_FEATURES
SIZE = PLAYERS_START + PLAYERS * PLAYER_FEATURES


def _button_tables():
  """Per action: which sticky buttons it keeps and which it presses.

  A direction replaces the held direction; release_direction drops it;
  sprint/dribble and their releases set or clear their own button; every
  other action leaves the buttons alone.
  """
  keep = np.ones((len(_ACTION_NAMES), STICKY_ACTIONS), dtype=bool)
  press = np.zeros((len(_ACTION_NAMES), STICKY_ACTIONS), dtype=bool)
  directions = [i for i, name in enumerate(_STICKY_NAMES)
                if name not in ('sprint', 'dribble')]
  for action, name in enumerate(_ACTION_NAMES):
    if name in _STICKY_NAMES:
      button = _STICKY_NAMES.index(name)
      if button in directions:
        keep[action, directions] = False
      press[action, button] = True
    elif name == 'release_direction':
      keep[action, directions] = False
    elif name in ('release_sprint', 'release_dribble'):
      keep[action, _STICKY_NAMES.index(name[len('release_'):])] = False
  return keep, press


_KEEP_BUTTONS, _PRESS_BUTTONS = _button_tables()


def press_buttons(sticky, actions):
  """Update each agent's held sticky buttons after it sends `actions`.

  The engine holds direction, sprint and dribble until they are replaced or
  released, and resets them all at every episode start (so does the env).
  Tracking them here from the actions sent is exact and avoids the ten
  engine queries per player per step of asking the engine.
  """
  sticky *= _KEEP_BUTTONS[actions]
  sticky[_PRESS_BUTTONS[actions]] = 1
  return sticky


def _team_block(view, team, length):
  """One team's arrays from a raw view, padded to TEAM rows."""
  positions = np.zeros((TEAM, 2), dtype=np.float32)
  velocities = np.zeros((TEAM, 2), dtype=np.float32)
  keepers = np.zeros(TEAM, dtype=np.float32)
  tired = np.zeros(TEAM, dtype=np.float32)
  positions[:length] = np.asarray(view[team + '_team']).reshape(-1, 2)
  velocities[:length] = np.asarray(
      view[team + '_team_direction']).reshape(-1, 2)
  keepers[:length] = np.asarray(view[team + '_team_roles']) == 0
  tired[:length] = view[team + '_team_tired_factor']
  return positions, velocities, keepers, tired


def _build_side(views, sticky, out):
  """Rows for the agents of one team, which all share one view of the pitch."""
  view = views[0]
  own_count = len(view['left_team'])
  opponent_count = len(view['right_team'])
  present = np.zeros(PLAYERS, dtype=bool)
  present[:own_count] = True
  present[TEAM:TEAM + opponent_count] = True
  blocks = [np.concatenate(parts) for parts in zip(
      _team_block(view, 'left', own_count),
      _team_block(view, 'right', opponent_count))]
  positions, velocities, keepers, tired = blocks
  ball = np.asarray(view['ball'], dtype=np.float32)
  owner_team = int(view['ball_owned_team'])
  has_ball = np.zeros(PLAYERS, dtype=np.float32)
  if owner_team >= 0:
    has_ball[owner_team * TEAM + int(view['ball_owned_player'])] = 1
  # Offside position: in the opponents' half, beyond the ball and beyond the
  # second-deepest opponent.  The agent's team attacks +x, the other -x.
  offside = np.zeros(PLAYERS, dtype=np.float32)
  own_x = positions[:own_count, 0]
  opponent_x = positions[TEAM:TEAM + opponent_count, 0]
  if opponent_count >= 2:
    line = max(np.sort(opponent_x)[-2], ball[0], 0.0)
    offside[:own_count] = own_x > line
  if own_count >= 2:
    line = min(np.sort(own_x)[1], ball[0], 0.0)
    offside[TEAM:TEAM + opponent_count] = opponent_x < line

  index = PLAYER_FEATURE_INDEX
  shared = np.zeros((PLAYERS, PLAYER_FEATURES), dtype=np.float32)
  shared[:, index['velocity_x']:index['velocity_y'] + 1] = (
      velocities / _PLAYER_VELOCITY_SCALE)
  ball_offset = (positions - ball[:2]) / PITCH_SIZE
  shared[:, index['ball_offset_x']:index['ball_offset_y'] + 1] = ball_offset
  shared[:, index['distance_to_ball']] = np.linalg.norm(ball_offset, axis=1)
  shared[:TEAM, index['teammate']] = 1
  shared[:, index['goalkeeper']] = keepers
  shared[:, index['has_ball']] = has_ball
  shared[:, index['offside']] = offside
  shared[:, index['tired']] = tired
  shared[:, index['present']] = 1
  shared[~present] = 0

  active = np.array([agent_view['active'] for agent_view in views])
  agents = np.flatnonzero(active >= 0)
  if not agents.size:
    return
  self_positions = positions[active[agents]]
  offsets = (positions[None] - self_positions[:, None]) / PITCH_SIZE
  distances = np.linalg.norm(offsets, axis=-1)
  players = np.repeat(shared[None], agents.size, axis=0)
  players[:, :, index['offset_x']:index['offset_y'] + 1] = offsets
  players[:, :, index['distance_to_self']] = distances
  rows = np.arange(agents.size)
  players[rows, active[agents], index['is_self']] = 1
  players[:, ~present] = 0
  # The agent first, then each team nearest first, absent players last.
  order_key = np.where(present, distances, np.inf)
  order_key[rows, active[agents]] = -1
  order = np.concatenate((
      np.argsort(order_key[:, :TEAM], axis=1, kind='stable'),
      TEAM + np.argsort(order_key[:, TEAM:], axis=1, kind='stable')), axis=1)
  out[agents, PLAYERS_START:] = players[rows[:, None], order].reshape(
      agents.size, -1)

  ball_row = out[agents, CONTEXT_FEATURES:PLAYERS_START]
  ball_index = BALL_FEATURE_INDEX
  ball_row[:, ball_index['height']] = ball[2]
  ball_row[:, ball_index['velocity_x']:ball_index['velocity_z'] + 1] = (
      np.asarray(view['ball_direction'], dtype=np.float32) /
      _BALL_VELOCITY_SCALE)
  ball_offset_from_self = (ball[:2] - self_positions) / PITCH_SIZE
  ball_row[:, ball_index['offset_x']:ball_index['offset_y'] + 1] = (
      ball_offset_from_self)
  ball_row[:, ball_index['distance_to_self']] = np.linalg.norm(
      ball_offset_from_self, axis=1)
  ball_row[:, ball_index['owner_none'] + owner_team + 1] = 1
  out[agents, CONTEXT_FEATURES:PLAYERS_START] = ball_row

  context = CONTEXT_FEATURE_INDEX
  out[agents, context['sticky']:context['sticky'] + STICKY_ACTIONS] = (
      sticky[agents])
  out[agents, context['game_mode'] + int(view['game_mode'])] = 1
  out[agents, context['steps_left']] = view['steps_left'] / STEPS_LEFT_SCALE
  for name, landmark in (('attacked_goal', (1.0, 0.0)),
                         ('own_goal', (-1.0, 0.0))):
    out[agents, context[name]:context[name] + 2] = (
        (np.asarray(landmark, dtype=np.float32) - self_positions) / PITCH_SIZE)
  half_width = PITCH_SIZE[1] / 2
  out[agents, context['touchlines']] = (
      (half_width - self_positions[:, 1]) / PITCH_SIZE[1])
  out[agents, context['touchlines'] + 1] = (
      (-half_width - self_positions[:, 1]) / PITCH_SIZE[1])


def build(views, sticky, out=None):
  """Entity rows for the 22 agents of one match.

  `views` are the engine's raw per-agent observations in agent order (11
  left, then 11 right, each team already turned to attack +x; 'active' is
  the controlled player, -1 for an agent without one), `sticky` the (22,
  STICKY_ACTIONS) buttons each agent holds (press_buttons).  Writes into
  `out` (22, SIZE) when given; agents without a player get all-zero rows.
  """
  if len(views) != PLAYERS:
    raise ValueError('expected 22 agent views, got {}'.format(len(views)))
  if out is None:
    out = np.zeros((PLAYERS, SIZE), dtype=np.float32)
  else:
    out.fill(0)
  for side in (slice(0, TEAM), slice(TEAM, PLAYERS)):
    _build_side(views[side], np.asarray(sticky)[side], out[side])
  return out
