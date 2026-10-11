# coding=utf-8
"""11v11 self-play scenes built from ADR settings (gfootball/adr.py).

Every episode reads its settings from the config value adr.VALUES (one
value per ADR setting; the start values when it is absent): the ball sits
`ball_distance` from the attacked goal line at y `ball_lateral`; the two
carriers stand squarely behind it; the other attackers ring the ball and the
defence holds its formation, both blended toward real match formations by
`formation`; then `blockers` goal-side blockers, `attackers_ahead` attackers
between the ball and the goal, the race for the ball (`fetch_distance`,
`defender_head_start`) and `held_share` of the outfield players holding a
direction.  Offside is on once the formation is a real match shape, with
every attacker placed onside.  The config value adr.HELD_DIRECTIONS lists
the held directions per agent (adr.read_scene); puffer_env presses them on
the first step, outside training.

Every episode ends as soon as the ball goes out of play.  With the magnet off
the engine never designates a set-piece taker, so a goal kick, corner or
throw-in would otherwise freeze the match until the timeout.
"""

import math
import random

from . import *
from gfootball import adr


_FORMATION = (
    (-1.0, 0.0, e_PlayerRole_GK),
    (0.0, 0.02, e_PlayerRole_RM),
    (0.0, -0.02, e_PlayerRole_CF),
    (-0.422, -0.19576, e_PlayerRole_LB),
    (-0.5, -0.06356, e_PlayerRole_CB),
    (-0.5, 0.063559, e_PlayerRole_CB),
    (-0.422, 0.19576, e_PlayerRole_RB),
    (-0.184212, -0.10568, e_PlayerRole_CM),
    (-0.267574, 0.0, e_PlayerRole_CM),
    (-0.184212, 0.10568, e_PlayerRole_CM),
    (-0.01, -0.21610, e_PlayerRole_LM),
)

_RING_RADII = (0.10, 0.19)
_SLOTS_PER_RING = 5
_PITCH_X = 0.94
_PITCH_Y = 0.36
_MIN_GAP = 0.025


def _ring_spawn(rank, ball_x, ball_y, direction, rng):
  """Spread the attackers other than the carriers evenly around the ball.

  With the ball parked near the goal line there is not enough room for a full
  circle, so the goal-side half of the ring is *compressed* rather than
  clipped.  Clipping drove several players onto the identical clamped
  coordinate; compression keeps every angle at a distinct position.
  """
  ring, slot = divmod(rank, _SLOTS_PER_RING)
  ring = min(ring, len(_RING_RADII) - 1)
  angle = 2 * math.pi * slot / _SLOTS_PER_RING
  radius = _RING_RADII[ring]
  offset_x = radius * math.cos(angle)
  offset_y = radius * math.sin(angle)
  # Scale by the OUTER radius, not each ring's own: compressing per-ring maps
  # both rings onto the same forward offset and stacks them on each other.
  outer = max(_RING_RADII)
  forward_room = max(0.05, _PITCH_X - abs(ball_x))
  if direction * offset_x > 0 and outer > forward_room:
    offset_x *= forward_room / outer
  side_room = max(0.08, _PITCH_Y - abs(ball_y))
  if outer > side_room:
    offset_y *= side_room / outer
  return (ball_x + offset_x, ball_y + offset_y + rng.uniform(-0.004, 0.004))


def _to_team_coordinates(team, x, y):
  side = 1.0 if team == Team.e_Left else -1.0
  return side * x, side * y


def _separate(points, min_gap=_MIN_GAP, iterations=12):
  """Push apart any pair closer than one player-width.

  Interpolating between the contested ring and the standard formation makes
  player paths cross, so some blends put two players on nearly the same spot.
  A few relaxation passes guarantee a legal, physical starting arrangement.
  """
  points = [list(point) for point in points]
  for _ in range(iterations):
    moved = False
    for i in range(len(points)):
      for j in range(i + 1, len(points)):
        dx = points[j][0] - points[i][0]
        dy = points[j][1] - points[i][1]
        distance = math.hypot(dx, dy)
        if distance >= min_gap:
          continue
        if distance < 1e-6:
          # Exactly coincident: nudge along x. Using min_gap here would make
          # the push below zero and leave them stacked.
          dx, dy, distance = 1e-4, 0.0, 1e-4
        push = 0.5 * (min_gap - distance)
        unit_x, unit_y = dx / distance, dy / distance
        points[i][0] -= unit_x * push
        points[i][1] -= unit_y * push
        points[j][0] += unit_x * push
        points[j][1] += unit_y * push
        moved = True
    if not moved:
      break
  return [(max(-_PITCH_X, min(_PITCH_X, x)),
           max(-_PITCH_Y, min(_PITCH_Y, y))) for x, y in points]


def _carrier_spawn(rank, ball_x, ball_y, direction, rng):
  """Two attackers squarely behind the ball, lined up to shoot.

  Without this the ring leaves whoever wins the ball facing an arbitrary
  direction, and some attackers spawn between the ball and the goal blocking
  their own shot -- measured as always-shoot scoring 0.000.  A shot has to
  be *available* for a scene to teach anything.
  """
  gap = 0.03 + 0.03 * rank
  return (ball_x - direction * gap,
          ball_y + rng.uniform(-0.004, 0.004))


# Lateral lane each successive blocker takes, as a multiple of _LANE_WIDTH.
# The first blocker sits off to one side; which side is a per-episode coin
# flip (see _blocker_spawn).  The second MIRRORS it on the other side, so the
# pair is symmetric and the shot line between them stays open; only the third
# closes the middle.  Before this the second blocker landed on the shot line.
_BLOCKER_LANES = (-1, 1, 0)
_LANE_WIDTH = 0.085


def _blocker_spawn(rank, ball_x, ball_y, direction, side, rng):
  """A defensive line between the ball and the goal it protects.

  `side` (+1 or -1) mirrors the lanes.  With a fixed side a lone blocker
  always stood on the world -y side of the ball, so it landed inside the
  goal mouth only for balls on the +y side, so one side of the pitch was
  much harder than the other.
  """
  row, slot = divmod(rank, len(_BLOCKER_LANES))
  depth = 0.07 + 0.04 * row
  offset = side * _BLOCKER_LANES[slot] * _LANE_WIDTH
  forward_room = max(0.04, _PITCH_X - abs(ball_x))
  depth = min(depth, forward_room)
  return (ball_x + direction * depth,
          ball_y + offset + rng.uniform(-0.004, 0.004))


# attackers_ahead: up to this many attackers start between the ball and the
# goal, onside behind a defensive line that sits _AHEAD_LINE_DEPTH ahead of
# the ball when the pitch leaves room for it.
_MAX_AHEAD = 3
_AHEAD_LINE_DEPTH = (0.10, 0.25)
_AHEAD_LINE_DEFENDERS = 3
# Goal mouth half-width; the shooting lane narrows from it to the ball, and
# an attacker placed ahead keeps this much clear of the lane.
_POST_Y = 0.044
_LANE_CLEARANCE = 0.03
# Default action set: 1-8 are the eight movement directions.
_DIRECTION_ACTIONS = tuple(range(1, 9))
# The attacking side switches every this many episodes (seed + episode), so
# both attack directions are played evenly.
_DIRECTION_PERIOD = 8


def _lane_gap(x, y, ball_x, ball_y, direction):
  """How far (x, y) is outside the shooting lane from the ball to the goal.

  Negative inside the lane.  In world coordinates; `direction` is +1 when
  the goal being attacked is at x = +1.
  """
  along = (direction * x - direction * ball_x) / (1.0 - direction * ball_x)
  centre = ball_y * (1.0 - along)
  return abs(y - centre) - (_POST_Y * along + _LANE_CLEARANCE)


def _lane_free_y(x, ball_x, ball_y, direction, rng):
  """A y across the width, outside the shooting lane at that x."""
  for _ in range(100):
    y = rng.uniform(-_PITCH_Y, _PITCH_Y)
    if _lane_gap(x, y, ball_x, ball_y, direction) > 0:
      return y
  return _PITCH_Y if ball_y < 0 else -_PITCH_Y


def _place_attackers_ahead(positions, roles, ball_x, ball_y, direction, count, rng):
  """Put some attackers between the ball and the goal, onside, lane clear.

  `positions` are world [x, y] lists for the outfield players and `roles`
  says, per player, 'attacker', 'blocker' or 'defender'.  If the defensive
  line (the second-last defender, keeper included) sits less than a drawn
  depth ahead of the ball, the defenders nearest the ball step up to that
  line; then min(count, _MAX_AHEAD) attackers take spots between the
  ball and the line, outside the shooting lane.  Returns those attackers'
  indices, which the final pass keeps ahead, onside and lane clear.
  """
  forward = lambda x: direction * x - direction * ball_x
  room = 0.94 - direction * ball_x
  defenders = [i for i, role in enumerate(roles) if role != 'attacker']
  depths = sorted([forward(positions[i][0]) for i in defenders] +
                  [forward(direction * 1.0)])
  line = depths[-2]
  target = min(rng.uniform(*_AHEAD_LINE_DEPTH), room - 0.01)
  if line < target:
    movable = sorted((i for i in defenders if roles[i] == 'defender'),
                     key=lambda i: math.hypot(positions[i][0] - ball_x,
                                              positions[i][1] - ball_y))
    for i in movable[:_AHEAD_LINE_DEFENDERS]:
      positions[i][0] = ball_x + direction * target
      positions[i][1] = rng.uniform(-_PITCH_Y, _PITCH_Y)
    line = target
  attackers = [i for i, role in enumerate(roles) if role == 'attacker']
  # The two carriers stay behind the ball; ring players move up.
  ahead = rng.sample(attackers[2:], min(count, _MAX_AHEAD))
  for i in ahead:
    depth = rng.uniform(0.035, max(0.04, line - 0.012))
    positions[i][0] = ball_x + direction * depth
    positions[i][1] = _lane_free_y(positions[i][0], ball_x, ball_y,
                                   direction, rng)
  return ahead


def _keep_attackers_ahead(spaced, roles, ahead, ball_x, ball_y, direction,
                      lane=True):
  """Undo what the overlap relaxation did to the attackers placed ahead.

  Every attacker ends up onside and, with `lane`, every attacker ahead of the
  ball clear of the shooting lane; the chosen ones stay at least 0.025
  ahead.  A race for a still ball passes lane=False: nobody is at the ball to
  shoot, and sliding an attacker onto the lane edge would put him beside it.
  """
  forward = lambda x: direction * x - direction * ball_x
  depths = sorted([forward(spaced[i][0]) for i, role in enumerate(roles)
                   if role != 'attacker'] + [forward(direction * 1.0)])
  line = max(0.0, depths[-2])
  for i, role in enumerate(roles):
    if role != 'attacker':
      continue
    x, y = spaced[i]
    depth = forward(x)
    if i in ahead:
      depth = max(depth, 0.025)
    depth = min(depth, line - 0.002) if depth > 0 else depth
    x = ball_x + direction * depth
    if lane and depth > 0.02 and _lane_gap(x, y, ball_x, ball_y,
                                           direction) <= 0:
      along = depth / (1.0 - direction * ball_x)
      centre = ball_y * (1.0 - along)
      side = 1.0 if y >= centre else -1.0
      y = centre + side * (_POST_Y * along + _LANE_CLEARANCE + 0.001)
      if abs(y) > _PITCH_Y:
        y = centre - side * (_POST_Y * along + _LANE_CLEARANCE + 0.001)
    spaced[i] = (x, y)


def _push_out(point, ball_x, ball_y, radius, rng):
  """Move `point` to `radius` from the ball, along its own bearing if it can.

  Bearings that would leave the pitch are retried at random, so a player
  pushed off the ball near a goal line or touchline still lands in play.
  """
  dx, dy = point[0] - ball_x, point[1] - ball_y
  angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else (
      rng.uniform(-math.pi, math.pi))
  for attempt in range(40):
    x = ball_x + radius * math.cos(angle)
    y = ball_y + radius * math.sin(angle)
    if abs(x) <= _PITCH_X and abs(y) <= _PITCH_Y:
      return [x, y]
    angle = rng.uniform(-math.pi, math.pi)
  return [max(-_PITCH_X, min(_PITCH_X, x)), max(-_PITCH_Y, min(_PITCH_Y, y))]


def _chance_count(value, rng):
  """The whole part of `value`, plus one more with its fractional chance."""
  whole = int(math.floor(value + 1e-9))
  return whole + (1 if rng.random() < value - whole else 0)


def _race_for_the_ball(positions, roles, ball_x, ball_y, fetch, head_start,
                       rng):
  """Nearest attacker exactly `fetch` from the ball, nearest outfield
  defender exactly `fetch + head_start` (at least 0.03), everyone else of
  each team at least that far.  The keepers are not moved."""
  distance = lambda i: math.hypot(positions[i][0] - ball_x,
                                  positions[i][1] - ball_y)
  defender_reach = max(0.03, fetch + head_start)
  for team, reach in (('attacker', fetch), ('defender', defender_reach)):
    members = [i for i, role in enumerate(roles)
               if (role == 'attacker') == (team == 'attacker')]
    for i in members:
      if distance(i) < reach:
        positions[i] = _push_out(positions[i], ball_x, ball_y,
                                 reach + rng.uniform(0.0, 0.05), rng)
    nearest = min(members, key=distance)
    positions[nearest] = _push_out(positions[nearest], ball_x, ball_y, reach,
                                   rng)


def build_scenario(builder):
  episode = builder.EpisodeNumber()
  values = builder._config._values
  seed = int(values.get('game_engine_random_seed', 0))
  settings = values.get(adr.VALUES) or {
      parameter.name: parameter.start for parameter in adr.PARAMETERS}
  rng = random.Random('{}:{}:adr'.format(seed, episode))
  attack_right = ((seed + episode) // _DIRECTION_PERIOD) % 2 == 0
  builder._config['reverse_team_processing'] = not attack_right
  direction = 1.0 if attack_right else -1.0
  distance = float(settings['ball_distance'])
  ball_x = direction * (1.0 - distance)
  ball_y = float(settings['ball_lateral'])
  builder.SetBallPosition(ball_x, ball_y)
  formation = max(0.0, min(1.0, float(settings['formation'])))
  progress = max(0.0, min(1.0, (distance - 0.1) / 0.9))
  builder.config().game_duration = int(round(300 + 400 * progress))
  builder.config().deterministic = False
  builder.config().use_magnet = False
  offsides = formation >= 0.9
  builder.config().offsides = offsides
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True

  attacking_team = Team.e_Left if attack_right else Team.e_Right
  goalside = max(0, min(10, _chance_count(float(settings['blockers']), rng)))
  values['goalside_blockers'] = goalside
  blocker_side = 1.0 if rng.random() < 0.5 else -1.0
  outfield = []
  for team in (Team.e_Left, Team.e_Right):
    attacking = team == attacking_team
    for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
      if index == 0:
        continue
      rank = index - 1
      standard = _to_team_coordinates(team, standard_x, standard_y)
      if attacking and rank < 2:
        position = _carrier_spawn(rank, ball_x, ball_y, direction, rng)
      elif rank < goalside and not attacking:
        position = _blocker_spawn(rank, ball_x, ball_y, direction,
                                  blocker_side, rng)
      else:
        contested = (_ring_spawn(rank - 2, ball_x, ball_y, direction, rng)
                     if attacking else standard)
        position = ((1 - formation) * contested[0] + formation * standard[0],
                    (1 - formation) * contested[1] + formation * standard[1])
      outfield.append((team, index, role, position[0], position[1]))

  positions = [[row[3], row[4]] for row in outfield]
  roles = ['attacker' if row[0] == attacking_team else
           'blocker' if row[1] - 1 < goalside else 'defender'
           for row in outfield]
  ahead_count = min(_MAX_AHEAD,
                    _chance_count(float(settings['attackers_ahead']), rng))
  ahead = (_place_attackers_ahead(positions, roles, ball_x, ball_y, direction,
                        ahead_count, rng) if ahead_count else [])
  _race_for_the_ball(positions, roles, ball_x, ball_y,
                     float(settings['fetch_distance']),
                     float(settings['defender_head_start']), rng)
  held = [0] * 22
  held_share = float(settings['held_share'])
  for row in outfield:
    if rng.random() < held_share:
      held[(0 if row[0] == Team.e_Left else 11) + row[1]] = rng.choice(
          _DIRECTION_ACTIONS)
  racing = float(settings['fetch_distance']) > 0.035
  values[adr.LOOSE_BALL] = racing
  values[adr.HELD_DIRECTIONS] = held
  spaced = _separate([tuple(p) for p in positions])
  if offsides or (ahead and not racing):
    # In a race only offside matters (the race already placed everyone).
    _keep_attackers_ahead(spaced, roles, [] if racing else ahead, ball_x, ball_y,
                      direction, lane=not racing)
  for team in (Team.e_Left, Team.e_Right):
    builder.SetTeam(team)
    for index, (_, _, role) in enumerate(_FORMATION):
      if index == 0:
        builder.AddPlayer(-1.0, 0.0, role)
        continue
      position = next(spaced[k] for k, row in enumerate(outfield)
                      if row[0] == team and row[1] == index)
      builder.AddPlayer(*_to_team_coordinates(team, *position), role)
