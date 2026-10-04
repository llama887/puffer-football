# coding=utf-8
"""11v11 self-play where both sides start contesting the ball.

Every one of the 22 players is active and playing normally at every level.
Both teams are spread evenly around the ball, interleaved, so neither side is
handed a positional edge -- the only thing a level controls is how deep in the
opponent's half the ball starts:

  advantage 1.0  ball on the opponent's goal line, both teams packed around
                 it, but only ONE defender standing between the ball and the
                 goal.  A shot is available, so there is something to learn
                 from; the rest of the defence is right there to contest it.
  advantage 0.0  ordinary kickoff, full defensive line, standard formations.

  Measured: ringing all ten defenders around the ball smothers it completely
  (always-shoot scored 0.000 and 90% of episodes had no goal at all), which
  leaves nothing to bootstrap from.  Ramping how many defenders actually block
  the goal keeps contest present from level 0 while restoring a shot.
  advantage 0.0  ordinary kickoff, standard formations both sides.

Contrast with a positional handicap: deferring all opposition to the later
levels teaches nothing but "shoot at an open goal", which is precisely what
the previous schedule produced.

With the config value 'uniform_spawn' set, levels are ignored instead: the
ball and every outfield player are placed uniformly over the pitch, each
keeper uniformly inside its own penalty area, and anyone who would start
offside is moved back onside.  That is the no-curriculum alternative: some
spawns land players next to the ball near a goal by chance, so there is
always something scoreable without maintaining a level schedule.

With 'width_spawn_fraction' set, that fraction of TRAINING episodes (never the
gate's evaluation episodes) is a width spawn: the level's scene is laid out as
usual, then every outfield player except the goal-side blockers takes a y
drawn uniformly across the pitch width while keeping the x the level gives it.
The ball, the blockers and the keepers stay where the level puts them, so the
level still sets the distance from goal and the defensive line, but nobody
starts lined up on the ball: whoever is nearest has to go and get it.  The
draw has its own generator, so the normal scenes are unchanged by it.

With 'mixed_shape' set, every level from 1 up (gate scenes included, since it
is part of the level) puts min(level, 3) attackers between the ball and the
goal, onside behind a defensive line that steps up when there is room, and
clear of the shooting lane.  Otherwise no attacker starts ahead of the ball
from level 3 on, so running forward is right from every spawn.

With 'loose_ball_fraction' set, that fraction of TRAINING episodes is a
loose-ball spawn built to look like a real dead ball: some players start on
the touchlines, the nearest attacker starts a fetch distance from the still
ball that grows with the level (0.05-0.15 at level 0, up to 0.9 by level
10), every defender starts farther still on the low levels (the head start
fades to an even race by level 10), and about half the outfield players are
given a direction to hold.  'curriculum_held_directions' lists those
directions per agent; puffer_env presses them on the first step, outside
training.

Every episode ends as soon as the ball goes out of play.  With the magnet off
the engine never designates a set-piece taker, so a goal kick, corner or
throw-in would otherwise freeze the match until the timeout.
"""

import math
import random

from . import *
from gfootball.curriculum import (
    ADVANTAGE_LEVELS, SPAWN_TEMPLATE_COUNT, goalside_blockers,
    lateral_half_width)


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

_BALL_DEPTH = 0.90
_RING_RADII = (0.10, 0.19)
_SLOTS_PER_RING = 5
_PITCH_X = 0.94
_PITCH_Y = 0.36
_MIN_GAP = 0.025


def _ring_spawn(rank, team_phase, ball_x, ball_y, direction, rng):
  """Spread a team's ten outfield players evenly around the ball.

  The two teams use a half-slot phase offset so they interleave rather than
  occupying separate arcs; neither side ends up systematically goal-side.

  With the ball parked near the goal line there is not enough room for a full
  circle, so the goal-side half of the ring is *compressed* rather than
  clipped.  Clipping drove several players onto the identical clamped
  coordinate; compression keeps every angle at a distinct position.
  """
  ring, slot = divmod(rank, _SLOTS_PER_RING)
  ring = min(ring, len(_RING_RADII) - 1)
  angle = 2 * math.pi * (slot + team_phase) / _SLOTS_PER_RING
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
  their own shot -- measured as always-shoot scoring 0.000 at every level.
  A shot has to be *available* for a level to teach anything.
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

# Uniform spawn: a keeper's own penalty area, as distance in from its goal
# line and lateral half-width, in pitch units.
_KEEPER_BOX_DEPTH = 0.15
_KEEPER_BOX_HALF_WIDTH = 0.2
_UNIFORM_DURATION = 400


def _blocker_spawn(rank, ball_x, ball_y, direction, side, rng):
  """A defensive line between the ball and the goal it protects.

  `side` (+1 or -1) mirrors the lanes.  With a fixed side a lone blocker
  always stood on the world -y side of the ball, so it landed inside the
  goal mouth only for balls on the +y side: one template of each mirror pair
  was much harder than the other.
  """
  row, slot = divmod(rank, len(_BLOCKER_LANES))
  depth = 0.07 + 0.04 * row
  offset = side * _BLOCKER_LANES[slot] * _LANE_WIDTH
  forward_room = max(0.04, _PITCH_X - abs(ball_x))
  depth = min(depth, forward_room)
  return (ball_x + direction * depth,
          ball_y + offset + rng.uniform(-0.004, 0.004))


def _onside(attackers, defenders, ball_x, sign, rng):
  """Move every attacker that would start offside back to an onside spot.

  `sign` is +1 for the team attacking +x.  In attacking-direction units an
  attacker is offside when it is in the opponents' half and nearer their goal
  line than both the ball and the second-last opponent; such an attacker gets
  a new x drawn uniformly between its own end and that line.  Moving
  attackers back only ever loosens the other team's line, so one pass per
  team leaves both teams legal.
  """
  second_last = sorted((sign * x for x, _ in defenders), reverse=True)[1]
  line = max(0.0, sign * ball_x, second_last)
  return [(x, y) if sign * x <= line else
          (sign * rng.uniform(-_PITCH_X, line), y) for x, y in attackers]


# Mixed shape: from level 1, up to this many attackers start between the ball
# and the goal (one more per level), onside behind a defensive line that sits
# _MIXED_LINE_DEPTH ahead of the ball when the pitch leaves room for it.
_MIXED_MAX_AHEAD = 3
_MIXED_LINE_DEPTH = (0.10, 0.25)
_MIXED_LINE_DEFENDERS = 3
# Goal mouth half-width; the shooting lane narrows from it to the ball, and
# an attacker placed ahead keeps this much clear of the lane.
_POST_Y = 0.044
_LANE_CLEARANCE = 0.03
# Loose-ball spawn.  Measured dead balls (9/30 traces) have the nearest
# attacker 0.4-0.9 from a still ball, every player holding a direction and
# the defence usually farther away.  The fetch distance and the defenders'
# head start both follow level progress, reaching the measured range and an
# even race at _LOOSE_RAMP_LEVELS.
_LOOSE_RAMP_LEVELS = 10
_LOOSE_FETCH = (0.05, 0.15, 0.90)  # minimum, maximum at level 0, at the ramp
_LOOSE_DEFENDER_MARGIN = 0.15      # at level 0, fading to 0
_LOOSE_TOUCHLINE_SHARE = 0.3
_LOOSE_TOUCHLINE_Y = (0.28, 0.36)
_LOOSE_HELD_SHARE = 0.5
# Default action set: 1-8 are the eight movement directions.
_DIRECTION_ACTIONS = tuple(range(1, 9))


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


def _mixed_shape(positions, roles, ball_x, ball_y, direction, level, rng):
  """Put some attackers between the ball and the goal, onside, lane clear.

  `positions` are world [x, y] lists for the outfield players and `roles`
  says, per player, 'attacker', 'blocker' or 'defender'.  If the defensive
  line (the second-last defender, keeper included) sits less than a drawn
  depth ahead of the ball, the defenders nearest the ball step up to that
  line; then min(level, _MIXED_MAX_AHEAD) attackers take spots between the
  ball and the line, outside the shooting lane.  Returns those attackers'
  indices, which the final pass keeps ahead, onside and lane clear.
  """
  forward = lambda x: direction * x - direction * ball_x
  room = 0.94 - direction * ball_x
  defenders = [i for i, role in enumerate(roles) if role != 'attacker']
  depths = sorted([forward(positions[i][0]) for i in defenders] +
                  [forward(direction * 1.0)])
  line = depths[-2]
  target = min(rng.uniform(*_MIXED_LINE_DEPTH), room - 0.01)
  if line < target:
    movable = sorted((i for i in defenders if roles[i] == 'defender'),
                     key=lambda i: math.hypot(positions[i][0] - ball_x,
                                              positions[i][1] - ball_y))
    for i in movable[:_MIXED_LINE_DEFENDERS]:
      positions[i][0] = ball_x + direction * target
      positions[i][1] = rng.uniform(-_PITCH_Y, _PITCH_Y)
    line = target
  attackers = [i for i, role in enumerate(roles) if role == 'attacker']
  # The two carriers stay behind the ball; ring players move up.
  ahead = rng.sample(attackers[2:], min(level, _MIXED_MAX_AHEAD))
  for i in ahead:
    depth = rng.uniform(0.035, max(0.04, line - 0.012))
    positions[i][0] = ball_x + direction * depth
    positions[i][1] = _lane_free_y(positions[i][0], ball_x, ball_y,
                                   direction, rng)
  return ahead


def _keep_mixed_shape(spaced, roles, ahead, ball_x, ball_y, direction,
                      lane=True):
  """Undo what the overlap relaxation did to the mixed shape.

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


def _loose_ball(positions, roles, ball_x, ball_y, direction, level, rng):
  """Make the opening a race for a still ball, like a real dead ball.

  Some players are sent to the touchlines, then every attacker is moved out
  to at least the fetch distance and one of them exactly to it, and every
  outfield defender to at least the fetch distance plus the level's head
  start.  The keeper cannot leave the goal, so on the levels where it is near
  the ball the fetch distance is capped below its distance.
  """
  progress = min(1.0, level / float(_LOOSE_RAMP_LEVELS))
  low, high0, high1 = _LOOSE_FETCH
  keeper = math.hypot(direction * 1.0 - ball_x, ball_y)
  high = min(high0 + (high1 - high0) * progress, keeper - 0.02)
  fetch = rng.uniform(low, max(low, high))
  margin = _LOOSE_DEFENDER_MARGIN * (1.0 - progress)
  for i in range(len(positions)):
    if rng.random() < _LOOSE_TOUCHLINE_SHARE:
      positions[i][1] = rng.choice((-1.0, 1.0)) * rng.uniform(
          *_LOOSE_TOUCHLINE_Y)
  distance = lambda i: math.hypot(positions[i][0] - ball_x,
                                  positions[i][1] - ball_y)
  for i, role in enumerate(roles):
    floor = fetch if role == 'attacker' else fetch + margin
    if distance(i) < floor:
      positions[i] = _push_out(positions[i], ball_x, ball_y,
                               floor + rng.uniform(0.0, 0.05), rng)
  attackers = [i for i, role in enumerate(roles) if role == 'attacker']
  nearest = min(attackers, key=distance)
  positions[nearest] = _push_out(positions[nearest], ball_x, ball_y, fetch, rng)
  return fetch


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


def _build_adr(builder, values, seed, episode, cycle, settings):
  """A scene from explicit ADR settings (gfootball/adr.py) instead of a level.

  The ball sits `ball_distance` from the attacked goal line at y
  `ball_lateral`; the attackers start packed around it and the defence in
  its formation, blended toward real match formations by `formation`; then
  `blockers` goal-side blockers, `attackers_ahead` attackers between the ball
  and the goal, the race for the ball (`fetch_distance`,
  `defender_head_start`) and `held_share` of the outfield players holding a
  direction.  Offside is on once the formation is a real match shape, with
  every attacker placed onside.
  """
  rng = random.Random('{}:{}:adr'.format(seed, episode))
  attack_right = (cycle // SPAWN_TEMPLATE_COUNT) % 2 == 0
  values['curriculum_episode_attackers'] = 11
  values['curriculum_episode_template'] = cycle % SPAWN_TEMPLATE_COUNT
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
  values['curriculum_goalside_defenders'] = goalside
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
        contested = (_ring_spawn(rank - 2, 0.0, ball_x, ball_y, direction, rng)
                     if attacking else standard)
        position = ((1 - formation) * contested[0] + formation * standard[0],
                    (1 - formation) * contested[1] + formation * standard[1])
      outfield.append((team, index, role, position[0], position[1]))

  positions = [[row[3], row[4]] for row in outfield]
  roles = ['attacker' if row[0] == attacking_team else
           'blocker' if row[1] - 1 < goalside else 'defender'
           for row in outfield]
  ahead_count = min(_MIXED_MAX_AHEAD,
                    _chance_count(float(settings['attackers_ahead']), rng))
  ahead = (_mixed_shape(positions, roles, ball_x, ball_y, direction,
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
  values['curriculum_loose_ball'] = racing
  values['curriculum_held_directions'] = held
  spaced = _separate([tuple(p) for p in positions])
  if offsides or (ahead and not racing):
    # In a race only offside matters (the race already placed everyone).
    _keep_mixed_shape(spaced, roles, [] if racing else ahead, ball_x, ball_y,
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


def _build_uniform(builder, values, seed, episode, cycle):
  """Uniform no-curriculum spawn: ball and players anywhere, all onside.

  Positions are drawn in world coordinates (left team attacking +x), spread
  apart, then made onside; the ball's half decides which side counts as
  attacking for success statistics, as in the curriculum spawn.
  """
  rng = random.Random('{}:{}:uniform'.format(seed, episode))
  ball_x = rng.uniform(-_PITCH_X, _PITCH_X)
  ball_y = rng.uniform(-_PITCH_Y, _PITCH_Y)
  attack_right = ball_x > 0
  values['curriculum_episode_attackers'] = 11
  values['curriculum_episode_template'] = cycle % SPAWN_TEMPLATE_COUNT
  values['curriculum_goalside_defenders'] = 0
  builder._config['reverse_team_processing'] = not attack_right
  builder.SetBallPosition(ball_x, ball_y)
  builder.config().game_duration = _UNIFORM_DURATION
  builder.config().deterministic = False
  builder.config().use_magnet = False
  builder.config().offsides = True
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True

  # Order: left keeper, right keeper, 10 left outfield, 10 right outfield.
  players = [
      (side * rng.uniform(-1.0, -1.0 + _KEEPER_BOX_DEPTH),
       rng.uniform(-_KEEPER_BOX_HALF_WIDTH, _KEEPER_BOX_HALF_WIDTH))
      for side in (1.0, -1.0)]
  players += [(rng.uniform(-_PITCH_X, _PITCH_X), rng.uniform(-_PITCH_Y, _PITCH_Y))
              for _ in range(20)]
  # Spreading players apart can nudge one offside and moving one onside can
  # land it on another, so alternate until a pass moves nobody.
  for _ in range(10):
    players = _separate(players)
    left = _onside(players[2:12], [players[1]] + players[12:22], ball_x,
                   1.0, rng)
    right = _onside(players[12:22], [players[0]] + left, ball_x, -1.0, rng)
    moved = left != players[2:12] or right != players[12:22]
    players = players[:2] + left + right
    if not moved:
      break
  for team, keeper, outfield in ((Team.e_Left, players[0], players[2:12]),
                                 (Team.e_Right, players[1], players[12:22])):
    builder.SetTeam(team)
    for index, (_, _, role) in enumerate(_FORMATION):
      position = keeper if index == 0 else outfield[index - 1]
      builder.AddPlayer(*_to_team_coordinates(team, *position), role)


def build_scenario(builder):
  episode = builder.EpisodeNumber()
  values = builder._config._values
  advantage = max(0.0, min(1.0, float(values.get('advantage', 1.0))))
  seed = int(values.get('game_engine_random_seed', 0))
  evaluation = bool(values.get('curriculum_evaluation', False))

  cycle = seed + episode
  values['curriculum_width_spawn'] = False
  values['curriculum_loose_ball'] = False
  values['curriculum_held_directions'] = [0] * 22
  if values.get('uniform_spawn', False):
    _build_uniform(builder, values, seed, episode, cycle)
    return
  if values.get('adr_values'):
    _build_adr(builder, values, seed, episode, cycle, values['adr_values'])
    return
  attack_right = (cycle // SPAWN_TEMPLATE_COUNT) % 2 == 0
  template = cycle % SPAWN_TEMPLATE_COUNT
  values['curriculum_episode_attackers'] = 11
  values['curriculum_episode_template'] = template
  builder._config['reverse_team_processing'] = not attack_right

  # Template and direction follow seed + episode so they cycle evenly, but
  # the rest of the scene is keyed on the (seed, episode) pair: evaluation
  # workers run consecutive seeds through consecutive episodes, and a sum key
  # made worker i's episode e the same scene as worker i+1's episode e-1.
  rng = random.Random('{}:{}'.format(seed, episode))
  phase = 1 / 6 + 2 / 3 * (
      template + (0.5 if evaluation else rng.random())) / SPAWN_TEMPLATE_COUNT
  # phase covers [1/6, 5/6]; rescale it onto [-1, 1] so the band edges are
  # exactly +/- the half-width for this level.
  template_ball_y = lateral_half_width(advantage) * (phase - 0.5) * 3.0

  direction = 1.0 if attack_right else -1.0
  ball_x = direction * _BALL_DEPTH * advantage
  ball_y = (advantage * template_ball_y +
            (1.0 - advantage) * rng.uniform(-0.22, 0.22))
  builder.SetBallPosition(ball_x, ball_y)

  builder.config().game_duration = int(300 + 400 * (1.0 - advantage))
  builder.config().deterministic = False
  builder.config().use_magnet = False
  # Both sides start level with the ball, so offside would flag half the
  # attacking team at spawn.  It only comes on once formations are normal.
  builder.config().offsides = advantage <= 0.1
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True

  # Lay every outfield player out in world coordinates first, so overlaps can
  # be resolved across both teams before anyone is committed to the scenario.
  attacking_team = Team.e_Left if attack_right else Team.e_Right
  # A separate generator for the blocker draw, keyed like `rng`, so adding or
  # removing blockers never shifts any other spawn draw.
  goalside = goalside_blockers(
      advantage, random.Random('{}:{}:blockers'.format(seed, episode)).random())
  values['curriculum_goalside_defenders'] = goalside
  # Own generator again, so the coin flip moves no other draw.
  blocker_side = (1.0 if random.Random(
      '{}:{}:blocker_side'.format(seed, episode)).random() < 0.5 else -1.0)
  # Width spawn, training only, with its own generator so no other draw moves.
  width_rng = random.Random('{}:{}:width'.format(seed, episode))
  width_spawn = (not evaluation and width_rng.random() <
                 float(values.get('width_spawn_fraction', 0.0)))
  values['curriculum_width_spawn'] = width_spawn
  outfield = []
  for team in (Team.e_Left, Team.e_Right):
    attacking = team == attacking_team
    team_phase = 0.0 if attacking else 0.5
    for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
      if index == 0:
        continue
      rank = index - 1
      # Everything below is in WORLD coordinates.  The ring/blocker/carrier
      # helpers already work in world space (they are built off ball_x and
      # direction); only the formation table is team-local, so that is the
      # single thing converted here.  The one conversion back to team space
      # happens at AddPlayer.
      standard = _to_team_coordinates(team, standard_x, standard_y)
      # Role-critical players are pinned to the ball at EVERY level.  Blending
      # them toward the formation is what silently removed the shot last time:
      # the ball moves as 0.90*advantage while the formation pulls harder, so
      # the carrier drifts off the ball and the blockers end up behind it.
      pinned = False
      if attacking and rank < 2:
        contested, pinned = _carrier_spawn(
            rank, ball_x, ball_y, direction, rng), True
      elif attacking:
        contested = _ring_spawn(
            rank - 2, team_phase, ball_x, ball_y, direction, rng)
      elif rank < goalside:
        contested, pinned = _blocker_spawn(
            rank, ball_x, ball_y, direction, blocker_side, rng), True
      else:
        # The rest of the defence holds its normal shape; ringing every
        # defender around the ball leaves the carrier no room to shoot.
        contested = standard
      if pinned:
        position = contested
      else:
        position = (advantage * contested[0] + (1 - advantage) * standard[0],
                    advantage * contested[1] + (1 - advantage) * standard[1])
      if width_spawn and (attacking or rank >= goalside):
        # Keep the level's x, spread across the width; blockers stay put.
        position = (position[0], width_rng.uniform(-_PITCH_Y, _PITCH_Y))
      outfield.append((team, index, role, position[0], position[1]))

  positions = [[row[3], row[4]] for row in outfield]
  roles = ['attacker' if row[0] == attacking_team else
           'blocker' if row[1] - 1 < goalside else 'defender'
           for row in outfield]
  levels = int(values.get('curriculum_levels', ADVANTAGE_LEVELS))
  level = int(round((1.0 - advantage) * (levels - 1)))
  # Mixed shape is part of the level, so gate scenes get it too.
  mixed = bool(values.get('mixed_shape', False)) and level >= 1
  ahead = (_mixed_shape(positions, roles, ball_x, ball_y, direction, level,
                        random.Random('{}:{}:mixed'.format(seed, episode)))
           if mixed else [])
  # Loose-ball spawn: training only, its own generator.
  loose_rng = random.Random('{}:{}:loose'.format(seed, episode))
  loose = (not evaluation and loose_rng.random() <
           float(values.get('loose_ball_fraction', 0.0)))
  held = [0] * 22
  if loose:
    _loose_ball(positions, roles, ball_x, ball_y, direction, level, loose_rng)
    for row in outfield:
      if loose_rng.random() < _LOOSE_HELD_SHARE:
        agent = (0 if row[0] == Team.e_Left else 11) + row[1]
        held[agent] = loose_rng.choice(_DIRECTION_ACTIONS)
  values['curriculum_loose_ball'] = loose
  values['curriculum_held_directions'] = held
  spaced = _separate([tuple(p) for p in positions])
  if mixed:
    _keep_mixed_shape(spaced, roles, ahead, ball_x, ball_y, direction)

  for team in (Team.e_Left, Team.e_Right):
    builder.SetTeam(team)
    for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
      if index == 0:
        builder.AddPlayer(-1.0, 0.0, role)
        continue
      position = next(spaced[k] for k, row in enumerate(outfield)
                      if row[0] == team and row[1] == index)
      builder.AddPlayer(*_to_team_coordinates(team, *position), role)
