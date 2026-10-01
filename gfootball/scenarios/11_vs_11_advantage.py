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

Every episode ends as soon as the ball goes out of play.  With the magnet off
the engine never designates a set-piece taker, so a goal kick, corner or
throw-in would otherwise freeze the match until the timeout.
"""

import math
import random

from . import *
from gfootball.curriculum import (
    SPAWN_TEMPLATE_COUNT, goalside_blockers, lateral_half_width)


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
  if values.get('uniform_spawn', False):
    _build_uniform(builder, values, seed, episode, cycle)
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
  spaced = _separate([(row[3], row[4]) for row in outfield])

  for team in (Team.e_Left, Team.e_Right):
    builder.SetTeam(team)
    for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
      if index == 0:
        builder.AddPlayer(-1.0, 0.0, role)
        continue
      position = next(spaced[k] for k, row in enumerate(outfield)
                      if row[0] == team and row[1] == index)
      builder.AddPlayer(*_to_team_coordinates(team, *position), role)
