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
# The first blocker sits off to one side (the measured level-0 anchor, left
# exactly as it was).  The second MIRRORS it on the other side, so the pair is
# symmetric and the shot line between them stays open; only the third closes
# the middle.  Before this the second blocker landed on the shot line itself.
_BLOCKER_LANES = (-1, 1, 0)
_LANE_WIDTH = 0.085


def _blocker_spawn(rank, ball_x, ball_y, direction, rng):
  """A defensive line between the ball and the goal it protects."""
  row, slot = divmod(rank, len(_BLOCKER_LANES))
  depth = 0.07 + 0.04 * row
  offset = _BLOCKER_LANES[slot] * _LANE_WIDTH
  forward_room = max(0.04, _PITCH_X - abs(ball_x))
  depth = min(depth, forward_room)
  return (ball_x + direction * depth,
          ball_y + offset + rng.uniform(-0.004, 0.004))


def build_scenario(builder):
  episode = builder.EpisodeNumber()
  values = builder._config._values
  advantage = max(0.0, min(1.0, float(values.get('advantage', 1.0))))
  seed = int(values.get('game_engine_random_seed', 0))
  evaluation = bool(values.get('curriculum_evaluation', False))

  cycle = seed + episode
  attack_right = (cycle // SPAWN_TEMPLATE_COUNT) % 2 == 0
  template = cycle % SPAWN_TEMPLATE_COUNT
  values['curriculum_episode_attackers'] = 11
  values['curriculum_episode_template'] = template
  builder._config['reverse_team_processing'] = not attack_right

  rng = random.Random(seed + episode)
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

  # Lay every outfield player out in world coordinates first, so overlaps can
  # be resolved across both teams before anyone is committed to the scenario.
  attacking_team = Team.e_Left if attack_right else Team.e_Right
  # A separate generator for the blocker draw keeps every other spawn draw,
  # and therefore level 0 itself, bit-identical to the measured anchor.
  goalside = goalside_blockers(
      advantage, random.Random((seed + episode) * 1000003 + 97).random())
  values['curriculum_goalside_defenders'] = goalside
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
            rank, ball_x, ball_y, direction, rng), True
      else:
        # The rest of the defence holds its normal shape; ringing every
        # defender around the ball leaves the carrier no room to shoot.
        contested = standard
      if pinned:
        position = contested
      else:
        position = (advantage * contested[0] + (1 - advantage) * standard[0],
                    advantage * contested[1] + (1 - advantage) * standard[1])
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
