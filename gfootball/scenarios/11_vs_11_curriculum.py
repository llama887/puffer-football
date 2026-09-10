# coding=utf-8
"""11v11 self-play that expands from a near-goal scoring curriculum."""

import math
import random

from . import *
from gfootball.curriculum import (
    ATTACKER_ORDER, DEFENDER_ORDER, SPAWN_TEMPLATE_COUNT, TOTAL_LEVELS,
    curriculum_episode, curriculum_episode_duration, curriculum_geometry,
    curriculum_state, keeper_spawn_offset)


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


def _spawn_parameters(evaluation, template_index, rng):
  phase = 1 / 6 + 2 / 3 * (
      template_index + (0.5 if evaluation else rng.random())) / (
          SPAWN_TEMPLATE_COUNT)
  angle = 2 * math.pi * phase
  return (0.03 * (2 * phase - 1), 0.03 * math.cos(angle),
          0.03 * math.sin(angle))


def _to_team_coordinates(team, x, y):
  side = 1.0 if team == Team.e_Left else -1.0
  return side * x, side * y


def _add_team(builder, team, attacking, active_count, progress, ball_x, ball_y,
              direction, carrier_gap, carrier_offset, rng, keeper_y=0.0):
  builder.SetTeam(team)
  for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
    order = ATTACKER_ORDER if attacking else DEFENDER_ORDER
    rank = order.index(index) if index in order else -1
    if attacking and rank < active_count:
      if rank < 2:
        world_x = ball_x - direction * (carrier_gap + 0.03 * rank)
        world_y = ball_y + carrier_offset * (1 if rank == 0 else -1)
      else:
        row, lane = divmod(rank - 2, 4)
        row_count = min(4, active_count - 2 - 4 * row)
        lane_offset = (lane - (row_count - 1) / 2) * 0.085
        if row_count == 1:
          lane_offset = -1.5 * carrier_offset
        world_x = ball_x - direction * (0.14 + 0.055 * row)
        world_y = ball_y + lane_offset
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36,
                                       world_y + rng.uniform(-0.006, 0.006))))
    elif attacking:
      world_x = -direction * (0.30 + 0.04 * (rank // 3))
      world_y = (rank % 5 - 2) * 0.15
      guided_x, guided_y = _to_team_coordinates(team, world_x, world_y)
    elif index == 0:
      guided_x, guided_y = -1.0, keeper_y
    elif rank < active_count:
      world_x = ball_x + direction * (0.07 + 0.025 * (rank // 2))
      world_y = ball_y + (rank // 2 + 1) * 0.055 * (
          -1.0 if rank % 2 else 1.0)
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36, world_y)))
    else:
      world_x = direction * (0.12 + 0.04 * (rank // 3))
      world_y = (rank % 5 - 2) * 0.15
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36, world_y)))
    x = progress * standard_x + (1.0 - progress) * guided_x
    y = progress * standard_y + (1.0 - progress) * guided_y
    builder.AddPlayer(x, y, role)


def build_scenario(builder):
  episode = builder.EpisodeNumber()
  curriculum_level = max(0, min(
      TOTAL_LEVELS - 1, int(builder._config['curriculum_level'])))
  _, active_defenders, progress = curriculum_state(curriculum_level)
  _, alignment = curriculum_geometry(curriculum_level)
  seed = int(builder._config._values.get('game_engine_random_seed', 0))
  evaluation = bool(
      builder._config._values.get('curriculum_evaluation', False))
  active_attackers, attack_right, template_index = curriculum_episode(
      curriculum_level, seed, episode)
  # Fix the engine processing order for every level.  Letting it change
  # mid-curriculum is a discontinuity the policy cannot see coming.
  builder._config['reverse_team_processing'] = not attack_right
  builder._config._values['curriculum_episode_attackers'] = active_attackers
  builder._config._values['curriculum_episode_template'] = template_index
  rng = random.Random(seed + episode)
  template_ball_y, spawn_gap, spawn_offset = _spawn_parameters(
      evaluation, template_index, rng)
  # At alignment 0 the carrier stands squarely behind the ball, lined up to
  # shoot, and depth varies instead; at alignment 1 it takes the full
  # randomized gap and lateral offset.
  ball_distance = 0.90 + (2 / 3) * spawn_gap * (1.0 - alignment)
  carrier_gap = 0.03 + alignment * (spawn_gap - 0.03)
  carrier_offset = alignment * spawn_offset
  direction = 1.0 if attack_right else -1.0
  ball_x = direction * ball_distance * (1.0 - progress)
  ball_y = ((1.0 - progress) * template_ball_y +
            progress * rng.uniform(-0.22, 0.22))
  # The keeper walks in from outside the post to the centre of the goal.
  keeper_y = keeper_spawn_offset(curriculum_level)

  builder.config().game_duration = curriculum_episode_duration(
      curriculum_level)
  builder.config().deterministic = False
  builder.config().use_magnet = False
  builder.config().offsides = progress >= 0.75
  builder.config().end_episode_on_score = progress < 1.0
  builder.SetBallPosition(ball_x, ball_y)

  attacking_team = Team.e_Left if attack_right else Team.e_Right
  _add_team(builder, Team.e_Left, attacking_team == Team.e_Left,
            active_attackers if attacking_team == Team.e_Left
            else active_defenders, progress, ball_x, ball_y, direction,
            carrier_gap, carrier_offset, rng, keeper_y=keeper_y)
  _add_team(builder, Team.e_Right, attacking_team == Team.e_Right,
            active_attackers if attacking_team == Team.e_Right
            else active_defenders, progress, ball_x, ball_y, direction,
            carrier_gap, carrier_offset, rng, keeper_y=keeper_y)
