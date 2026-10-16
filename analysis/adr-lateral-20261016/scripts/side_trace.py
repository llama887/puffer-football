"""Trace what the attack does when the ball starts to the side.

One scene per run: the ADR start scene with ball_lateral = --lateral and
ball_distance = --distance, the left team always attacking (adr.ATTACKING_TEAM),
so every position below is in the attack's frame (attacked goal at x = +1,
posts at |y| = 0.044, touchlines at |y| = 0.42).  The attack is a checkpoint
played as in scripts/attack_vs_defence.py; the defence is 'idle' (11 agents
doing nothing) or 'bot:D' (the engine's AI at difficulty D, its team with no
agents).

Per episode it records how it ended (goal, wide = out over the goal line off
the attack, corner = out over the goal line off the defence, touchline, or
still in play at the timeout), whether the keeper or an outfield defender ever
had the ball, the ball's path, every step an attacker was at the ball (within
0.03 or owning it: its position and action), and the first kick: the first
step the ball moves faster than a dribble (0.02 per step), where it was struck
from and where its line of travel crosses the goal line (its aim).  The
policies often strike the ball without the engine ever giving them
possession, so kicks are found from the ball's speed, not from ownership.
"""
KICK_SPEED = 0.02
AT_BALL = 0.03
KICK_BUFFER = 25

import argparse
import json
import os
import tempfile

import numpy as np

import gfootball.env as football_env
from gfootball import adr
from gfootball.env import defence
from gfootball.env import entity_observation
from gfootball.env import wrappers
from gfootball.examples.train_puffer import ACTION_NAMES

HORIZON = 32
GAME_MODES = {0: 'normal', 1: 'kickoff', 2: 'goal_kick', 3: 'free_kick',
              4: 'corner', 5: 'throw_in', 6: 'penalty'}


A = {name: i for i, name in enumerate(ACTION_NAMES)}
KICKS = {A['short_pass'], A['long_pass'], A['high_pass']}
# Physical units: y counts 83.6 / 54.4 times an x unit.
Y_TO_X = 83.6 / 54.4
# A shot along a diagonal direction (45 degrees physically) moves the ball
# this much in y per unit of x travelled.
DIAGONAL_Y_PER_X = 1.0 / Y_TO_X
# Where a shot may cross the goal line and still go in (posts at 0.044).
TARGET = 0.025
# Steps to hold the shooting direction before pressing shot, so the
# player has turned to it.
HOLD = 3
DIRECTIONS = {  # action: unit vector in physical units, engine y down
    'left': (-1, 0), 'top_left': (-1, -1), 'top': (0, -1),
    'top_right': (1, -1), 'right': (1, 0), 'bottom_right': (1, 1),
    'bottom': (0, 1), 'bottom_left': (-1, 1)}


def nearest_direction(vector):
  """The movement action closest to `vector` (pitch units)."""
  physical = np.array([vector[0], vector[1] * Y_TO_X])
  return max(DIRECTIONS, key=lambda name: np.dot(
      physical, np.asarray(DIRECTIONS[name]) / np.hypot(*DIRECTIONS[name])))


class DribbleThenShoot:
  """Scripted attack: the attacker nearest the ball runs to it; at the ball
  it dribbles sideways toward the goal centre until a straight or diagonal
  shot (the only directions there are without aim assist) would cross the
  goal line inside |y| <= TARGET, holds that direction HOLD steps, then
  shoots.  Everyone else stands still."""

  def __init__(self):
    self.want, self.held = None, 0

  def start_episode(self):
    self.want, self.held = None, 0

  def act(self, first):
    actions = np.zeros(11, np.int64)
    ball = np.asarray(first['ball'][:2], float)
    team = np.asarray(first['left_team'], float)
    carrier = int(np.argmin(np.linalg.norm(team - ball, axis=1)))
    if int(first['ball_owned_team']) == 0:
      carrier = int(first['ball_owned_player'])
    gap = float(np.linalg.norm(team[carrier] - ball))
    if gap > 0.02 and int(first['ball_owned_team']) != 0:
      actions[carrier] = A[nearest_direction(ball - team[carrier])]
      self.want, self.held = None, 0
      return actions
    side = 'top' if ball[1] > 0 else 'bottom'
    room = 1.0 - ball[0]
    straight = abs(ball[1])
    diagonal = abs(ball[1]) - DIAGONAL_Y_PER_X * room
    if straight <= TARGET:
      want = 'right'
    elif abs(diagonal) <= TARGET:
      want = side + '_right'
    else:
      # Too wide for a diagonal, or between the straight and diagonal
      # lanes: carry the ball sideways toward the centre.
      want = side
    if want == side:
      self.want, self.held = want, 0
      actions[carrier] = A[want]
      return actions
    self.held = self.held + 1 if want == self.want else 1
    self.want = want
    actions[carrier] = A['shot'] if self.held > HOLD else A[want]
    return actions


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--checkpoint', default=None,
                      help='policy checkpoint (not needed for --scripted)')
  parser.add_argument('--scripted', action='store_true',
                      help='play the attack with DribbleThenShoot')
  parser.add_argument('--adr-log', default=None,
                      help='one-line ADR log: every episode draws all the '
                           'settings from its ranges (a real probe scene), '
                           'then pins ball_lateral at --lateral; '
                           '--distance is then ignored')
  parser.add_argument('--video-dir', default=None,
                      help='render every episode to webm here (needs an EGL '
                           'display: Torch GPU nodes)')
  parser.add_argument('--kick-to-shot', action='store_true',
                      help="turn the policy's pass presses at the ball into "
                           'shot (with GFOOTBALL_SHOT_AUTOAIM set in a '
                           'patched engine, shots are aimed at the goal)')
  parser.add_argument('--defence', required=True)
  parser.add_argument('--lateral', type=float, required=True)
  parser.add_argument('--distance', type=float, required=True)
  parser.add_argument('--episodes', type=int, default=40)
  parser.add_argument('--seed', type=int, default=4242)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  bot = args.defence.startswith('bot:')
  values = {p.name: float(p.start) for p in adr.PARAMETERS}
  values['ball_lateral'] = args.lateral
  values['ball_distance'] = args.distance
  options = {'action_set': 'default', 'game_engine_random_seed': args.seed,
             'physics_steps_per_frame': 10, 'real_time': False,
             adr.ATTACKING_TEAM: 'left'}
  if bot:
    options[adr.BOT_DIFFICULTY] = float(args.defence.split(':')[1])
  video = args.video_dir is not None
  if video:
    os.makedirs(args.video_dir, exist_ok=True)
    options.update({'display_game_stats': True, 'render_resolution_x': 1280,
                    'render_resolution_y': 720, 'video_format': 'webm'})
  env = football_env.create_environment(
      env_name=adr.SCENARIO, representation='raw', rewards='scoring',
      render=video, write_goal_dumps=False, write_full_episode_dumps=video,
      write_video=video, dump_frequency=1,
      logdir=args.video_dir if video else tempfile.mkdtemp(),
      number_of_left_players_agent_controls=11,
      number_of_right_players_agent_controls=0 if bot else 11,
      other_config_options=options)
  attack = (DribbleThenShoot() if args.scripted else
            defence.Defence(args.checkpoint, len(ACTION_NAMES), args.seed,
                            memory_window=HORIZON))
  sticky = np.zeros((22, entity_observation.STICKY_ACTIONS), np.float32)
  episodes = []
  try:
    ranges = None
    if args.adr_log:
      ranges = adr.AutomaticDomainRandomization()
      line = [l for l in open(args.adr_log) if l.startswith('ADR {')][-1]
      ranges.load_metrics(json.loads(line[4:]))
    scene_rng = np.random.default_rng([args.seed, 0xad5])
    for _ in range(args.episodes):
      if ranges is not None:
        values = ranges.sample(scene_rng)
        values['ball_lateral'] = args.lateral
      env.unwrapped._config[adr.VALUES] = dict(values)
      raw = env.reset()
      scene = adr.read_scene(env.unwrapped._config)
      assert scene.attacking_left
      held = scene.held_directions
      sticky.fill(0)
      attack.start_episode()
      done, step, reward = False, 0, 0.0
      path, owners, carrier = [], [], []
      keeper_had, defender_had = False, False
      first_fast = None
      # Kicks: the ball speeds past KICK_SPEED within KICK_BUFFER steps of
      # an attacker near it pressing a kick button (the engine buffers the
      # press until the touch), so fast dribble touches are not kicks.
      kicks, pressed, last_dir = [], {}, ['none'] * 11
      previous_speed = 0.0
      while not done:
        views = list(raw) * 2 if bot else raw
        rows = entity_observation.build(views, sticky)
        frames = wrappers.Simple115StateWrapper.convert_observation(views, True)
        first = raw[0]
        ball = np.asarray(first['ball'], float)
        team = int(first['ball_owned_team'])
        player = int(first['ball_owned_player'])
        actions = np.zeros(22, np.int64)
        if args.scripted:
          actions[:11] = attack.act(first)
        else:
          actions[:11] = attack.act(rows[:11], frames[:11])
        if args.kick_to_shot:
          team_now = np.asarray(first['left_team'], float)
          at_ball = (player if team == 0 else int(np.argmin(
              np.linalg.norm(team_now - ball[:2], axis=1))))
          if (team == 0 or np.linalg.norm(team_now[at_ball] - ball[:2]) <
              AT_BALL) and actions[at_ball] in KICKS:
            actions[at_ball] = A['shot']
        if step == 0 and held.any():
          actions[held > 0] = held[held > 0]
        path.append([round(float(ball[0]), 4), round(float(ball[1]), 4),
                     round(float(ball[2]), 3)])
        owners.append(1 if team == 0 else (-1 if team == 1 else 0))
        if team == 1:
          if player == 0:
            keeper_had = True
          else:
            defender_had = True
        attackers = np.asarray(first['left_team'], float)
        near = int(np.argmin(np.linalg.norm(attackers - ball[:2], axis=1)))
        gap = float(np.linalg.norm(attackers[near] - ball[:2]))
        if team == 0 or gap < AT_BALL:
          at = player if team == 0 else near
          carrier.append([step, round(float(attackers[at][0]), 4),
                          round(float(attackers[at][1]), 4),
                          ACTION_NAMES[actions[at]]])
        for i in range(11):
          name = ACTION_NAMES[actions[i]]
          if name in DIRECTIONS:
            last_dir[i] = name
          elif name == 'release_direction':
            last_dir[i] = 'none'
          if (actions[i] in KICKS or actions[i] == A['shot']) and float(
              np.linalg.norm(attackers[i] - ball[:2])) < 0.05:
            pressed[i] = (step, name, last_dir[i])
        speed = float(np.hypot(*np.asarray(first['ball_direction'][:2])))
        if len(path) > 1 and team != 1:
          before = path[-2]
          direction = np.asarray(first['ball_direction'][:2], float)
          aim = (before[1] + direction[1] * (1.0 - before[0]) / direction[0]
                 if direction[0] > 1e-4 else None)
          if first_fast is None and speed > KICK_SPEED:
            first_fast = {'step': step - 1, 'x': before[0], 'y': before[1],
                          'aim_y': aim, 'speed': speed}
          if speed > KICK_SPEED >= previous_speed:
            recent = [(s, i, b, d) for i, (s, b, d) in pressed.items()
                      if step - s <= KICK_BUFFER and float(np.linalg.norm(
                          attackers[i] - np.asarray(before[:2]))) < 0.06]
            if recent:
              s, i, button, held_dir = max(recent)
              kicks.append({'step': step - 1, 'x': before[0], 'y': before[1],
                            'aim_y': aim, 'speed': speed, 'button': button,
                            'held': held_dir, 'kicker': i})
              del pressed[i]
        previous_speed = speed
        entity_observation.press_buttons(sticky, actions)
        raw, step_reward, done, _ = env.step(
            actions[:11 if bot else 22].astype(np.int32))
        reward += float(np.asarray(step_reward).reshape(-1)[0])
        step += 1
      last = raw[0]
      mode = GAME_MODES.get(int(last['game_mode']), str(last['game_mode']))
      final = np.asarray(last['ball'], float)
      path.append([round(float(final[0]), 4), round(float(final[1]), 4),
                   round(float(final[2]), 3)])
      if reward > 0:
        end = 'goal'
      elif mode == 'goal_kick':
        end = 'wide'
      elif mode == 'corner':
        end = 'corner'
      elif mode == 'throw_in':
        end = 'touchline'
      else:
        end = 'timeout'
      cross = next((p[1] for p in path if p[0] >= 1.0), None)
      episodes.append({'end': end, 'mode': mode, 'steps': step,
                       'values': {k: values[k] for k in ('ball_lateral', 'ball_distance', 'blockers', 'fetch_distance', 'defender_head_start', 'attackers_ahead', 'formation', 'held_share')},
                       'keeper_had': keeper_had, 'defender_had': defender_had,
                       'cross_y': cross, 'first_kick': kicks[0] if kicks else None,
                       'first_fast': first_fast, 'kicks': kicks,
                       'path': path, 'owners': owners, 'carrier': carrier})
      print('EPISODE', end, step, flush=True)
  finally:
    env.close()
  summary = {'checkpoint': args.checkpoint, 'scripted': args.scripted,
             'kick_to_shot': args.kick_to_shot,
             'autoaim': os.environ.get('GFOOTBALL_SHOT_AUTOAIM'), 'defence': args.defence,
             'lateral': args.lateral, 'distance': args.distance,
             'episodes': len(episodes),
             'goal_rate': float(np.mean([e['end'] == 'goal' for e in episodes]))}
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump({'summary': summary, 'episodes': episodes}, f)
  print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
  main()
