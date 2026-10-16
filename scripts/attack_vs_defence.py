"""Score one policy's attack against a chosen defence, optionally on video.

This is the fixed-opponent check used to compare training runs: the same
attack-vs-defence match on the same scenes, whatever opponents each run
trained against.

The attack is --checkpoint.  The defence (--defence) is:
  self      the same policy (what ADR training episodes play)
  random    uniform random actions (what the default ADR probes play)
  idle      no action at all
  PATH      any policy snapshot, simple115 or entity observation (the
            fixed defender used to score runs is a simple115 MLP)
  bot:LEVEL the game's built-in AI, LEVEL easy / medium / hard (0.05 / 0.6
            / 0.95, as in the stock 11_vs_11_*_stochastic scenarios) or a
            difficulty in [0, 1]
Every defence except the bot is gfootball/env/defence.py, exactly what the
training env and its evaluation pools play.  The bot is the engine's own AI
(adr.BOT_DIFFICULTY) with the ADR scenes' physics (no magnet, no automatic
aiming), like the policies.  Its team has no agents at all: players handed
to the AI with the builtin_ai action do not defend when the magnet is off
(they walk to the halfway line, 2026-10-14), while uncontrolled players do.
Agents then control only the left team, so in bot matches the left team
attacks every scene (adr.ATTACKING_TEAM).  Bot episodes share the other
defences' ADR values but not their placements (which side, blocker lanes,
spawn jitter), since the other defences alternate sides.
Scenes (--scene):
  adr:difficulty:D   every ADR setting uniform over its range at test
                     difficulty D (adr set_difficulty; what the trainer's
                     evaluations play)
  adr:start          every ADR setting at its start value
  adr:full           every ADR setting uniform over its ranges
  adr:pin:NAME:EDGE  as adr:full with NAME pinned at edge 0 (low) or 1
                     (high), exactly like an ADR probe
The ADR ranges are each setting's valid limits, or with --adr-log the last
`ADR {json}` line of a training log (the ranges that run had reached).

Each episode records whether the attack scored and how the defence played:
whether it ever won the ball, how many outfield defenders were goal-side of
the ball and how close the nearest defender was while the attack had it.
With --video-dir every episode is rendered to webm; rendering needs an EGL
display, which Torch has on GPU nodes but not on the cs6xx CPU nodes.
"""

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

# Training resets the recurrent state every rollout window (32 steps), so
# both sides restart their memory every 32 of their own steps and at every
# episode start (gfootball/env/defence.py).  Scores from before the defence
# module (2026-10-11) used other seeds and windows, so they do not reproduce
# exactly.
HORIZON = 32
# The built-in AI's difficulty per bot:LEVEL name.
BOT_LEVELS = {'easy': 0.05, 'medium': 0.6, 'hard': 0.95}


def bot_difficulty(spec):
  """The built-in AI difficulty a --defence bot:LEVEL names, else None.

  LEVEL is easy / medium / hard or a number in [0, 1]; anything else is
  refused, so a typo never falls through to a snapshot path.
  """
  if not spec.startswith('bot:'):
    return None
  level = spec[len('bot:'):]
  difficulty = BOT_LEVELS[level] if level in BOT_LEVELS else float(level)
  if not 0.0 <= difficulty <= 1.0:
    raise ValueError('bot difficulty must be in [0, 1], got {}'.format(spec))
  return difficulty


def scene_ranges(scene, log_path):
  """The ADR ranges a --scene draws its episodes from.

  A test difficulty's ranges, the start values (points), or for full and
  pin scenes every setting's valid limits, or with --adr-log the ranges of
  that training log's last `ADR {json}` line.  Any other scene is refused,
  so a misspelt or retired one (level:L) never quietly plays full ranges.
  """
  randomizer = adr.AutomaticDomainRandomization()
  kind = scene.split(':')
  parts = {'difficulty': 3, 'start': 2, 'full': 2, 'pin': 4}
  if kind[0] != 'adr' or len(kind) < 2 or parts.get(kind[1]) != len(kind):
    raise ValueError('unknown --scene {!r} (see the module docstring)'.format(
        scene))
  if kind[1] == 'difficulty':
    randomizer.set_difficulty(float(kind[2]))
  elif kind[1] != 'start':
    if log_path:
      with open(log_path) as log:
        lines = [line for line in log if line.startswith('ADR {')]
      randomizer.load_metrics(json.loads(lines[-1][4:]))
    else:
      randomizer.set_difficulty(1.0)
  return randomizer


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--defence', required=True)
  parser.add_argument('--scene', required=True)
  parser.add_argument('--adr-log', default=None)
  parser.add_argument('--episodes', type=int, default=64)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--video-dir', default=None)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  ranges = scene_ranges(args.scene, args.adr_log)
  pinned = args.scene.split(':')[2:] if ':pin:' in args.scene else None
  video = args.video_dir is not None
  bot = bot_difficulty(args.defence)
  options = {
      'action_set': 'default', 'game_engine_random_seed': args.seed,
      'physics_steps_per_frame': 10, 'real_time': False}
  if bot is not None:
    options[adr.BOT_DIFFICULTY] = bot
    options[adr.ATTACKING_TEAM] = 'left'
  if video:
    options.update({'display_game_stats': True, 'render_resolution_x': 1280,
                    'render_resolution_y': 720, 'video_format': 'webm'})
  env = football_env.create_environment(
      env_name=adr.SCENARIO, representation='raw', rewards='scoring',
      render=video, write_goal_dumps=False, write_full_episode_dumps=video,
      write_video=video, dump_frequency=1,
      logdir=args.video_dir if video else tempfile.mkdtemp(),
      number_of_left_players_agent_controls=11,
      number_of_right_players_agent_controls=0 if bot is not None else 11,
      other_config_options=options)
  # Both sides are fixed policies played like the training env's defence
  # ('self' defends with the attacking checkpoint); separate seeds keep
  # their sampling noise independent.
  attacking_side = defence.Defence(args.checkpoint, len(ACTION_NAMES),
                                   args.seed, memory_window=HORIZON)
  defending_side = None if bot is not None else defence.Defence(
      args.checkpoint if args.defence == 'self' else args.defence,
      len(ACTION_NAMES), args.seed + 1, memory_window=HORIZON)
  # Scenes have their own generator so every defence sees the same scenes.
  scene_rng = np.random.default_rng([args.seed, 0xad5])
  sticky = np.zeros((22, entity_observation.STICKY_ACTIONS), np.float32)

  episodes = []
  try:
    for _ in range(args.episodes):
      if pinned:
        # Exactly like an ADR probe: one setting at one edge of its range.
        values = ranges.sample_pinned(
            scene_rng, adr.PARAMETER_INDEX[pinned[0]], int(pinned[1]))
      else:
        values = ranges.sample(scene_rng)
      env.unwrapped._config[adr.VALUES] = values
      raw = env.reset()
      scene = adr.read_scene(env.unwrapped._config)
      sticky.fill(0)
      held = scene.held_directions
      attacking_side.start_episode()
      if defending_side is not None:
        defending_side.start_episode()
      attacking_left = scene.attacking_left
      attack = slice(0, 11) if attacking_left else slice(11, 22)
      defend = slice(11, 22) if attacking_left else slice(0, 11)
      flip = 1.0 if attacking_left else -1.0
      done, steps, reward = False, 0, 0.0
      regained, goal_side, pressure = False, [], []
      while not done:
        # Against the bot only the 11 left agents have views; build reads
        # each team's views on their own, so the copy stands in for the
        # right team's and its rows are never used.
        views = list(raw) * 2 if bot is not None else raw
        rows = entity_observation.build(views, sticky)
        frames = wrappers.Simple115StateWrapper.convert_observation(views, True)
        actions = np.zeros(22, np.int64)
        actions[attack] = attacking_side.act(rows[attack], frames[attack])
        if defending_side is not None:
          actions[defend] = defending_side.act(rows[defend], frames[defend])
        # A loose-ball ADR scene opens with held directions pressed, as in
        # training.
        if steps == 0 and held.any():
          actions[held > 0] = held[held > 0]
        first = raw[0]
        ball = np.asarray(first['ball'][:2]) * flip
        defenders = (np.asarray(first['right_team' if attacking_left
                                      else 'left_team']) * flip)[1:]
        owner = int(first['ball_owned_team'])
        if owner >= 0:
          attack_owns = (owner == 0) == attacking_left
          regained |= not attack_owns
          if attack_owns:
            goal_side.append(int((defenders[:, 0] > ball[0]).sum()))
            pressure.append(float(np.min(np.linalg.norm(
                defenders - ball, axis=1))))
        entity_observation.press_buttons(sticky, actions)
        raw, step_reward, done, _ = env.step(
            actions[:len(raw)].astype(np.int32))
        reward += float(np.asarray(step_reward).reshape(-1)[
            0 if attacking_left else 11])
        steps += 1
      record = {'scored': reward > 0, 'conceded': reward < 0,
                'steps': steps, 'regained': regained,
                'goal_side': float(np.mean(goal_side)) if goal_side else None,
                'pressure': float(np.mean(pressure)) if pressure else None,
                'values': values}
      episodes.append(record)
      print('EPISODE ' + json.dumps(record), flush=True)
  finally:
    env.close()

  def mean(key):
    present = [e[key] for e in episodes if e[key] is not None]
    return float(np.mean(present)) if present else float('nan')

  summary = {'scene': args.scene, 'defence': args.defence,
             'checkpoint': args.checkpoint, 'episodes': len(episodes),
             'scored': mean('scored'), 'conceded': mean('conceded'),
             'regained': mean('regained'), 'goal_side': mean('goal_side'),
             'pressure': mean('pressure'), 'steps': mean('steps')}
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump({'summary': summary, 'episodes': episodes}, f)
  print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
  main()
