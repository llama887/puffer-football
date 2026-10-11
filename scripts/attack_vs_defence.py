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
The defence is gfootball/env/defence.py, exactly what the training env and
its evaluation pools play.
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


def adr_ranges(log_path):
  """{setting: (low, high)}: valid limits, or a training log's last ranges."""
  ranges = {p.name: (p.minimum, p.maximum) for p in adr.PARAMETERS}
  if log_path:
    with open(log_path) as log:
      lines = [line for line in log if line.startswith('ADR {')]
    last = json.loads(lines[-1][4:])
    ranges = {name: (last['adr/{}/low'.format(name)],
                     last['adr/{}/high'.format(name)]) for name in ranges}
  return ranges


def scene_values(scene, ranges, rng):
  """ADR settings for one episode."""
  kind = scene.split(':')
  if kind[1] == 'difficulty':
    randomizer = adr.AutomaticDomainRandomization()
    randomizer.set_difficulty(float(kind[2]))
    return randomizer.sample(rng)
  if kind[1] == 'start':
    return {p.name: float(p.start) for p in adr.PARAMETERS}
  values = {name: float(rng.uniform(low, high)) if high > low else float(low)
            for name, (low, high) in ranges.items()}
  if kind[1] == 'pin':
    values[kind[2]] = float(ranges[kind[2]][int(kind[3])])
  return values


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
  ranges = adr_ranges(args.adr_log)
  video = args.video_dir is not None
  options = {
      'action_set': 'default', 'game_engine_random_seed': args.seed,
      'physics_steps_per_frame': 10, 'real_time': False}
  if video:
    options.update({'display_game_stats': True, 'render_resolution_x': 1280,
                    'render_resolution_y': 720, 'video_format': 'webm'})
  env = football_env.create_environment(
      env_name=adr.SCENARIO, representation='raw', rewards='scoring',
      render=video, write_goal_dumps=False, write_full_episode_dumps=video,
      write_video=video, dump_frequency=1,
      logdir=args.video_dir if video else tempfile.mkdtemp(),
      number_of_left_players_agent_controls=11,
      number_of_right_players_agent_controls=11,
      other_config_options=options)
  # Both sides are fixed policies played like the training env's defence
  # ('self' defends with the attacking checkpoint); separate seeds keep
  # their sampling noise independent.
  attacking_side = defence.Defence(args.checkpoint, len(ACTION_NAMES),
                                   args.seed, memory_window=HORIZON)
  defending_side = defence.Defence(
      args.checkpoint if args.defence == 'self' else args.defence,
      len(ACTION_NAMES), args.seed + 1, memory_window=HORIZON)
  # Scenes have their own generator so every defence sees the same scenes.
  scene_rng = np.random.default_rng([args.seed, 0xad5])
  sticky = np.zeros((22, entity_observation.STICKY_ACTIONS), np.float32)

  episodes = []
  try:
    for _ in range(args.episodes):
      values = scene_values(args.scene, ranges, scene_rng)
      env.unwrapped._config['adr_values'] = values
      raw = env.reset()
      sticky.fill(0)
      config = env.unwrapped._config
      held = np.asarray(config._values.get('held_directions', [0] * 22),
                        np.int64)
      attacking_side.start_episode()
      defending_side.start_episode()
      attacking_left = config.ScenarioConfig().ball_position[0] > 0
      attack = slice(0, 11) if attacking_left else slice(11, 22)
      defend = slice(11, 22) if attacking_left else slice(0, 11)
      flip = 1.0 if attacking_left else -1.0
      done, steps, reward = False, 0, 0.0
      regained, goal_side, pressure = False, [], []
      while not done:
        rows = entity_observation.build(raw, sticky)
        frames = wrappers.Simple115StateWrapper.convert_observation(raw, True)
        actions = np.zeros(22, np.int64)
        actions[attack] = attacking_side.act(rows[attack], frames[attack])
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
        raw, step_reward, done, _ = env.step(actions.astype(np.int32))
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
