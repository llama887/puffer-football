#!/usr/bin/env python3
"""Capture or compare a rollout trace, to prove a refactor changed nothing.

The observation pipeline was rewritten for speed.  Unit tests cover each piece
against a reference implementation, but the end-to-end guarantee is that the
same seed and the same actions still produce the same observations, rewards and
terminals out of the real engine:

    git checkout main
    python -m gfootball.examples.observation_trace --out /tmp/main.npz
    git checkout wesley-speed-improvement
    python -m gfootball.examples.observation_trace --compare /tmp/main.npz

Run it at a few curriculum levels: level 0 controls one player, the later ones
control all 22 and exercise the right-team observation flip.
"""

import argparse
from multiprocessing import RawValue

import numpy as np

from gfootball.curriculum import TOTAL_LEVELS
from gfootball.env.puffer_env import FootballPufferEnv


def capture(level, steps, seed, frame_stack, sort_players, env_name,
            attacker_only_levels):
  """Deterministic rollout: fixed seed, fixed action sequence, pinned level."""
  # A shared level value pins the curriculum: the env only self-advances when
  # it owns its level, so this also keeps the trace on one level throughout.
  env = FootballPufferEnv(
      env_name=env_name, seed=seed, frame_stack=frame_stack,
      sort_players=sort_players, attacker_only_levels=attacker_only_levels,
      curriculum_level_value=RawValue('i', level))
  try:
    rng = np.random.default_rng(seed)
    num_actions = env.single_action_space.n
    observations, rewards, terminals = [], [], []
    env.reset(seed=seed)
    observations.append(env.observations.copy())
    for _ in range(steps):
      actions = rng.integers(0, num_actions, env.num_agents)
      observation, reward, terminal, _, _ = env.step(actions)
      observations.append(observation.copy())
      rewards.append(reward.copy())
      terminals.append(terminal.copy())
  finally:
    env.close()
  return {
      'observations': np.asarray(observations),
      'rewards': np.asarray(rewards),
      'terminals': np.asarray(terminals),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', help='write the trace to this .npz')
  parser.add_argument('--compare', help='compare against this .npz')
  parser.add_argument('--levels', default='0,20,30',
                      help='comma separated curriculum levels to trace')
  parser.add_argument('--steps', type=int, default=200)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--frame-stack', type=int, default=1, choices=(1, 4))
  parser.add_argument('--sort-players',
                      action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--env-name', default='11_vs_11_curriculum')
  parser.add_argument('--attacker-only-levels', type=int, default=0,
                      help='0 activates defenders from level 0, which covers '
                           'more of the observation pipeline than training '
                           'does at the early levels')
  args = parser.parse_args()
  if bool(args.out) == bool(args.compare):
    parser.error('pass exactly one of --out or --compare')

  levels = [int(level) for level in args.levels.split(',')]
  for level in levels:
    if not 0 <= level < TOTAL_LEVELS:
      parser.error('level {} is outside the curriculum'.format(level))

  trace = {}
  for level in levels:
    for name, array in capture(
        level, args.steps, args.seed, args.frame_stack, args.sort_players,
        args.env_name, args.attacker_only_levels).items():
      trace['level{}_{}'.format(level, name)] = array

  if args.out:
    np.savez_compressed(args.out, **trace)
    print('Wrote {} arrays to {}'.format(len(trace), args.out))
    return

  expected = np.load(args.compare)
  missing = set(expected.files) ^ set(trace)
  if missing:
    raise SystemExit('trace keys differ: {}'.format(sorted(missing)))
  for name in sorted(trace):
    np.testing.assert_array_equal(
        trace[name], expected[name], err_msg='{} differs'.format(name))
    print('{}: identical {}'.format(name, trace[name].shape))
  print('All {} arrays identical to {}'.format(len(trace), args.compare))


if __name__ == '__main__':
  main()
