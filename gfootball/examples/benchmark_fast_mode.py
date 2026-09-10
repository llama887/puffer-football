"""Measure environment-only throughput without starting RL training.

Defaults to the shape training actually runs -- 11 agents a side, simple115v2
-- because almost all of the per-step Python cost scales with the number of
controlled players, so a single-agent measurement says very little about it.
"""

import argparse
import cProfile
import json
import pstats
import statistics
import sys
import time

import gfootball.env as football_env


def make_env(players, fast_mode, seed, needs_sticky_actions):
  return football_env.create_environment(
      env_name='11_vs_11_stochastic',
      representation='simple115v2',
      render=False,
      write_goal_dumps=False,
      write_full_episode_dumps=False,
      write_video=False,
      stacked=False,
      number_of_left_players_agent_controls=players,
      number_of_right_players_agent_controls=players,
      other_config_options={
          'action_set': 'default',
          'fast_mode': fast_mode,
          'game_engine_random_seed': seed,
          'needs_sticky_actions': needs_sticky_actions,
          'real_time': False,
      })


def measure(fast_mode, steps, repeats, players, needs_sticky_actions,
            resets=5):
  """Median steps/second and median reset seconds over `repeats` runs."""
  step_rates = []
  reset_times = []
  for repeat in range(repeats):
    env = make_env(players, fast_mode, repeat, needs_sticky_actions)
    try:
      env.reset()
      actions = [0] * (2 * players)
      for _ in range(20):
        _, _, done, _ = env.step(actions)
        if done:
          env.reset()
      start = time.perf_counter()
      for _ in range(steps):
        _, _, done, _ = env.step(actions)
        if done:
          env.reset()
      step_rates.append(steps / (time.perf_counter() - start))

      # Resets are timed separately: they cost far more than a step, and with a
      # synchronous vector env one straggler reset stalls every other worker.
      start = time.perf_counter()
      for _ in range(resets):
        env.reset()
      reset_times.append((time.perf_counter() - start) / resets)
    finally:
      env.close()
  return statistics.median(step_rates), statistics.median(reset_times)


def profile(steps, players, needs_sticky_actions, top):
  """Where the per-step time goes, for the fast-mode training shape."""
  env = make_env(players, True, 0, needs_sticky_actions)
  try:
    env.reset()
    actions = [0] * (2 * players)
    for _ in range(20):
      env.step(actions)

    def run():
      for _ in range(steps):
        _, _, done, _ = env.step(actions)
        if done:
          env.reset()

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
  finally:
    env.close()
  stats = pstats.Stats(profiler, stream=sys.stdout)
  stats.sort_stats('cumulative').print_stats(top)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=500)
  parser.add_argument('--repeats', type=int, default=3)
  parser.add_argument('--players', type=int, default=11,
                      help='controlled players per side; training uses 11')
  parser.add_argument('--sticky-actions',
                      action=argparse.BooleanOptionalAction, default=False,
                      help='expose sticky-action bits, which simple115v2 does '
                           'not read but cost ten engine queries per player '
                           'per step')
  parser.add_argument('--profile', action='store_true',
                      help='print a cProfile breakdown of the step loop '
                           'instead of a throughput comparison')
  parser.add_argument('--profile-top', type=int, default=25)
  args = parser.parse_args()

  if args.profile:
    profile(args.steps, args.players, args.sticky_actions, args.profile_top)
    return

  baseline, baseline_reset = measure(
      False, args.steps, args.repeats, args.players, args.sticky_actions)
  fast, fast_reset = measure(
      True, args.steps, args.repeats, args.players, args.sticky_actions)
  print(json.dumps({
      'players_per_side': args.players,
      'sticky_actions': args.sticky_actions,
      'baseline_steps_per_second': baseline,
      'fast_steps_per_second': fast,
      'speedup': fast / baseline,
      'baseline_reset_seconds': baseline_reset,
      'fast_reset_seconds': fast_reset,
      'fast_agent_steps_per_second': fast * 2 * args.players,
  }, sort_keys=True))


if __name__ == '__main__':
  main()
