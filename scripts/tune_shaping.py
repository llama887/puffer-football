#!/usr/bin/env python3
"""Bounded, paired-seed shaping screen using the existing Slurm trainer.

Run 8 configurations x 3 seeds at 50M agent steps each. After training,
evaluate on identical level 4/6/7 scenarios against immutable common frozen
opponents, with shaping off and no early abort. Rank only completed studies
by held-out goal success, not shaped returns or run-specific promotion gates.
The study directory must contain a source snapshot in repo/ and frozen
opponents in opponents/level{4,6,7}.pt. Submit sbatch/tune_shaping.sbatch as
--array=0-23%3 and set SHAPING_STUDY to that directory.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


# Ball scale, player scale, learning rate, entropy coefficient. Only these
# change: gamma and the success objective stay fixed across comparisons.
CONFIGS = [
    (0.0, 0.0, 3e-4, 0.001),
    (1.0, 0.0, 3e-4, 0.001),
    (1.0, 0.1, 3e-4, 0.001),
    (1.0, 0.3, 3e-4, 0.001),
    (1.0, 1.0, 3e-4, 0.001),
    (0.3, 0.3, 3e-4, 0.001),
    (1.0, 0.3, 1e-4, 0.001),
    (1.0, 0.3, 3e-4, 0.003),
]
SEEDS = (20260923, 20260924, 20260925)
STEPS = 50_000_000
LEVELS = (4, 6, 7)


def configuration(index):
  """Map one Slurm task to a paired seed and explicit training overrides."""
  if not 0 <= index < len(CONFIGS) * len(SEEDS):
    raise ValueError('trial index must be in [0, 23]')
  config_id, seed_index = divmod(index, len(SEEDS))
  ball, player, learning_rate, entropy = CONFIGS[config_id]
  return config_id, dict(
      BALL_POTENTIAL=ball, PLAYER_POTENTIAL=player,
      LEARNING_RATE=learning_rate, ENT_COEF=entropy, SEED=SEEDS[seed_index],
      TOTAL_TIMESTEPS=STEPS, GAMMA=0.997, GAE_LAMBDA=0.98,
      VF_COEF=0.5, CLIP_COEF=0.2, UPDATE_EPOCHS=4,
      BPTT_HORIZON=32, MINIBATCH_SEGMENTS=32, HIDDEN_SIZE=256,
      FRAME_STACK=1, START_LEVEL=0, ANNEAL_LR=0,
      FROZEN_DEFENCE_GATE=1, SCORED_PROMOTION_LEVELS=21,
      PROMOTION_INTERVAL=100, PROMOTION_EPISODES=256,
      PROMOTION_EARLY_ABORT_MARGIN=0.2,
      GREEDY_PROMOTION_EPISODES=64, SELFPLAY_PROMOTION_EPISODES=64,
      ENV_NAME='11_vs_11_advantage')


def run_trial(study, index, steps=STEPS, episodes=256):
  """Train once, then compare the final checkpoint on a shared benchmark.

  Reuse training and evaluation implementations to preserve observation and
  recurrent-window contracts. Failures leave no result file, so incomplete
  trials cannot silently enter the ranking. All candidates see the same
  evaluation seeds, levels, and frozen opponents; their training seeds differ.
  """
  import torch
  from gfootball.examples.train_puffer import (
      FootballPolicy, _make_promotion_env, build_parser, evaluate_promotion)
  from types import SimpleNamespace

  config_id, config = configuration(index)
  config['TOTAL_TIMESTEPS'] = steps
  config['PROMOTION_WORKERS'] = int(os.environ['SLURM_CPUS_PER_TASK']) - 2
  repo = study / 'repo'
  env = dict(os.environ, **{k: str(v) for k, v in config.items()})
  env.update(REPO=str(repo), WANDB_GROUP=study.name,
             WANDB_TAG='config-{}'.format(config_id))
  print('TUNING_CONFIG ' + json.dumps(config), flush=True)
  subprocess.run(['bash', str(repo / 'sbatch/train_selfplay.sbatch')],
                 env=env, check=True)
  run_root = Path('/scratch/fyy2003/experiments') / (
      'football-selfplay-{}-{}'.format(os.environ['SLURM_ARRAY_JOB_ID'], index))
  checkpoints = list((run_root / 'checkpoints').glob('*/model_*.pt'))
  if not checkpoints:
    raise RuntimeError('training finished without a model checkpoint')
  checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
  state_path = checkpoint.parent / 'trainer_state.pt'
  state = torch.load(state_path, map_location='cpu', weights_only=False)
  if state['global_step'] < steps:
    raise RuntimeError('checkpoint did not reach the common training budget')
  args = build_parser().parse_args([])
  args.env_name = '11_vs_11_advantage'
  args.curriculum_levels = 21
  args.attacker_only_levels = 0
  args.promotion_workers = int(os.environ['SLURM_CPUS_PER_TASK']) - 2
  args.promotion_episodes = episodes
  args.frame_stack = 1
  benchmark = {}
  for level in LEVELS:
    vector = _make_promotion_env(
        args, SimpleNamespace(value=level),
        frozen_defence_path=str(study / 'opponents/level{}.pt'.format(level)))
    try:
      policy = FootballPolicy(vector, hidden_size=256).to('cuda')
      policy.load_state_dict(torch.load(
          checkpoint, map_location='cuda', weights_only=True))
      metrics = evaluate_promotion(
          policy, vector, episodes, 20261999 + level, 'cuda', 32)
      benchmark[str(level)] = metrics
      print('BENCHMARK ' + json.dumps({'level': level, **metrics}), flush=True)
    finally:
      vector.close()
  result = dict(index=index, config_id=config_id, config=config,
                benchmark_episodes=episodes,
                checkpoint=str(checkpoint), global_step=int(state['global_step']),
                benchmark=benchmark)
  destination = study / 'results/{}.json'.format(index)
  destination.parent.mkdir(exist_ok=True)
  destination.write_text(json.dumps(result, indent=2) + '\n')


def summarize(study):
  """Write progress and a ranking; select a winner only when all trials finish.

  Average scoring success equally across the fixed levels and paired seeds.
  Report the weaker-scenario average and seed variability alongside it, so
  a small screening lead is not presented as conclusive improvement.
  """
  rows = [json.loads(p.read_text()) for p in (study / 'results').glob('*.json')]
  expected = len(CONFIGS) * len(SEEDS)
  complete = ({r['index'] for r in rows} == set(range(expected)) and
              all(r['config']['TOTAL_TIMESTEPS'] == STEPS and
                  r['global_step'] >= STEPS and r['benchmark_episodes'] == 256
                  for r in rows))
  ranking = []
  for config_id, values in enumerate(CONFIGS):
    trials = [r for r in rows if r['config_id'] == config_id]
    if len(trials) != len(SEEDS):
      continue
    scores = [statistics.mean(m['promotion_success_rate']
              for m in r['benchmark'].values()) for r in trials]
    ranking.append(dict(
        config_id=config_id, parameters=dict(zip(
            ('ball_potential', 'player_potential', 'learning_rate', 'ent_coef'),
            values)),
        mean_success=statistics.mean(scores), seed_success=scores,
        seed_std=statistics.stdev(scores),
        mean_worst_two=statistics.mean(
            m['promotion_worst_two_template_success_rate']
            for r in trials for m in r['benchmark'].values())))
  ranking.sort(key=lambda r: (r['mean_success'], r['mean_worst_two']), reverse=True)
  report = dict(completed=len(rows), expected=expected, complete=complete,
                ranking=ranking, selected=ranking[0] if complete else None)
  (study / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))
  return report


def main():
  """Run a single allocated trial or summarize the on-disk study results."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('run', 'summarize'))
  parser.add_argument('--study', type=Path, required=True)
  parser.add_argument('--index', type=int)
  parser.add_argument('--steps', type=int, default=STEPS)
  parser.add_argument('--benchmark-episodes', type=int, default=256)
  args = parser.parse_args()
  if args.action == 'run':
    if args.index is None:
      parser.error('run requires --index')
    if args.steps < 21120 or args.benchmark_episodes < 8:
      parser.error('need at least one rollout and eight benchmark episodes')
    run_trial(args.study.resolve(), args.index, args.steps,
              args.benchmark_episodes)
  else:
    summarize(args.study.resolve())


if __name__ == '__main__':
  main()
