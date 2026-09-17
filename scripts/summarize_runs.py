#!/usr/bin/env python3
"""Compare curriculum progress across seeds from training logs.

Every promotion check writes one `PROMOTION {...}` JSON line.  This reads any
number of Slurm logs (one per seed) and prints, per run, when each level was
cleared and the latest gate numbers, then the across-seed view: how many
seeds cleared each level and the median epoch at which they did.  Single-seed
comparisons on this task are inside the noise, so use this over three or more.

    python scripts/summarize_runs.py sbatch/logs/football-selfplay-1234-*.out
"""

import argparse
import json
import statistics
import sys


def read_promotions(path):
  rows = []
  with open(path, errors='replace') as handle:
    for line in handle:
      marker = line.find('PROMOTION {')
      if marker >= 0:
        rows.append(json.loads(line[marker + len('PROMOTION '):]))
  return rows


def summarize(path, rows):
  cleared = {}
  epoch = 0
  interval = None
  for index, row in enumerate(rows):
    if interval is None and index == 1:
      interval = row['epochs_on_level'] - rows[0]['epochs_on_level']
    if row['advanced']:
      level = int(row['level'])
      # epochs_on_level counts from level entry; accumulate for a run clock.
      epoch += row['epochs_on_level']
      cleared[level] = epoch
  last = rows[-1] if rows else {}
  return {
      'path': path,
      'evaluations': len(rows),
      'cleared': cleared,
      'final_level': int(last.get('level', -1)),
      'epochs_on_final_level': int(last.get('epochs_on_level', 0)),
      'gate': last.get('gate', 'selfplay'),
      'success': last.get('promotion_success_rate'),
      'worst': last.get('promotion_worst_template_success_rate'),
      'greedy_success': last.get('greedy_promotion_success_rate'),
      'selfplay_success': last.get('selfplay_promotion_success_rate'),
      'entropy': last.get('promotion_policy_entropy_fraction'),
      'max_probability': last.get('promotion_policy_max_probability'),
      'eval_seconds': last.get('promotion_wall_seconds'),
  }


def _fmt(value, digits=3):
  if value is None:
    return '   -  '
  return '{:6.{}f}'.format(value, digits)


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('logs', nargs='+')
  args = parser.parse_args(argv)
  summaries = []
  for path in args.logs:
    rows = read_promotions(path)
    if not rows:
      print('{}: no PROMOTION lines'.format(path), file=sys.stderr)
      continue
    summaries.append(summarize(path, rows))
  if not summaries:
    return 1

  print('per run (epochs at which each level was cleared; gate numbers are '
        'from the latest evaluation)')
  print('{:<44} {:>5} {:>6} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6}  cleared'
        .format('log', 'level', 'evals', 'gate', 'worst', 'greedy',
                'selfpl', 'entr', 'maxp'))
  for row in summaries:
    name = row['path'][-44:]
    cleared = ' '.join('L{}@{}'.format(level, epoch)
                       for level, epoch in sorted(row['cleared'].items()))
    print('{:<44} {:>5} {:>6} {} {} {} {} {} {}  {}'.format(
        name, row['final_level'], row['evaluations'],
        _fmt(row['success']), _fmt(row['worst']),
        _fmt(row['greedy_success']), _fmt(row['selfplay_success']),
        _fmt(row['entropy']), _fmt(row['max_probability']), cleared))

  print()
  print('across {} run(s)'.format(len(summaries)))
  levels = sorted({level for row in summaries for level in row['cleared']})
  print('{:>5} {:>8} {:>14} {:>14}'.format(
      'level', 'cleared', 'median epoch', 'epoch range'))
  for level in levels:
    epochs = sorted(row['cleared'][level] for row in summaries
                    if level in row['cleared'])
    print('{:>5} {:>4}/{:<3} {:>14} {:>14}'.format(
        level, len(epochs), len(summaries), int(statistics.median(epochs)),
        '{}-{}'.format(epochs[0], epochs[-1])))
  finals = [row['final_level'] for row in summaries]
  print('final level: min {} median {} max {}'.format(
      min(finals), statistics.median(finals), max(finals)))
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
