"""Configs for replaying ADR lateral probes through training.

For each run (seed) of an experiment: about `--points` checkpoints spread over
training; at each, the ADR ranges and bot difficulty from the nearest `ADR`
log line at or before that epoch.  Three scenes per checkpoint, every other
setting drawn from that epoch's ranges (exactly what a probe saw):
  edge    ball_lateral pinned at the current upper edge
  beyond  pinned one step (0.02) further out
  centre  pinned at 0
Writes one ADR-log file per scene (read by attack_vs_defence.py --adr-log
with --scene adr:full) and configs.txt rows:
  LABEL CHECKPOINT DEFENCE SCENE_LOG
"""
import argparse
import glob
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--points', type=int, default=10)
parser.add_argument('--targets', default=None,
                    help='comma-separated epochs instead of --points; rows '
                         'are appended to configs_extra.txt')
args = parser.parse_args()

os.makedirs(os.path.join(args.out, 'scenes'), exist_ok=True)
rows = []
for line in open(os.path.join(args.experiment, 'jobs.txt')):
  index, _, job = line.split()[:3]
  log = glob.glob(os.path.join(args.experiment, 'repo/sbatch/logs/*{}*.out'.format(job)))[0]
  adr_lines = [json.loads(l[4:]) for l in open(log, errors='replace')
               if l.startswith('ADR {')]
  run = glob.glob('/scratch/fyy2003/experiments/football-selfplay-{}-{}'.format(
      job, index))[0]
  checkpoints = {int(re.search(r'model_(\d+)\.pt', p).group(1)): p
                 for p in glob.glob(run + '/checkpoints/*/model_*.pt')}
  last_epoch = adr_lines[-1]['epoch']
  wanted = ([int(t) for t in args.targets.split(',')] if args.targets else
            [int(last_epoch * k / args.points)
             for k in range(1, args.points + 1)])
  for target in wanted:
    epoch = min(checkpoints, key=lambda e: abs(e - target))
    adr_line = max((a for a in adr_lines if a['epoch'] <= epoch),
                   key=lambda a: a['epoch'])
    edge = adr_line['adr/ball_lateral/high']
    for kind, y in (('edge', edge), ('beyond', min(0.36, edge + 0.02)),
                    ('centre', 0.0)):
      metrics = {k: v for k, v in adr_line.items() if k.startswith('adr/')}
      metrics['adr/ball_lateral/low'] = metrics['adr/ball_lateral/high'] = y
      label = 's{}_e{:06d}_{}'.format(index, epoch, kind)
      path = os.path.join(args.out, 'scenes', label + '.log')
      with open(path, 'w') as f:
        f.write('ADR ' + json.dumps(metrics) + '\n')
      rows.append('{} {} bot:{} {}'.format(
          label, checkpoints[epoch], adr_line['bot/difficulty'], path))
name = 'configs_extra.txt' if args.targets else 'configs.txt'
with open(os.path.join(args.out, name), 'w') as f:
  f.write('\n'.join(rows) + '\n')
print(len(rows), 'configs')
