"""Deep dive at the ADR lateral edge (0.3/0.6 runs, 2026-10-16).

Inputs in this directory:
  side/           side_trace.py, policy finals, clean scenes, d 0.1 and 0.3
  proof/          side_trace.py results_proof: scripted DribbleThenShoot
                  (script_r*), policy finals at d 0.15 / 0.2 (s*), and the
                  edge-probe autopsy (autopsy_s<seed>_e<epoch>_<kind>)
  logs/*.out      ADR lines of the three runs
Writes plots/deep_*.png and prints the geometry numbers.
  uv run --no-project --with matplotlib --with numpy python plot_deep.py
"""
import collections
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_side import (GRID, INK, INK_2, OUTCOMES, OUTCOME_LABELS, POST,
                       SEEDS, SLOTS, SURFACE, Y_TO_X, outcome)

# Data folders (side/, proof/, history/, autopsy2/, logs/) and plots/ sit
# next to scripts/, in the analysis folder.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'plots')
YS = [0.00, 0.04, 0.08, 0.10, 0.12, 0.14, 0.17, 0.20]
DS = [0.10, 0.15, 0.20, 0.30]
# Measured diagonal kick of the policies (median 35-40 degrees, mostly long
# passes) and an exact 45-degree shot.
POLICY_DIAGONAL = 37.0
SHOT_DIAGONAL = 45.0


def reach(d, degrees):
  """Widest start |y| from which one straight-or-diagonal kick from d out
  still crosses the goal line inside the posts."""
  return POST + d * np.tan(np.radians(degrees)) / Y_TO_X


def load(folder, pattern):
  cells = {}
  for path in glob.glob(os.path.join(HERE, folder, '*.json')):
    m = re.match(pattern, os.path.basename(path)[:-5])
    if m:
      cells[m.groups()] = json.load(open(path))
  return cells


def adr_rows():
  rows = {}
  for s, job in zip(SEEDS, ('19820209', '19820210', '19820211')):
    log = glob.glob(os.path.join(HERE, 'logs', '*{}*.out'.format(job)))[0]
    rows[s] = [json.loads(l[4:]) for l in open(log)]
  return rows


def geometry_numbers(rows):
  rng = np.random.default_rng(0)
  print('Share of scene space where no single straight/diagonal kick scores')
  for name, degrees in (('policy ~37 deg kick', POLICY_DIAGONAL),
                        ('exact 45 deg shot', SHOT_DIAGONAL)):
    y = rng.uniform(0, 0.36, 200000)
    d = rng.uniform(0.05, 1.0, 200000)
    print('  {:20s} full ADR limits (|y|<=0.36, d 0.05-1): {:.1%}'.format(
        name, np.mean(y > reach(d, degrees))))
  for s, rs in rows.items():
    edge_need, train_need = [], []
    for r in rs:
      y = rng.uniform(r['adr/ball_lateral/low'], r['adr/ball_lateral/high'], 5000)
      d = rng.uniform(r['adr/ball_distance/low'], r['adr/ball_distance/high'], 5000)
      train_need.append(np.mean(np.abs(y) > reach(d, POLICY_DIAGONAL)))
      edge = max(abs(r['adr/ball_lateral/low']), abs(r['adr/ball_lateral/high']))
      edge_need.append(np.mean(edge > reach(d, POLICY_DIAGONAL)))
    print('  seed {}: training scenes needing more than one kick {:.2%}; '
          'lateral-edge probes needing it {:.1%}'.format(
              s[1], np.mean(train_need), np.mean(edge_need)))


def plot_map(rows, script, policy):
  fig, ax = plt.subplots(figsize=(9, 5.2))
  d = np.linspace(0.05, 0.6, 200)
  ax.fill_between(d, 0, POST, color=SLOTS[0], alpha=0.25, lw=0)
  ax.fill_between(d, POST, reach(d, POLICY_DIAGONAL), color=SLOTS[2],
                  alpha=0.25, lw=0)
  ax.plot(d, reach(d, SHOT_DIAGONAL), color=INK_2, lw=1, ls=':')
  ax.text(0.45, 0.02, 'straight kick scores', color=INK, fontsize=9)
  ax.text(0.45, reach(0.45, POLICY_DIAGONAL) - 0.07,
          'diagonal kick scores\n(~37°, as the policies kick)', color=INK,
          fontsize=9)
  ax.text(0.37, reach(0.37, SHOT_DIAGONAL) + 0.01, 'exact 45° shot',
          color=INK_2, fontsize=8)
  ax.text(0.07, 0.31, 'no single kick scores:\nmust dribble in or pass',
          color=INK, fontsize=9)
  for s, rs in rows.items():
    last = rs[-1]
    lat = max(abs(last['adr/ball_lateral/low']), abs(last['adr/ball_lateral/high']))
    ax.add_patch(plt.Rectangle(
        (last['adr/ball_distance/low'], 0),
        last['adr/ball_distance/high'] - last['adr/ball_distance/low'], lat,
        fill=False, ec=SLOTS[1], lw=1.5,
        label='final ADR training range' if s == ', s0'[1:] or s == 's0' else None))
  # Measured: where the policy (idle defenders) scores at least half the time.
  for (seed, y, dd, defence), cell in policy.items():
    if seed != 's1' or defence != 'idle':
      continue
    rate = cell['summary']['goal_rate']
    ax.scatter(float(dd), float(y), s=40, marker='o',
               color=SLOTS[0] if rate >= 0.5 else SURFACE,
               edgecolors=SLOTS[0], linewidths=1.2, zorder=3)
  ax.scatter([], [], s=40, color=SLOTS[0], label='policy (seed 1) scores ≥ 50%')
  ax.scatter([], [], s=40, color=SURFACE, edgecolors=SLOTS[0],
             label='policy scores < 50%')
  ax.set_xlabel('ball distance from the goal line')
  ax.set_ylabel('ball |y| (posts 0.044)')
  ax.set_xlim(0.05, 0.6)
  ax.set_ylim(0, 0.36)
  ax.legend(loc='upper right', fontsize=8)
  fig.suptitle('What one kick can reach, and where training puts the ball',
               x=0.01, ha='left', fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'deep_1_map.png'), dpi=140)
  plt.close(fig)


def grid(cells, key_of):
  """goal rate on YS x DS, averaged over whatever seeds each cell has."""
  table = np.full((len(YS), len(DS)), np.nan)
  for i, y in enumerate(YS):
    for j, d in enumerate(DS):
      rates = [c['summary']['goal_rate'] for k, c in cells.items()
               if key_of(k) == (y, d)]
      if rates:
        table[i, j] = np.mean(rates)
  return table


def plot_heatmaps(policy, script):
  fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
      'blue', ['#f4f8fd', '#2a78d6', '#0d3a73'])
  for row, (name, cells) in enumerate((('trained policy (3 finals)', policy),
                                       ('scripted dribble-then-shoot', script))):
    for col, defence in enumerate(('idle', 'bot0.2')):
      ax = axes[row][col]
      table = grid({k: v for k, v in cells.items() if k[-1] == defence},
                   lambda k: (float(k[1]), float(k[2])))
      ax.imshow(table, origin='lower', cmap=cmap, vmin=0, vmax=1,
                aspect='auto')
      for i in range(len(YS)):
        for j in range(len(DS)):
          if not np.isnan(table[i, j]):
            ax.text(j, i, '{:.2f}'.format(table[i, j]), ha='center',
                    va='center', fontsize=8,
                    color=SURFACE if table[i, j] > 0.55 else INK)
      # One-kick reach (policy diagonal) as a step line over the grid.
      edge_rows = [np.searchsorted(YS, reach(d, POLICY_DIAGONAL)) - 0.5
                   for d in DS]
      ax.step(np.arange(len(DS)), edge_rows, where='mid', color=SLOTS[1],
              lw=2)
      ax.set_title('{}, {}'.format(name, 'idle defenders' if defence ==
                                   'idle' else 'bot 0.2'), fontsize=10)
      ax.set_xticks(range(len(DS)), [str(d) for d in DS])
      ax.set_yticks(range(len(YS)), ['{:.2f}'.format(y) for y in YS])
      ax.grid(False)
      if row == 1:
        ax.set_xlabel('ball distance from the goal line')
      if col == 0:
        ax.set_ylabel('ball start |y|')
  fig.suptitle('Goal rate by start position. Orange line: above it, no single '
               'straight/diagonal kick scores', x=0.01, ha='left', fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'deep_2_heatmaps.png'), dpi=140)
  plt.close(fig)


def plot_autopsy(autopsy):
  kinds = (('centre', 'ball centred'), ('edge', 'at the lateral edge'),
           ('beyond', 'one step beyond'))
  fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                           gridspec_kw={'width_ratios': [1.2, 1]})
  ax = axes[0]
  x = np.arange(len(kinds))
  bottom = np.zeros(len(kinds))
  for i, name in enumerate(OUTCOMES):
    shares = []
    for kind, _ in kinds:
      counts = collections.Counter(
          outcome(e) for (s, ep, k), c in autopsy.items() if k == kind
          for e in c['episodes'])
      shares.append(counts[name] / max(1, sum(counts.values())))
    ax.bar(x, shares, 0.6, bottom=bottom, color=SLOTS[i], edgecolor=SURFACE,
           linewidth=1.5, label=OUTCOME_LABELS[name])
    bottom += np.asarray(shares)
  ax.set_xticks(x, [label for _, label in kinds])
  ax.set_ylabel('share of probe episodes')
  ax.grid(axis='x', visible=False)
  ax.set_title('How real probe scenes end (all checkpoints, 3 seeds, vs the '
               'bot of the time)', fontsize=10)
  ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
  # Failed edge probes: did the scene need more than one kick?
  ax = axes[1]
  labels, needed, possible = [], [], []
  for kind, label in kinds:
    fails = [e for (s, ep, k), c in autopsy.items() if k == kind
             for e in c['episodes'] if e['end'] != 'goal']
    need = [abs(e['values']['ball_lateral']) >
            reach(e['values']['ball_distance'], POLICY_DIAGONAL) for e in fails]
    labels.append(label)
    needed.append(np.mean(need) if need else np.nan)
    possible.append(1 - np.mean(need) if need else np.nan)
  ax.bar(x, possible, 0.6, color=SLOTS[2], edgecolor=SURFACE, linewidth=1.5,
         label='one kick could have scored')
  ax.bar(x, needed, 0.6, bottom=possible, color=SLOTS[6], edgecolor=SURFACE,
         linewidth=1.5, label='needed a dribble or pass')
  ax.set_xticks(x, labels)
  ax.set_ylabel('share of failed probe episodes')
  ax.grid(axis='x', visible=False)
  ax.set_title('Failed probes: was the scene beyond one-kick reach?',
               fontsize=10)
  ax.legend(loc='lower right', fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'deep_3_autopsy.png'), dpi=140)
  plt.close(fig)


def plot_autopsy_epochs(autopsy):
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
  for ax, s in zip(axes, SEEDS):
    epochs = sorted({ep for (ss, ep, k) in autopsy if ss == s})
    for i, name in enumerate(('goal', 'wide', 'keeper', 'defender',
                              'timeout')):
      slot = OUTCOMES.index(name)
      share = []
      for ep in epochs:
        cell = autopsy.get((s, ep, 'edge'))
        if cell is None:
          share.append(np.nan)
          continue
        ends = [outcome(e) for e in cell['episodes']]
        share.append(ends.count(name) / len(ends))
      ax.plot([int(e) for e in epochs], share, color=SLOTS[slot], marker='o',
              ms=4, label=OUTCOME_LABELS[name])
    ax.set_title('seed {}: how edge probes end over training'.format(s[1]),
                 fontsize=10)
    ax.set_xlabel('epoch')
  axes[0].set_ylabel('share of edge-probe episodes')
  axes[0].legend(fontsize=8, loc='upper right')
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'deep_4_autopsy_epochs.png'), dpi=140)
  plt.close(fig)


if __name__ == '__main__':
  rows = adr_rows()
  geometry_numbers(rows)
  policy = load('side', r'(s\d)_y([\d.]+)_d([\d.]+)_(idle|bot0\.2)$')
  policy.update(load('proof', r'(s\d)_y([\d.]+)_d([\d.]+)_(idle|bot0\.2)$'))
  script = load('proof', r'script_(r\d)_y([\d.]+)_d([\d.]+)_(idle|bot0\.2)$')
  autopsy = load('proof', r'autopsy_(s\d)_e(\d+)_(centre|edge|beyond)$')
  print('cells: policy', len(policy), 'script', len(script), 'autopsy',
        len(autopsy))
  plot_map(rows, script, policy)
  plot_heatmaps(policy, script)
  if autopsy:
    plot_autopsy(autopsy)
    plot_autopsy_epochs(autopsy)
  print('wrote', sorted(p for p in os.listdir(OUT) if p.startswith('deep')))
