"""Why the ADR lateral edge never moves: plots from the side traces and the
probe replays (football-adr-bot-ladder-20261014, 0.3/0.6 runs).

Inputs (copied from Torch into this directory):
  side/*.json      side_trace.py: final checkpoints, clean start scene with
                   ball_lateral y and ball_distance d, idle or bot 0.2 defence
  history/*.json   attack_vs_defence.py summaries: a checkpoint at epoch E,
                   scenes from that epoch's ADR ranges, lateral pinned at the
                   edge, one step beyond, or 0 (s<seed>_e<epoch>_<kind>)
  logs/*.out       the training logs (ADR lines: ranges, bot, epoch)
Writes plots/side_*.png.  Run:
  uv run --no-project --with matplotlib --with numpy python plot_side.py
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

# Data folders (side/, proof/, history/, autopsy2/, logs/) and plots/ sit
# next to scripts/, in the analysis folder.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'plots')
os.makedirs(OUT, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
         '#4a3aa7']
POST = 0.044
OUTCOMES = ['goal', 'wide', 'corner', 'touchline', 'keeper', 'defender',
            'timeout']
OUTCOME_LABELS = {
    'goal': 'goal', 'wide': 'wide (out over goal line)',
    'corner': 'corner (defence put it out)', 'touchline': 'out at touchline',
    'keeper': 'keeper got it', 'defender': 'defender won it',
    'timeout': 'still in play at timeout'}
SEEDS = ['s0', 's1', 's2']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'text.color': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
    'legend.frameon': False, 'lines.linewidth': 2})


def outcome(episode):
  if episode['end'] in ('goal', 'wide', 'corner', 'touchline'):
    return episode['end']
  if episode['keeper_had']:
    return 'keeper'
  if episode['defender_had']:
    return 'defender'
  return 'timeout'


def load_side():
  cells = {}
  for path in glob.glob(os.path.join(HERE, 'side', '*.json')):
    name = os.path.basename(path)[:-5]
    match = re.match(r'(s\d)_y(-?[\d.]+)_d([\d.]+)_(.+)$', name)
    # The mirrored runs (y < 0) only checked symmetry; the plots use y >= 0.
    if match is None or float(match.group(2)) < 0:
      continue
    seed, y, d, defence = match.groups()
    cells[(seed, float(y), float(d), defence)] = json.load(open(path))
  return cells


def toward_centre(action, y):
  """+1 toward the goal centre (y -> 0), -1 away, 0 neither; y > 0 is
  'bottom' in the engine frame, so 'top*' moves toward the centre."""
  if y < 0.01:
    return None
  if action in ('top', 'top_right', 'top_left'):
    return 1
  if action in ('bottom', 'bottom_right', 'bottom_left'):
    return -1
  if action in ('right', 'left'):
    return 0
  return None


Y_TO_X = 83.6 / 54.4


def kick_angles(cells):
  """Physical angle (degrees, + toward the goal centre) of every first kick
  with an aim, from where it was struck and its line of travel."""
  angles = []
  for cell in cells.values():
    for e in cell['episodes']:
      kick = e['first_kick']
      if kick is None or kick.get('aim_y') is None:
        continue
      sign = 1.0 if kick['y'] >= 0 else -1.0
      angles.append(sign * np.degrees(np.arctan2(
          (kick['y'] - kick['aim_y']) * Y_TO_X, 1.0 - kick['x'])))
  return np.asarray(angles)


def diagonal_reach(cells, d):
  """Widest start |y| from which the policies' diagonal kick (the median
  angle of kicks between 25 and 55 degrees) still crosses inside the post
  from d out."""
  angles = kick_angles(cells)
  diagonal = np.median(angles[(angles > 25) & (angles < 55)])
  return POST + d * np.tan(np.radians(diagonal)) / Y_TO_X, diagonal


def plot_goal_rate(cells):
  ys = sorted({k[1] for k in cells})
  fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
  for ax, d in zip(axes, (0.1, 0.3)):
    ax.axvspan(0, POST, color=GRID, alpha=0.6, lw=0)
    ax.axvspan(0, 0.10, color=GRID, alpha=0.35, lw=0)
    reach, diagonal = diagonal_reach(cells, d)
    ax.axvline(reach, color=INK_2, lw=1.2, ls='--')
    ax.text(reach + 0.002, 0.92, 'widest start a {:.0f}° kick\nstill scores '
            'from ({:.3f})'.format(diagonal, reach), color=INK_2, fontsize=8)
    for slot, (defence, label) in enumerate((('idle', 'idle defenders'),
                                             ('bot0.2', 'bot 0.2'))):
      per_seed = np.array([[cells[(s, y, d, defence)]['summary']['goal_rate']
                            if (s, y, d, defence) in cells else np.nan
                            for y in ys] for s in SEEDS])
      mean = np.nanmean(per_seed, axis=0)
      ax.fill_between(ys, np.nanmin(per_seed, axis=0),
                      np.nanmax(per_seed, axis=0), color=SLOTS[slot],
                      alpha=0.15, lw=0)
      ax.plot(ys, mean, color=SLOTS[slot], marker='o', ms=5, label=label)
    ax.set_title('ball {} from the goal line'.format(d))
    ax.set_xlabel('ball start |y| (posts 0.044, touchline 0.42)')
    ax.text(0.002, 1.02, 'inside posts', color=INK_2, fontsize=8)
    ax.text(0.046, 1.02, 'trained (to 0.10)', color=INK_2, fontsize=8)
    ax.set_ylim(0, 1.08)
  axes[0].set_ylabel('goal rate (mean of 3 seeds, band = min-max)')
  axes[0].legend(loc='lower left')
  fig.suptitle('Final policies: goal rate by where the ball starts sideways',
               x=0.01, ha='left', fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'side_1_goal_rate.png'), dpi=140)
  plt.close(fig)


def plot_outcomes(cells):
  ys = sorted({k[1] for k in cells})
  fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
  for row, d in enumerate((0.1, 0.3)):
    for col, (defence, label) in enumerate((('idle', 'idle defenders'),
                                            ('bot0.2', 'bot 0.2'))):
      ax = axes[row][col]
      shares = np.zeros((len(OUTCOMES), len(ys)))
      for j, y in enumerate(ys):
        counts = collections.Counter(
            outcome(e) for s in SEEDS
            for e in cells.get((s, y, d, defence), {'episodes': []})[
                'episodes'])
        total = max(1, sum(counts.values()))
        for i, name in enumerate(OUTCOMES):
          shares[i, j] = counts[name] / total
      bottom = np.zeros(len(ys))
      x = np.arange(len(ys))
      for i, name in enumerate(OUTCOMES):
        ax.bar(x, shares[i], 0.72, bottom=bottom, color=SLOTS[i],
               edgecolor=SURFACE, linewidth=1.5,
               label=OUTCOME_LABELS[name] if (row, col) == (0, 0) else None)
        bottom += shares[i]
      ax.set_xticks(x, ['{:.2f}'.format(y) for y in ys])
      ax.set_title('ball {} from goal line, {}'.format(d, label))
      ax.grid(axis='x', visible=False)
      if row == 1:
        ax.set_xlabel('ball start |y|')
    axes[row][0].set_ylabel('share of episodes')
  fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
  fig.suptitle('How episodes end, by where the ball starts sideways '
               '(final policies, 3 seeds pooled)', x=0.01, ha='left',
               fontsize=12)
  fig.tight_layout(rect=(0, 0.07, 1, 1))
  fig.savefig(os.path.join(OUT, 'side_2_outcomes.png'), dpi=140)
  plt.close(fig)


def plot_aim(cells):
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
  rng = np.random.default_rng(0)
  for ax, d in zip(axes, (0.1, 0.3)):
    ax.axhspan(-POST, POST, color=GRID, alpha=0.8, lw=0)
    ax.plot([0, 0.21], [0, 0.21], color=INK_2, lw=1, ls='--')
    ax.text(0.205, 0.215, 'straight down the pitch', color=INK_2,
            fontsize=8, ha='right')
    ax.text(0.003, 0.034, 'goal mouth', color=INK_2, fontsize=8)
    for scored, slot, label in ((False, 1, 'no goal'), (True, 0, 'goal')):
      xs, aims = [], []
      for (s, y, dd, defence), cell in cells.items():
        if dd != d or defence != 'idle':
          continue
        for e in cell['episodes']:
          kick = e['first_kick']
          if kick is None or kick.get('aim_y') is None:
            continue
          if (e['end'] == 'goal') != scored:
            continue
          xs.append(y + rng.uniform(-0.004, 0.004))
          aims.append(np.clip(kick['aim_y'], -0.3, 0.3))
      ax.scatter(xs, aims, s=14, color=SLOTS[slot], alpha=0.55,
                 edgecolors=SURFACE, linewidths=0.4, label=label)
    ax.set_title('ball {} from the goal line, idle defenders'.format(d))
    ax.set_xlabel('ball start y')
    ax.set_ylim(-0.12, 0.25)
  axes[0].set_ylabel("first kick's aim: y where its line crosses the goal line")
  axes[0].legend(loc='upper left')
  fig.suptitle('Where the first kick is aimed (3 seeds pooled)', x=0.01,
               ha='left', fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'side_3_aim.png'), dpi=140)
  plt.close(fig)


def plot_carrier(cells):
  ys = sorted({k[1] for k in cells})
  fig, axes = plt.subplots(1, 3, figsize=(14, 4))
  for slot, d in enumerate((0.1, 0.3)):
    toward, moved, kick_step = [], [], []
    for y in ys:
      votes, shifts, steps = [], [], []
      for s in SEEDS:
        for e in cells.get((s, y, d, 'idle'), {'episodes': []})['episodes']:
          for _, cx, cy, action in e['carrier']:
            v = toward_centre(action, abs(cy))
            if v is not None:
              votes.append(v == 1)
          kick = e['first_kick']
          if kick is not None:
            shifts.append(y - abs(kick['y']))
            steps.append(kick['step'])
      toward.append(np.mean(votes) if votes else np.nan)
      moved.append(np.mean(shifts) if shifts else np.nan)
      kick_step.append(np.median(steps) if steps else np.nan)
    label = 'ball {} out'.format(d)
    axes[0].plot(ys, toward, color=SLOTS[slot], marker='o', ms=5, label=label)
    axes[1].plot(ys, moved, color=SLOTS[slot], marker='o', ms=5, label=label)
    axes[2].plot(ys, kick_step, color=SLOTS[slot], marker='o', ms=5,
                 label=label)
  axes[0].set_title('direction presses toward the centre\n'
                    '(share of up/down presses at the ball)')
  axes[0].set_ylim(0, 1.02)
  axes[1].set_title('how far the ball moved toward the centre\n'
                    'before the first kick (start |y| - kick |y|)')
  axes[1].axhline(0, color=INK_2, lw=1)
  axes[2].set_title('steps before the first kick (median)')
  for ax in axes:
    ax.set_xlabel('ball start |y|')
  axes[0].legend(loc='lower right')
  fig.suptitle('What the attacker at the ball does (idle defenders, 3 seeds '
               'pooled)', x=0.01, ha='left', fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'side_4_carrier.png'), dpi=140)
  plt.close(fig)


def plot_paths(cells):
  picks = [0.08, 0.12, 0.14, 0.20]
  fig, axes = plt.subplots(2, len(picks), figsize=(15, 6.5), sharex=True,
                           sharey=True)
  colour = {'goal': SLOTS[0], 'wide': SLOTS[1]}
  for row, d in enumerate((0.1, 0.3)):
    for col, y in enumerate(picks):
      ax = axes[row][col]
      ax.plot([1, 1], [-0.42, 0.42], color=INK_2, lw=1)
      ax.plot([1, 1], [-POST, POST], color=INK, lw=4, solid_capstyle='butt')
      cell = cells.get(('s1', y, d, 'idle'))
      if cell is None:
        continue
      for e in cell['episodes'][:25]:
        path = np.asarray(e['path'])
        ax.plot(path[:, 0], path[:, 1], lw=1.1, alpha=0.75,
                color=colour.get(e['end'], SLOTS[6]))
      ax.set_title('start y {:.2f}, {} out'.format(y, d), fontsize=10)
      ax.set_xlim(0.55, 1.03)
      ax.set_ylim(-0.3, 0.3)
      ax.invert_yaxis()
      ax.set_aspect(83.6 / 54.4)
  handles = [plt.Line2D([], [], color=SLOTS[0], label='goal'),
             plt.Line2D([], [], color=SLOTS[1], label='wide'),
             plt.Line2D([], [], color=SLOTS[6], label='other ending')]
  fig.legend(handles=handles, loc='lower center', ncol=3)
  fig.suptitle('Ball paths, seed 1 final, idle defenders (thick bar = goal '
               'mouth; physical aspect)', x=0.01, ha='left', fontsize=12)
  fig.tight_layout(rect=(0, 0.05, 1, 1))
  fig.savefig(os.path.join(OUT, 'side_5_paths.png'), dpi=140)
  plt.close(fig)


def plot_angles(cells):
  angles = kick_angles(cells)
  fig, ax = plt.subplots(figsize=(8, 3.8))
  bins = np.arange(-60, 66, 5) - 2.5
  ax.hist(np.clip(angles, -60, 60), bins=bins, color=SLOTS[0],
          edgecolor=SURFACE, linewidth=1.5)
  ax.set_xlabel('first-kick direction, degrees from straight down the pitch '
                '(+ = toward the goal centre)')
  ax.set_ylabel('kicks (all scenes, 3 seeds)')
  ax.grid(axis='x', visible=False)
  for angle, label in ((0, 'straight\n(right)'), (40, 'diagonal\n(top_right)')):
    ax.annotate(label, (angle, ax.get_ylim()[1] * 0.9), ha='center',
                color=INK_2, fontsize=9)
  fig.suptitle('Kicks go straight or diagonal, nothing in between '
               '(automatic aiming is off in ADR scenes)', x=0.01, ha='left',
               fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'side_7_kick_angles.png'), dpi=140)
  plt.close(fig)


def adr_lines(log):
  return [json.loads(l[4:]) for l in open(log, errors='replace')
          if l.startswith('ADR {')]


def plot_history():
  results = {}
  for path in glob.glob(os.path.join(HERE, 'history', '*.json')):
    name = os.path.basename(path)[:-5]
    seed, epoch, kind = re.match(r'(s\d)_e(\d+)_(\w+)$', name).groups()
    results[(seed, int(epoch), kind)] = json.load(open(path))['summary'][
        'scored']
  logs = {s: glob.glob(os.path.join(HERE, 'logs', '*{}*.out'.format(job)))[0]
          for s, job in zip(SEEDS, ('19820209', '19820210', '19820211'))}
  fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
  kinds = (('centre', 'ball centred (y = 0)'),
           ('edge', 'ball at the lateral edge (what the probe tests)'),
           ('beyond', 'one step beyond the edge'))
  for col, s in enumerate(SEEDS):
    ax = axes[0][col]
    ax.axhspan(0.3, 0.6, color=GRID, alpha=0.6, lw=0)
    ax.text(1000, 0.61, 'grow edge', color=INK_2, fontsize=8)
    ax.text(1000, 0.27, 'shrink edge', color=INK_2, fontsize=8, va='top')
    for slot, (kind, label) in enumerate(kinds):
      points = sorted((e, v) for (ss, e, k), v in results.items()
                      if ss == s and k == kind)
      if points:
        ax.plot(*zip(*points), color=SLOTS[slot], marker='o', ms=5,
                label=label)
    ax.set_ylim(0, 1)
    ax.set_title('seed {}: probe-scene goal rate vs the bot it faced'.format(
        s[1]))
    rows = adr_lines(logs[s])
    epochs = [r['epoch'] for r in rows]
    lower = axes[1][col]
    lower.plot(epochs, [r['adr/ball_lateral/high'] for r in rows],
               color=SLOTS[0], label='ball_lateral upper edge')
    lower.plot(epochs, [r['adr/ball_distance/high'] for r in rows],
               color=SLOTS[1], label='ball_distance upper edge')
    lower.plot(epochs, [r.get('bot/difficulty', np.nan) for r in rows],
               color=SLOTS[2], label='bot difficulty')
    lower.axhline(POST, color=INK_2, lw=1, ls=':')
    lower.set_xlabel('epoch')
    lower.set_ylim(0, 0.6)
  axes[0][0].set_ylabel('goal rate (64 episodes)')
  axes[1][0].set_ylabel('value')
  axes[0][0].legend(loc='upper right', fontsize=8)
  axes[1][0].legend(loc='upper right', fontsize=8)
  fig.suptitle('Through training: could the lateral edge ever have grown? '
               '(0.3/0.6 runs; band = no move)', x=0.01, ha='left',
               fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'side_6_history.png'), dpi=140)
  plt.close(fig)


if __name__ == '__main__':
  side = load_side()
  if side:
    plot_goal_rate(side)
    plot_outcomes(side)
    plot_aim(side)
    plot_carrier(side)
    plot_paths(side)
    plot_angles(side)
  if glob.glob(os.path.join(HERE, 'history', '*.json')):
    plot_history()
  print('wrote', sorted(os.listdir(OUT)))
