"""Button-confirmed kicks in real probe scenes (autopsy2: side_trace.py with
kick detection from button presses).

1. Lane choice: for each episode's first kick, which lanes would have scored
   from where it was struck (straight: |y| inside the posts; diagonal: the
   ~37-degree kick's crossing inside the posts), and which lane the policy
   took (the direction held when the button was pressed).
2. Timeouts: after the first kick, how far the ball got toward the goal
   line, and how many more kicks followed.
Prints tables; run with uv run --no-project --with numpy python analyze_kicks.py
"""
import collections
import glob
import json
import os
import re

import numpy as np

# Data folders (side/, proof/, history/, autopsy2/, logs/) and plots/ sit
# next to scripts/, in the analysis folder.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POST = 0.044
Y_TO_X = 83.6 / 54.4
DIAGONAL = 37.0


def lanes_that_score(x, y):
  room = 1.0 - x
  straight = abs(y) < POST
  diagonal = abs(abs(y) - room * np.tan(np.radians(DIAGONAL)) / Y_TO_X) < POST
  return {(True, True): 'both', (True, False): 'straight only',
          (False, True): 'diagonal only', (False, False): 'neither'}[
              (straight, diagonal)]


def lane_taken(held, y):
  toward = 'top' if y > 0 else 'bottom'
  if held == 'right':
    return 'straight'
  if held == toward + '_right':
    return 'diagonal in'
  if held == 'none':
    return 'no direction'
  return 'other'


def main():
  cells = {}
  for path in glob.glob(os.path.join(HERE, 'autopsy2', '*.json')):
    m = re.match(r'autopsy_(s\d)_e(\d+)_(centre|edge|beyond)$',
                 os.path.basename(path)[:-5])
    cells[m.groups()] = json.load(open(path))
  for kind in ('centre', 'edge', 'beyond'):
    episodes = [e for (s, ep, k), c in cells.items() if k == kind
                for e in c['episodes']]
    kicked = [e for e in episodes if e['first_kick']]
    print('\n== {}: {} episodes, {} with a button-confirmed kick ({:.0%}), '
          'goal rate {:.2f}'.format(kind, len(episodes), len(kicked),
                                    len(kicked) / len(episodes),
                                    np.mean([e['end'] == 'goal'
                                             for e in episodes])))
    table = collections.defaultdict(lambda: collections.Counter())
    goals = collections.defaultdict(list)
    for e in kicked:
      k = e['first_kick']
      need = lanes_that_score(k['x'], k['y'])
      took = lane_taken(k['held'], k['y'])
      table[need][took] += 1
      goals[(need, took)].append(e['end'] == 'goal')
    print('  {:15s} {:>6s} | {}'.format('lanes that score', 'kicks',
                                        'lane taken (goal rate)'))
    for need in ('straight only', 'both', 'diagonal only', 'neither'):
      row = table[need]
      total = sum(row.values())
      if not total:
        continue
      parts = ['{} {:.0%} ({:.2f})'.format(took, n / total,
                                           np.mean(goals[(need, took)]))
               for took, n in row.most_common()]
      print('  {:15s} {:6d} | {}'.format(need, total, '; '.join(parts)))
    print('  buttons:', collections.Counter(
        e['first_kick']['button'] for e in kicked).most_common())
    timeouts = [e for e in kicked if e['end'] == 'timeout']
    if timeouts:
      reach = []
      more = []
      for e in timeouts:
        k = e['first_kick']
        after = np.asarray(e['path'][k['step']:k['step'] + 40])
        reach.append(after[:, 0].max())
        more.append(len(e['kicks']) - 1)
      reach = np.asarray(reach)
      on_target = np.mean([abs(e['first_kick']['aim_y'] or 9) < POST
                           for e in timeouts])
      print('  timeouts with a kick: {} | first kick aimed on target {:.0%} | '
            'ball within 40 steps reaches x >= 0.99: {:.0%}, median max x '
            '{:.3f} | further kicks median {}'.format(
                len(timeouts), on_target, np.mean(reach >= 0.99),
                np.median(reach), int(np.median(more))))
    no_kick = [e for e in episodes if not e['first_kick']]
    print('  episodes with no kick at all: {} ({}), ends {}'.format(
        len(no_kick), '{:.0%}'.format(len(no_kick) / len(episodes)),
        collections.Counter(e['end'] for e in no_kick).most_common()))


if __name__ == '__main__':
  main()
