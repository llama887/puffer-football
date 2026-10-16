"""Frame strips for rendered side_trace episodes.

argv: TRACE_JSON VIDEO_DIR OUT_DIR.  The engine writes one webm per episode
(dump_frequency=1) in episode order; they are matched to the trace's episodes
by modification time.  For every timeout episode, and the first two goals,
writes OUT_DIR/<scene>_ep<i>_<end>.png: 12 frames evenly spread over the
episode, 4 x 3, each labelled with its step.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

trace, video_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
episodes = json.load(open(trace))['episodes']
videos = sorted(glob.glob(os.path.join(video_dir, '*.webm')),
                key=os.path.getmtime)
print(len(episodes), 'episodes,', len(videos), 'videos')
scene = os.path.basename(trace)[:-5]
goals = 0
for i, (episode, video) in enumerate(zip(episodes, videos)):
  if episode['end'] == 'goal':
    goals += 1
    if goals > 2:
      continue
  elif episode['end'] != 'timeout':
    continue
  capture = cv2.VideoCapture(video)
  frames = []
  while True:
    ok, frame = capture.read()
    if not ok:
      break
    frames.append(frame)
  capture.release()
  if not frames:
    print('no frames in', video)
    continue
  picks = np.linspace(0, len(frames) - 1, 12).astype(int)
  tiles = []
  for k in picks:
    tile = cv2.resize(frames[k], (480, 270))
    step = int(round(k * episode['steps'] / max(1, len(frames) - 1)))
    cv2.putText(tile, 'step ~{}'.format(step), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(tile)
  rows = [np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)]
  sheet = np.vstack(rows)
  kicks = ', '.join('{}@{} {} aim {:.3f}'.format(
      k['button'], k['step'], k['held'], k['aim_y'] if k['aim_y'] is not None
      else float('nan')) for k in episode['kicks'][:4])
  name = '{}_ep{:02d}_{}.png'.format(scene, i, episode['end'])
  cv2.imwrite(os.path.join(out, name), sheet)
  print(name, episode['steps'], 'steps; kicks:', kicks)
