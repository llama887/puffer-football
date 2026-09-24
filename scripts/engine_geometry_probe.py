"""Where do the ball and the goal-side blockers ACTUALLY stand in the engine?

Every earlier probe read positions out of the scenario config, i.e. the
numbers the scenario asked for.  This reads them back from the running engine
(raw observations are absolute pitch coordinates) per template and attack
direction, for level 0, 1 and 3, so a team-coordinate convention mismatch
between the scenario builder and the engine would show up as a defender that
is not where the scenario put it.
"""
import numpy as np
from gfootball.env.puffer_env import FootballPufferEnv
from gfootball.curriculum import ADVANTAGE_ENV_NAME, ADVANTAGE_LEVELS

POST = 0.044 * (36.0 / 0.42) / 36.0  # goal half-width in observation units (0.044 obs)

for level in (0, 1, 3):
  print('=== level', level)
  rows = {}
  for seed in range(0, 64):
    env = FootballPufferEnv(env_name=ADVANTAGE_ENV_NAME, frame_stack=1, seed=seed,
                            curriculum_levels=ADVANTAGE_LEVELS, curriculum_evaluation=True)
    env._curriculum_level = level
    try:
      env.reset()
      raw = env._env.unwrapped._env.observation()
      cfg = env._env.unwrapped._config
      tmpl = int(cfg._values['curriculum_episode_template'])
      goalside = int(cfg._values['curriculum_goalside_defenders'])
      ball = np.asarray(raw['ball'])[:2]
      left = np.asarray(raw['left_team']); right = np.asarray(raw['right_team'])
      attacking_left = env._attacking_left
      defence = right if attacking_left else left
      attack = left if attacking_left else right
      goal_x = 1.0 if attacking_left else -1.0
      # blockers are defence indices 1..goalside (rank order in the scenario)
      blockers = defence[1:1 + goalside]
      key = (tmpl, 'L' if attacking_left else 'R')
      if key in rows:
        continue
      rows[key] = True
      rel = ' '.join('(%+.3f,%+.3f)' % (goal_x * (b[0] - ball[0]), b[1] - ball[1]) for b in blockers)
      inmouth = ' '.join('%s' % ('IN' if abs(b[1]) < 0.044 else 'out') for b in blockers)
      keeper = defence[0]
      carrier = attack[1]
      print('t%d %s ball=(%+.3f,%+.3f) n=%d blockers rel ball (fwd,y)=%s mouth=%s | keeper y=%+.3f carrier rel=(%+.3f,%+.3f)' % (
          tmpl, key[1], ball[0], ball[1], goalside, rel, inmouth, keeper[1],
          goal_x * (carrier[0] - ball[0]), carrier[1] - ball[1]))
    finally:
      env.close()
    if len(rows) >= 16:
      break
