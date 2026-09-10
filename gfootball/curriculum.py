"""Shared schedule for the near-goal scoring curriculum.

Every early level moves exactly one thing, so a policy that has mastered
level N starts level N+1 with most of its behaviour still worth something:

  keeper phase     one attacker lined up to shoot; the goalkeeper moves
                   from outside the post into the goal.  Measured to be
                   free, so it is only two levels wide.
  alignment phase  keeper centred, while the carrier's spawn drifts from
                   squarely behind the ball to the fully randomized gap and
                   lateral offset.  This is the one real behavioural
                   transition in the schedule, so most levels live here.
  attacker mix     a second attacker fades in by probability.
  attacker phase   three up to eleven attackers.
  defender phase   one up to ten field defenders, keeper now controllable.
  distance phase   guided near-goal spawn opens out to the full pitch.

Level 0 is deliberately the easiest possible scoring task -- open goal,
carrier aligned -- and is the anchor the whole schedule is measured against.
"""

import random

ATTACKER_ORDER = (2, 1, 10, 7, 9, 8, 3, 6, 4, 5, 0)
DEFENDER_ORDER = (4, 5, 3, 6, 8, 7, 9, 1, 10, 2)
SPAWN_TEMPLATE_COUNT = 8

# Measured: a stationary uncontrolled keeper blocks nothing, so walking it
# into the goal is free.  Level 0 keeps the open goal purely as the anchor the
# schedule is calibrated against; level 1 centres it and costs nothing.
KEEPER_LEVELS = 2

# Measured with a constant-shot reference policy: success runs 1.000 at
# alignment 0.0, 0.432 at 0.2 and 0.000 by 0.4, so the whole "line up, then
# shoot" -> "fetch the ball, turn, then shoot" transition lives inside
# [0, 0.4].  Spend levels where the difficulty actually is, not uniformly.
ALIGNMENT_SCHEDULE = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.55, 0.70, 0.85, 1.00)
ALIGNMENT_LEVELS = len(ALIGNMENT_SCHEDULE)
ATTACKER_MIX_LEVELS = 3
FIELD_ATTACKER_LEVELS = 9
DEFENDER_LEVELS = 10
DISTANCE_LEVELS = 10

KEEPER_PHASE_END = KEEPER_LEVELS
ALIGNMENT_PHASE_END = KEEPER_PHASE_END + ALIGNMENT_LEVELS
ATTACKER_MIX_PHASE_END = ALIGNMENT_PHASE_END + ATTACKER_MIX_LEVELS
ATTACKER_PHASE_END = ATTACKER_MIX_PHASE_END + FIELD_ATTACKER_LEVELS
DEFENDER_PHASE_END = ATTACKER_PHASE_END + DEFENDER_LEVELS
TOTAL_LEVELS = DEFENDER_PHASE_END + DISTANCE_LEVELS

# The defending goalkeeper only becomes a controlled agent once field
# defenders join, which is also where the defender phase starts.
ATTACKER_ONLY_LEVELS = ATTACKER_PHASE_END

# Keeper spawn offset that leaves the goal completely open.
OPEN_GOAL_KEEPER_OFFSET = 0.36

# --- advantage curriculum --------------------------------------------------
# An alternative schedule in which every one of the 22 players is active and
# playing normally at every level, and the only thing a level controls is how
# far the ball and the attacking side start in the opponent's half.  Keeping
# the player count fixed keeps the observation distribution fixed too, which
# the player-count schedule cannot do.
#
# Measured with fixed-action reference policies (96 episodes per point):
#   advantage 1.00 0.90 0.80 0.70 0.60 0.55 0.50 0.45 0.40
#   shot      1.00 0.73 0.52 0.40 0.30 0.16 0.12 0.08 0.00
#   uniform   0.38 0.21 0.10 0.14 0.08 0.06 0.05 0.02 0.03
# Smooth and monotone down to 0.45; below that a constant action cannot score
# at all and progress depends on learned ball advancing.
ADVANTAGE_ENV_NAME = '11_vs_11_advantage'
ADVANTAGE_LEVELS = 21


def advantage_for_level(level, levels=ADVANTAGE_LEVELS):
  """Advantage for a level: 1.0 at level 0 down to 0.0 at the last level."""
  levels = max(2, int(levels))
  level = max(0, min(levels - 1, int(level)))
  return max(0.0, 1.0 - level / (levels - 1))


def _clamp_level(level):
  return max(0, min(TOTAL_LEVELS - 1, int(level)))


def _ramp(index, count):
  """0.0 on a phase's first level, 1.0 on its last."""
  if count <= 1:
    return 1.0
  return min(1.0, max(0.0, index / (count - 1)))


def curriculum_state(level):
  """Return active attackers, field defenders, and distance progress."""
  level = _clamp_level(level)
  if level < ALIGNMENT_PHASE_END:
    return 1, 0, 0.0
  if level < ATTACKER_MIX_PHASE_END:
    return 2, 0, 0.0
  if level < ATTACKER_PHASE_END:
    return level - ATTACKER_MIX_PHASE_END + 3, 0, 0.0
  if level < DEFENDER_PHASE_END:
    return 11, level - ATTACKER_PHASE_END + 1, 0.0
  return 11, 10, _ramp(level - DEFENDER_PHASE_END + 1, DISTANCE_LEVELS + 1)


def curriculum_episode_duration(level):
  """Engine steps a level's episode is allowed to run.

  Episodes stay short while the scene is sparse and open out to a full match
  as the distance phase progresses.  Callers that need to budget wall clock
  (the promotion evaluation) read this rather than rediscovering the formula.
  """
  attackers, defenders, progress = curriculum_state(level)
  # Give an episode time in proportion to how crowded the scene is, so the
  # easy single-attacker levels stay short and cheap.
  near_goal_duration = min(599, 119 + 40 * (attackers - 1 + defenders))
  return int(near_goal_duration + (3000 - near_goal_duration) * progress)


def curriculum_geometry(level):
  """Return goalkeeper progress and carrier-alignment progress.

  Goalkeeper progress walks the defending keeper from outside the post
  (0.0, an open goal) to the centre of the goal (1.0).  Alignment progress
  moves the carrier's spawn from lined up behind the ball (0.0) to the full
  randomized gap and offset (1.0).
  """
  level = _clamp_level(level)
  if level < KEEPER_PHASE_END:
    return _ramp(level, KEEPER_LEVELS), 0.0
  if level < ALIGNMENT_PHASE_END:
    return 1.0, ALIGNMENT_SCHEDULE[level - KEEPER_PHASE_END]
  return 1.0, 1.0


def keeper_spawn_offset(level):
  """Lateral spawn offset of the defending goalkeeper for this level."""
  keeper_progress, _ = curriculum_geometry(level)
  return OPEN_GOAL_KEEPER_OFFSET * (1.0 - keeper_progress)


def curriculum_episode(level, seed, episode):
  """Return episode attacker count, attack direction, and spawn template."""
  level = _clamp_level(level)
  attackers, _, _ = curriculum_state(level)
  if ALIGNMENT_PHASE_END <= level < ATTACKER_MIX_PHASE_END:
    # Fade the second attacker in rather than adding it in one step.  Use an
    # integer draw: the last mix level must be *always* two attackers, so the
    # step up to three is one player, and a float share of 1.0 compared with
    # random() is a needlessly delicate way to say that.
    reached = level - ALIGNMENT_PHASE_END + 1
    rng = random.Random((int(seed) + 1) * 1000003 + int(episode))
    attackers = 2 if rng.randrange(ATTACKER_MIX_LEVELS) < reached else 1
  cycle = int(seed) + int(episode)
  return attackers, (cycle // SPAWN_TEMPLATE_COUNT) % 2 == 0, (
      cycle % SPAWN_TEMPLATE_COUNT)
