"""PufferLib-native interface for headless 22-player self-play."""

from functools import partial

import gymnasium
import numpy as np
import psutil
import pufferlib
import pufferlib.vector

import gfootball.env as football_env
from gfootball import adr
from gfootball.env import defence
from gfootball.env import entity_observation
from gfootball.env import football_action_set
from gfootball.env import wrappers


# Engine units converted to simple115v2's per-step coordinate system.
_RELATIVE_POSITION_MAX = np.array([
    2 * (55.0 + 2.55) / 54.4,
    2 * 36.0 / 83.6,
], dtype=np.float32)
_RELATIVE_PLAYER_STEP_MAX = np.array([
    2 * 8.0 / 10.0 / 54.4,
    2 * 8.0 / 10.0 / 83.6,
], dtype=np.float32)
_BALL_STEP_SCALE = np.array([
    45.0 / 10.0 / 54.4,
    45.0 / 10.0 / 83.6,
    45.0 / 10.0,
], dtype=np.float32)
_GOAL_HEIGHT = 2.5


def _soft_scale(values, scale):
  """Bound values without clipping when the simulator has no hard maximum."""
  values /= scale + np.abs(values)


def normalize_egocentric(observations):
  """Center simple115v2 physical features on each controlled player."""
  frames = observations.reshape(-1, 115)
  own_positions = frames[:, :22].reshape(-1, 11, 2)
  own_directions = frames[:, 22:44].reshape(-1, 11, 2)
  opponent_positions = frames[:, 44:66].reshape(-1, 11, 2)
  opponent_directions = frames[:, 66:88].reshape(-1, 11, 2)
  active = frames[:, 97:108].argmax(axis=1)
  rows = np.arange(frames.shape[0])
  ego_position = own_positions[rows, active].copy()
  ego_direction = own_directions[rows, active].copy()

  for positions, directions in (
      (own_positions, own_directions),
      (opponent_positions, opponent_directions)):
    missing = np.all(positions == -1, axis=-1)
    positions -= ego_position[:, None, :]
    positions /= _RELATIVE_POSITION_MAX
    directions -= ego_direction[:, None, :]
    directions /= _RELATIVE_PLAYER_STEP_MAX
    positions[missing] = -1
    directions[missing] = -1

  frames[:, 88:90] -= ego_position
  frames[:, 88:90] /= _RELATIVE_POSITION_MAX
  _soft_scale(frames[:, 90], _GOAL_HEIGHT)
  frames[:, 91:93] -= ego_direction
  _soft_scale(frames[:, 91:94], _BALL_STEP_SCALE)
  return observations


def sort_players_by_distance(observations):
  """Order teammates and opponents by distance from the controlled player.

  simple115v2 lists all 22 players in a fixed slot order, so one policy shared
  across 22 agents has to learn permutation invariance from data: the same
  teammate appears in a different slot depending on who is being controlled.
  Sorting each block by distance makes slot 0 always "me", slot 1 always "my
  nearest teammate", and so on, which is the same information in a form the
  network does not have to spend capacity untangling.

  Must run after normalize_egocentric, which is what makes positions relative.
  """
  frames = observations.reshape(-1, 115)
  rows = np.arange(frames.shape[0])[:, None]
  for base in (0, 44):
    positions = frames[:, base:base + 22].reshape(-1, 11, 2)
    directions = frames[:, base + 22:base + 44].reshape(-1, 11, 2)
    distance = np.linalg.norm(positions, axis=-1)
    # Absent players are the -1 sentinel; keep them last so they never
    # displace a real player from a near slot.
    distance[np.all(positions == -1, axis=-1)] = np.inf
    order = np.argsort(distance, axis=1)
    frames[:, base:base + 22] = positions[rows, order].reshape(-1, 22)
    frames[:, base + 22:base + 44] = directions[rows, order].reshape(-1, 22)
  return observations


def ball_potential(advance, scale):
  """Potential of a state for the side attacking the goal at advance +1.

  Ng, Harada and Russell (1999) shape with an estimate of V*; here it is how
  far the ball has been carried toward the attacked goal line, in the [-1, 1]
  pitch coordinate: +scale on that line, -scale on the far one.  The other
  team's potential is this at -advance, i.e. exactly its negative, so the
  two teams' shaping cancels on every step, terminal correction included,
  and the shaped game stays zero-sum like the score.  An earlier form,
  -scale * (1 - advance), differed per team only by a constant but summed to
  -2 * scale, which paid both teams together 2 * scale * (1 - gamma) per
  step and 2 * scale at every episode end.
  """
  return float(scale) * float(advance)


def potential_shaping(previous, current, gamma, terminal):
  """F(s, a, s') = gamma * Phi(s') - Phi(s), with Phi(absorbing) = 0.

  Theorem 1 of Ng et al.: a shaping reward of exactly this form, and only
  this form, leaves every optimal policy of the original MDP optimal in the
  shaped one.  Corollary 2 needs the absorbing state's potential to be zero,
  which is what the `terminal` branch does; the shaping over an episode then
  telescopes to -Phi(s_0), a constant the critic absorbs.
  """
  return (0.0 if terminal else float(gamma) * float(current)) - float(previous)


def closest_player_potential(positions, ball_position, scale):
  """Negative nearest teammate distance to the ball, in pitch-length units.

  Read absolute, unnormalised simple115v2 positions. Its x and y axes use
  different physical scales, so convert y to x-distance units before taking
  the Euclidean distance. Every present teammate, including the goalkeeper,
  is eligible; absent (-1, -1) slots are ignored. Use the ball's ground-plane
  position in the same absolute frame as the teammates. An empty team has zero
  potential. Taking the minimum makes this independent of player ordering
  and allows the identity of the nearest player to change without a bonus.
  This state potential is used only through potential_shaping, never as a
  per-step proximity reward. With zero terminal potential, the discounted
  shaping return is independent of the actions and episode outcome.
  """
  if not scale:
    return 0.0
  positions = np.asarray(positions).reshape(-1, 2)
  present = positions[~np.all(positions == -1, axis=1)]
  if not len(present):
    return 0.0
  delta = present - np.asarray(ball_position)[:2]
  distance = np.hypot(delta[:, 0], delta[:, 1] * (83.6 / 54.4))
  return -float(scale) * float(distance.min())


def settled_ball_weight(ball_speed, fade_speed):
  """How much the player potential counts for a ball moving at `ball_speed`.

  1 for a still or rolling ball, falling linearly to 0 at `fade_speed` and
  staying 0 above it; a `fade_speed` of 0 disables the fade (always 1).
  Speeds are simple115v2 ball-direction magnitudes, y in x-distance units:
  that feature runs about 18 times the ball's per-step movement, a dribbled
  ball stays under ~0.16-0.27 and a struck one is ~1 (probed 2026-10-01).

  Multiplying the nearest-player potential by this keeps it a function of the
  observed state, so the shaping stays exactly potential-based, but changes
  what a kick does to it: the kicker was at distance zero, so the potential
  was ~0 before the kick and stays ~0 while the ball flies, where the plain
  distance potential dropped as soon as the ball left the kicker's feet.  Once
  a loose ball settles far from a team the full distance counts again, so the
  team gains potential by walking to it.
  """
  if not fade_speed:
    return 1.0
  return float(np.clip(1.0 - float(ball_speed) / float(fade_speed), 0.0, 1.0))


def centralized_score_rewards(score_reward, active_mask, out=None):
  """Share the zero-sum match score with every active player on each team."""
  active_mask = np.asarray(active_mask, dtype=bool)
  if active_mask.shape != (22,):
    raise ValueError('active_mask must have shape (22,)')
  score_reward = float(score_reward)
  rewards = np.empty(22, dtype=np.float32) if out is None else out
  rewards[:11] = score_reward
  rewards[11:] = -score_reward
  rewards[~active_mask] = 0
  return rewards


class FootballPufferEnv(pufferlib.PufferEnv):
  """One GRF match exposed as 22 PufferLib agents.

  Every episode is an ADR scene (gfootball/adr.py): its settings are drawn
  from the ranges in `adr_shared`, the shared-memory array the trainer moves
  (or a fixed test difficulty, for evaluation pools).  Without `adr_shared`
  the ranges are private points at the start values: the start scene.
  """

  def __init__(self, env_name=adr.SCENARIO, buf=None, seed=0,
               frame_stack=4, sort_players=True, frozen_defence_path=None,
               frozen_defence_horizon=32, ball_potential_scale=0.0,
               potential_gamma=0.99, player_potential_scale=0.0,
               observation='simple115', player_potential_fade_speed=0.0,
               adr_shared=None, adr_probe_fraction=0.0,
               adr_probe_defence=defence.RANDOM):
    if frame_stack not in (1, 4):
      raise ValueError('frame_stack must be 1 or 4')
    # 'simple115': simple115v2 made egocentric (and sorted); 'entities': the
    # per-entity rows of entity_observation, built from the raw engine views.
    if observation not in ('simple115', 'entities'):
      raise ValueError("observation must be 'simple115' or 'entities'")
    if observation == 'entities' and frame_stack != 1:
      raise ValueError('entity observations need frame_stack=1')
    self._observation = observation
    # Directions to press on the first step of an episode with held players.
    self._warmup_actions = None
    if not 0 <= adr_probe_fraction <= 1:
      raise ValueError('adr_probe_fraction must be in [0, 1]')
    self._adr = adr.AutomaticDomainRandomization(shared=adr_shared)
    self._adr_probe_fraction = float(adr_probe_fraction)
    self._adr_values = None
    self._adr_probe = None
    for name, scale in (('ball_potential_scale', ball_potential_scale),
                        ('player_potential_scale', player_potential_scale),
                        ('player_potential_fade_speed',
                         player_potential_fade_speed)):
      if not np.isfinite(scale) or scale < 0:
        raise ValueError(name + ' must be finite and non-negative')
    if not 0 < potential_gamma <= 1:
      raise ValueError('potential_gamma must be in (0, 1]')
    self.num_envs = 1
    self.num_agents = 22
    self.agents_per_batch = self.num_agents
    if observation == 'entities':
      self.single_observation_space = gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(entity_observation.SIZE,),
          dtype=np.float32)
    else:
      self.single_observation_space = gymnasium.spaces.Box(
          low=-1, high=1, shape=(115 * frame_stack,), dtype=np.float32)
    self.single_action_space = gymnasium.spaces.Discrete(
        len(football_action_set.action_set_dict['default']))
    super().__init__(buf)

    self._env_name = env_name
    self._seed = int(seed)
    # Episode-level draws (ADR settings and probes) use their
    # own generator so they move nothing else.
    self._episode_rng = np.random.default_rng([self._seed, 0xad5])
    self._frame_stack = frame_stack
    self._sort_players = bool(sort_players)
    self._episode_loose_ball = False
    self._attacking_left = True
    self._active_mask = np.ones(self.num_agents, dtype=bool)
    # Scratch buffers reused by step(), so a transition allocates nothing.
    self._episode_active_mask = np.ones(self.num_agents, dtype=bool)
    self._step_actions = np.zeros(self.num_agents, dtype=np.int64)
    self._step_rewards = np.zeros(self.num_agents, dtype=np.float32)
    # Who defends when the caller does not (gfootball/env/defence.py): the
    # frozen defence plays every episode (evaluation pools), the probe
    # defence plays ADR probe episodes.  Either way the defending rows are
    # hidden from the caller, so they are never trained.
    def side(spec):
      return defence.Defence(spec, self.single_action_space.n, self._seed,
                             memory_window=frozen_defence_horizon,
                             sort_players=sort_players)
    self._frozen_defence = (None if frozen_defence_path is None
                            else side(frozen_defence_path))
    self._probe_defence = side(adr_probe_defence)
    # The defence of the current episode (None: the caller, self-play), and
    # its actions for the next step, computed with each observation.
    self._episode_defence = self._frozen_defence
    self._defence_actions = None
    # Entity mode: the engine's raw per-agent views of the latest frame, and
    # the sticky buttons each agent holds (entity_observation.press_buttons).
    self._raw_views = None
    self._sticky = np.zeros(
        (self.num_agents, entity_observation.STICKY_ACTIONS), dtype=np.float32)
    self._env = self._make_env()
    # Where a freshly built engine's episode counter starts, so an in-place
    # reseed can put it back there.
    self._initial_episode_number = self._env.unwrapped._config[
        'episode_number']
    self._episode_return = np.zeros(2, dtype=np.float32)
    self._episode_length = 0
    # Goals are far too rare to read progress from: a whole evaluation yields
    # a handful. These two move every step, so they show whether play is
    # improving before it starts converting.
    self._possession_steps = 0
    self._advance_sum = 0.0
    # Potential-based shaping on the ball's progress toward the attacked
    # goal.  The score reward stays the only thing that defines success;
    # shaping is added to the rewards the learner sees and nothing else.
    self._ball_potential_scale = float(ball_potential_scale)
    self._player_potential_scale = float(player_potential_scale)
    self._player_potential_fade_speed = float(player_potential_fade_speed)
    self._potential_gamma = float(potential_gamma)
    self._attack_potential = 0.0
    self._defence_potential = 0.0
    self._shaping_return = np.zeros(2, dtype=np.float32)

  @staticmethod
  def _ball_advance(raw_observations, attacking_left):
    """Ball x in [-1, 1], signed so +1 is the goal being attacked."""
    frame = np.asarray(raw_observations, dtype=np.float32).reshape(
        22, -1)[0, -115:]
    return float(frame[88] if attacking_left else -frame[88])

  def _set_potentials(self, raw_observations):
    """Initialise both teams' combined potentials from the new episode."""
    self._attack_potential, self._defence_potential = self._potentials(
        raw_observations)

  def _potentials(self, raw_observations):
    """Return attacker/defender potentials before egocentric normalisation.

    Row zero is the left team's absolute view: left teammates occupy 0:22,
    right teammates 44:66, and the ball 88:90. Both teams approach that same
    ball; the episode's designated attacker only changes the return order.
    Adding fixed state potentials preserves their telescoping
    property; both terms use the same discount and terminal correction.
    The player terms are weighted by settled_ball_weight of the observed
    ball speed (91:93), which is also part of the state.
    """
    frame = np.asarray(raw_observations, dtype=np.float32).reshape(
        22, -1)[0, -115:]
    advance = self._ball_advance(raw_observations, self._attacking_left)
    weight = settled_ball_weight(
        np.hypot(frame[91], frame[92] * (83.6 / 54.4)),
        self._player_potential_fade_speed)
    left = weight * closest_player_potential(
        frame[:22], frame[88:90], self._player_potential_scale)
    right = weight * closest_player_potential(
        frame[44:66], frame[88:90], self._player_potential_scale)
    attack, defence = (left, right) if self._attacking_left else (right, left)
    return (ball_potential(advance, self._ball_potential_scale) + attack,
            ball_potential(-advance, self._ball_potential_scale) + defence)

  def _make_env(self):
    return football_env.create_environment(
        env_name=self._env_name,
        representation=('raw' if self._observation == 'entities'
                        else 'simple115v2'),
        rewards='scoring',
        render=False,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        write_video=False,
        stacked=self._frame_stack == 4,
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=11,
        extra_players=None,
        other_config_options={
            'action_set': 'default',
            'fast_mode': True,
            'game_engine_random_seed': self._seed,
            # simple115v2 does not encode sticky actions, and producing them
            # costs ten engine queries per controlled player per step.
            'needs_sticky_actions': False,
            'real_time': False,
        })

  def _reset_match(self):
    self._adr_probe = None
    if self._episode_rng.random() < self._adr_probe_fraction:
      self._adr_values, index, edge = self._adr.sample_probe(
          self._episode_rng)
      self._adr_probe = (index, edge)
    else:
      self._adr_values = self._adr.sample(self._episode_rng)
    self._env.unwrapped._config['adr_values'] = self._adr_values
    self._episode_defence = (self._probe_defence if self._adr_probe is not None
                             else self._frozen_defence)
    if self._episode_defence is not None:
      self._episode_defence.start_episode()
    observations = self._from_engine(self._env.reset())
    self._sticky.fill(0)
    scene = adr.read_scene(self._env.unwrapped._config)
    self._attacking_left = scene.attacking_left
    self._episode_loose_ball = scene.loose_ball
    self._set_active_players()
    # A scene with held players (ADR held_share) presses their directions on
    # the first step.  Every row is hidden for that step, so the trainer
    # (which skips all-zero rows) never trains on actions the policy did not
    # choose.
    held = scene.held_directions
    self._warmup_actions = held if held.any() else None
    if self._warmup_actions is not None:
      self._active_mask.fill(False)
    if self._ball_potential_scale or self._player_potential_scale:
      self._set_potentials(observations)
    return observations

  def _defending_rows(self):
    return slice(11, 22) if self._attacking_left else slice(0, 11)

  def _set_active_players(self):
    self._active_mask.fill(True)
    if self._episode_defence is not None:
      # The defence is acted by this worker, never by the caller.
      self._active_mask[self._defending_rows()] = False

  def _adr_info(self):
    """Episode-end ADR fields: whether it was a probe, which setting and
    edge it tested (-1 if none) and every setting it was played with."""
    index, edge = self._adr_probe if self._adr_probe else (-1, -1)
    info = {'adr_probe': float(self._adr_probe is not None),
            'adr_parameter': float(index), 'adr_edge': float(edge)}
    for name, value in self._adr_values.items():
      info['adr_' + name] = float(value)
    return info

  def _from_engine(self, observations):
    """simple115v2 frames of what the engine returned, for this env's own use.

    Shaping, possession and ball-advance bookkeeping read simple115v2 frames
    in every mode.  In entity mode the engine returns raw per-agent views;
    they are kept for _write_observations and converted here exactly as
    the simple115v2 wrapper would.
    """
    if self._observation != 'entities':
      return observations
    self._raw_views = observations
    return wrappers.Simple115StateWrapper.convert_observation(
        observations, True)

  def _write_observations(self, observations):
    """Fill the agent rows the trainer reads, then hide inactive agents.

    `observations` are simple115v2 frames (made egocentric and sorted here);
    in entity mode the rows come from the raw views kept by _from_engine.
    """
    if self._observation == 'entities':
      entity_observation.build(self._raw_views, self._sticky,
                               out=self.observations)
    else:
      observations = np.asarray(observations, dtype=np.float32)
      if observations.shape != self.observations.shape:
        raise ValueError('Expected observations with shape {}, got {}'.format(
            self.observations.shape, observations.shape))
      self.observations[:] = observations
      normalize_egocentric(self.observations)
      if self._sort_players:
        sort_players_by_distance(self.observations)
    if self._episode_defence is not None:
      # The defence reads its rows before they are hidden below.
      defending = self._defending_rows()
      self._defence_actions = self._episode_defence.act(
          self.observations[defending],
          np.asarray(observations, dtype=np.float32).reshape(
              self.num_agents, -1)[defending])
    self.observations[~self._active_mask] = 0

  def reset(self, seed=None):
    if seed is not None:
      # Reseed in place instead of rebuilding the engine (10-20 s).  A fresh
      # engine is fully determined by these two values: the engine reseeds
      # its RNG from the scenario seed at every kickoff, and spawns depend
      # only on (seed, episode number).  This runs even for an unchanged
      # seed, so a reused evaluation pool replays the same episodes each time.
      self._seed = int(seed)
      config = self._env.unwrapped._config
      config['game_engine_random_seed'] = self._seed
      config['episode_number'] = self._initial_episode_number
      # ADR settings and the defences are drawn here, not by the engine;
      # only a reseeding reset restarts the defences' streams.
      self._episode_rng = np.random.default_rng([self._seed, 0xad5])
      for side in (self._frozen_defence, self._probe_defence):
        if side is not None:
          side.reseed(self._seed)
    self._write_observations(self._reset_match())
    self.rewards.fill(0)
    self.terminals.fill(False)
    self.truncations.fill(False)
    self._episode_return.fill(0)
    self._shaping_return.fill(0)
    self._episode_length = 0
    self._possession_steps = 0
    self._advance_sum = 0.0
    return self.observations, []

  def step(self, actions):
    # _reset_match rewrites _active_mask mid-step, so the mask that governs
    # this transition has to be snapshotted first.  Reuse the buffers rather
    # than allocating three arrays on every step.
    episode_active_mask = self._episode_active_mask
    np.copyto(episode_active_mask, self._active_mask)
    np.copyto(self._step_actions, np.asarray(actions).reshape(self.num_agents),
              casting='unsafe')
    actions = self._step_actions
    actions[~episode_active_mask] = 0
    if self._episode_defence is not None:
      actions[self._defending_rows()] = self._defence_actions
    warmup = self._warmup_actions is not None
    if warmup:
      pressed = self._warmup_actions > 0
      actions[pressed] = self._warmup_actions[pressed]
      self._warmup_actions = None
    if self._observation == 'entities':
      entity_observation.press_buttons(self._sticky, actions)
    observations, _, done, info = self._env.step(actions)
    observations = self._from_engine(observations)
    rewards = centralized_score_rewards(
        info['score_reward'], episode_active_mask, out=self._step_rewards)
    for team, team_slice in enumerate((slice(0, 11), slice(11, 22))):
      team_active = episode_active_mask[team_slice]
      if team_active.any():
        self._episode_return[team] += rewards[team_slice][team_active].mean()
    self._episode_length += 1
    frame = np.asarray(observations, dtype=np.float32).reshape(
        self.num_agents, -1)[0, -115:]
    # simple115v2: 88 is ball x, 94:97 is a none/left/right ownership one-hot.
    owner = int(np.argmax(frame[94:97]))
    attacking_owner = 1 if self._attacking_left else 2
    if owner == attacking_owner:
      self._possession_steps += 1
    # +1 means the ball is on the goal being attacked, -1 the other end.
    advance = float(frame[88] if self._attacking_left else -frame[88])
    self._advance_sum += advance
    if self._ball_potential_scale or self._player_potential_scale:
      # Shape each side from its own combined state potential. The terminal
      # state has zero potential even on timeout or a conceded goal, so the
      # discounted shaping return is always exactly -Phi(initial state).
      current_attack, current_defence = self._potentials(observations)
      attack_shaping = potential_shaping(
          self._attack_potential,
          current_attack,
          self._potential_gamma, done)
      defence_shaping = potential_shaping(
          self._defence_potential,
          current_defence,
          self._potential_gamma, done)
      attacking_team = 0 if self._attacking_left else 1
      shaping = np.zeros(self.num_agents, dtype=np.float32)
      shaping[:11] = attack_shaping if attacking_team == 0 else defence_shaping
      shaping[11:] = defence_shaping if attacking_team == 0 else attack_shaping
      shaping[~episode_active_mask] = 0
      self._shaping_return[attacking_team] += attack_shaping
      self._shaping_return[1 - attacking_team] += defence_shaping
      rewards = rewards + shaping
      if not done:
        self._attack_potential = current_attack
        self._defence_potential = current_defence
    self.rewards[:] = rewards
    self.terminals.fill(done)
    self.truncations.fill(False)

    infos = []
    if done:
      attacking_return = self._episode_return[
          0 if self._attacking_left else 1]
      infos.append({
          # The attack scored: what ADR probes and every evaluation count.
          'success': float(attacking_return > 0),
          'attacking_left': float(self._attacking_left),
          'loose_ball': float(self._episode_loose_ball),
          **self._adr_info(),
          # Identifies the match, so an asynchronous evaluator can take a
          # fixed quota of episodes from every match.
          'env_seed': float(self._seed),
          'episode_length': self._episode_length,
          'possession_fraction': (
              self._possession_steps / max(1, self._episode_length)),
          'mean_ball_advance': (
              self._advance_sum / max(1, self._episode_length)),
          'left_episode_return': float(self._episode_return[0]),
          'right_episode_return': float(self._episode_return[1]),
          'attacking_shaping_return': float(
              self._shaping_return[0 if self._attacking_left else 1]),
          'score_reward': float(info['score_reward']),
      })
      observations = self._reset_match()
      self._episode_return.fill(0)
      self._shaping_return.fill(0)
      self._episode_length = 0
      self._possession_steps = 0
      self._advance_sum = 0.0
    elif warmup:
      # The held directions are pressed; the policy takes over from here.
      self._set_active_players()
    self._write_observations(observations)
    return (self.observations, self.rewards, self.terminals,
            self.truncations, infos)

  def close(self):
    if self._env is not None:
      self._env.close()
      self._env = None


def make_vector_env(num_envs=None, num_workers=None, batch_size=None,
                    reserved_cpus=2, seed=0, envs_per_worker=1,
                    async_batch_workers=None, **env_kwargs):
  """Create headless matches across PufferLib multiprocessing workers.

  Two independent ways to stop a GRF reset (far dearer than a step) from
  stalling the whole batch; they compose:

  ``envs_per_worker`` above 1 has each worker hold several matches, so a
  worker keeps stepping its other matches while one resets.

  ``async_batch_workers`` switches PufferLib to its fully asynchronous path:
  every recv() returns whichever that many workers finished first, in any
  order, so a worker stuck in a reset delays nobody else.  The caller must
  then track agents by the ids recv() returns.
  """
  if num_workers is None:
    available_cpus = (len(psutil.Process().cpu_affinity())
                      if hasattr(psutil.Process(), 'cpu_affinity') else
                      psutil.cpu_count(logical=False) or 1)
    num_workers = max(1, available_cpus - reserved_cpus)
  envs_per_worker = max(1, int(envs_per_worker))
  num_envs = num_workers * envs_per_worker if num_envs is None else num_envs
  batch_size = num_workers if batch_size is None else batch_size
  if num_envs % num_workers:
    raise ValueError('num_envs must be a multiple of num_workers')
  if batch_size % envs_per_worker:
    raise ValueError('batch_size must be a multiple of envs_per_worker')
  is_async = async_batch_workers is not None
  if is_async:
    if not 0 < async_batch_workers < num_workers:
      raise ValueError('async_batch_workers must be between 1 and '
                       'num_workers - 1')
    batch_size = async_batch_workers * (num_envs // num_workers)
  vecenv = pufferlib.vector.make(
      partial(FootballPufferEnv, **env_kwargs),
      backend=pufferlib.vector.Multiprocessing,
      num_envs=num_envs,
      num_workers=num_workers,
      batch_size=batch_size,
      zero_copy=not is_async,
      seed=seed)
  if is_async:
    # recv() reads these per call; make() does not accept sync_traj.
    vecenv.zero_copy = False
    vecenv.sync_traj = False
  vecenv.is_async = is_async
  return vecenv
