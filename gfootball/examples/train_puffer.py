#!/usr/bin/env python3
"""Recurrent PPO self-play training for the football curriculum.

One shared trunk feeds an LSTM; the actor and critic read the same recurrent
state and train under a single optimizer.  The vector env always exposes 22
agent rows, but the curriculum only controls a few of them at the early
levels, so every loss is masked down to the rows that are actually playing.
"""

import argparse
import contextlib
from collections import defaultdict
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch

import pufferlib
from pufferlib import pufferl
import pufferlib.pytorch

from gfootball.env.puffer_env import make_vector_env
from gfootball.env import football_action_set
# Re-exported: scripts and tests import the policy from this module.
from gfootball.env.puffer_policy import (  # noqa: F401
    FootballPolicy, RunningNormalizer, episode_batch_sizes, episode_pieces,
    save_policy_snapshot)
from gfootball.curriculum import (
    ADVANTAGE_ENV_NAME, ADVANTAGE_LEVELS, ATTACKER_ONLY_LEVELS,
    SPAWN_TEMPLATE_COUNT, TOTAL_LEVELS, curriculum_episode_duration)


ACTION_NAMES = tuple(
    str(action) for action in football_action_set.action_set_dict['default'])
SHOT_ACTION = ACTION_NAMES.index('shot')

# Order must match how train() stacks them; see the minibatch loop.
MINIBATCH_STAT_NAMES = (
    'old_approx_kl', 'approx_kl', 'clipfrac', 'importance',
    'policy_loss', 'value_loss', 'entropy', 'gradient_norm',
    'minibatch_transitions')


def generalized_advantages(values, rewards, terminals, gamma, gae_lambda):
  """Standard GAE for Puffer's next-step reward storage convention."""
  advantages = torch.zeros_like(rewards)
  running = torch.zeros(rewards.shape[0], device=rewards.device)
  for timestep in range(rewards.shape[1] - 2, -1, -1):
    next_timestep = timestep + 1
    next_active = ~terminals[:, next_timestep].bool()
    delta = (rewards[:, next_timestep] +
             gamma * values[:, next_timestep] * next_active -
             values[:, timestep])
    running = delta + gamma * gae_lambda * next_active * running
    advantages[:, timestep] = running
  valid = torch.ones_like(terminals, dtype=torch.bool)
  valid[:, -1] = False
  return advantages, advantages + values, valid


def cosine_learning_rate(initial, global_step, total_timesteps):
  """Cosine-annealed learning rate for a point in training, zero at the end.

  Replaces PufferLib's CosineAnnealingLR, which counts epochs and is sized as
  total_timesteps // batch_size.  batch_size is buffer room (2x for async
  collection, 3x with overlap) while an epoch trains one segment per agent,
  and overlap drops surplus segments, so no epoch count matches the run: a
  1B-step run hit zero at ~500M and, the cosine being periodic, climbed back
  to the full rate by 1B.  Keyed on agent steps done instead, the rate falls
  monotonically to exactly zero at total_timesteps and stays there.
  """
  progress = min(1.0, max(0.0, global_step / total_timesteps))
  if progress >= 1.0:
    return 0.0
  return initial * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer, value):
  """Set every param group's rate; tensor rates are filled in place.

  train() calls it before each update.  A captured update graph reads the
  rate tensor's memory on every replay, so replacing the tensor would leave
  the graph on the old value.
  """
  for group in optimizer.param_groups:
    if torch.is_tensor(group['lr']):
      group['lr'].fill_(value)
    else:
      group['lr'] = value


def normalize_advantages(advantages):
  advantages = advantages.float()
  return ((advantages - advantages.mean()) /
          advantages.std(unbiased=False).clamp_min(1e-8))


def explained_variance(predictions, targets):
  """Fraction of return variance the critic accounts for."""
  predictions = predictions.float()
  targets = targets.float()
  target_variance = targets.var(unbiased=False)
  if target_variance == 0:
    return float('nan')
  return (1 - (targets - predictions).var(unbiased=False) /
          target_variance).item()


def policy_diagnostics(logits):
  """Small policy-health signals that expose uniform or collapsed behavior."""
  probabilities = torch.softmax(logits.float(), dim=-1)
  top_two = probabilities.topk(2, dim=-1).values
  entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(-1)
  return {
      'policy_entropy_fraction': entropy.mean() / math.log(logits.shape[-1]),
      'policy_max_abs_logit': logits.float().abs().max(),
      'policy_max_probability': top_two[:, 0].mean(),
      'policy_probability_margin': (top_two[:, 0] - top_two[:, 1]).mean(),
  }


def promotion_statistics(episodes):
  """Summarize held-out success overall and for the weakest spawn."""
  by_template = defaultdict(list)
  for episode in episodes:
    by_template[int(episode['curriculum_template'])].append(
        float(episode['curriculum_success']))
  success_rate = sum(map(float, (
      episode['curriculum_success'] for episode in episodes))) / len(episodes)
  template_rates = {
      template: sum(values) / len(values)
      for template, values in by_template.items()
  }
  ranked = sorted(template_rates.values())
  metrics = {
      'promotion_success_rate': success_rate,
      'promotion_worst_template_success_rate': ranked[0] if ranked else 0.0,
      # One template is a 32-64 episode bin, so a single worst template
      # flickers around the threshold from check to check.  Averaging the two
      # weakest still refuses a policy with a hole in it while halving the
      # noise the gate has to see through.
      'promotion_worst_two_template_success_rate': (
          sum(ranked[:2]) / len(ranked[:2]) if ranked else 0.0),
      'promotion_templates_covered': float(len(template_rates)),
  }
  metrics.update({
      'promotion_template_{}_success_rate'.format(template):
      template_rates.get(template, 0.0)
      for template in range(SPAWN_TEMPLATE_COUNT)
  })
  return metrics


def promotion_min_episodes(args):
  """Episodes an evaluation must run before early abort may end it.

  Enough to cover every spawn template a few times over, so a hopeless
  verdict is never called on a handful of unlucky draws.
  """
  if args.promotion_min_episodes > 0:
    return args.promotion_min_episodes
  return max(SPAWN_TEMPLATE_COUNT * 4, args.promotion_episodes // 4)


def promotion_passes(metrics, success_threshold, worst_template_threshold):
  """Overall success and the mean of the two weakest templates both pass."""
  return (
      metrics['promotion_templates_covered'] == SPAWN_TEMPLATE_COUNT and
      metrics['promotion_success_rate'] >= success_threshold and
      metrics['promotion_worst_two_template_success_rate'] >=
      worst_template_threshold)


def evaluate_promotion(policy, vecenv, episodes, seed, device,
                       recurrent_horizon, greedy=False, early_abort=None):
  """Evaluate with the same recurrent windows as PufferLib training rollouts.

  Memory restarts every `recurrent_horizon` steps, as the training rollout
  restarts it at every segment; None keeps it for the whole episode, for a
  policy trained with carried memory.  Episode ends always clear it.
  `greedy` takes the argmax action instead of sampling, which shows whether
  the network has actually learned a shot underneath a near-uniform
  distribution.  `early_abort` is an optional `(after_episodes, min_rate)`
  pair: once that many episodes are in, a success rate below the floor ends
  the evaluation early, since it cannot pass the gate and the remaining
  episodes would only refine a number nobody acts on.
  """
  if recurrent_horizon is not None and recurrent_horizon < 1:
    raise ValueError('recurrent_horizon must be positive or None')
  observations, _ = vecenv.reset(seed=seed)
  generator = torch.Generator(device=device).manual_seed(seed)
  state = {'lstm_h': None, 'lstm_c': None, 'done': None}
  rows = []
  action_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
  # An unbiased but bounded sample of the per-step logits.  Keeping every
  # step held ~700MB on a full-length evaluation; policy_diagnostics only
  # needs a representative draw, so reservoir-sample a fixed number of them.
  logits_reservoir = []
  reservoir_capacity = 64
  # A private RNG: drawing from `generator` would shift action sampling.
  reservoir_rng = random.Random(seed)
  decisions = 0
  steps = 0
  aborted = False
  was_training = policy.training
  policy.eval()
  try:
    while len(rows) < episodes:
      if early_abort is not None and len(rows) >= early_abort[0]:
        rate = sum(float(row['curriculum_success']) for row in rows) / len(rows)
        if rate < early_abort[1]:
          aborted = True
          break
      # PuffeRL.evaluate starts every rollout window from zero memory. Keep
      # this clock independent of episode resets, just like the collector.
      if recurrent_horizon is not None and steps % recurrent_horizon == 0:
        state['lstm_h'] = state['lstm_c'] = None
      observation_tensor = torch.as_tensor(observations, device=device)
      active = observation_tensor.flatten(1).abs().sum(dim=-1) > 0
      with torch.no_grad():
        logits, _ = policy.forward_eval(observation_tensor, state)
        # `steps` is this sample's 0-based index, giving textbook reservoir
        # sampling: every step ends up equally likely to be retained.
        sampled_logits = logits[active].float().cpu()
        if len(logits_reservoir) < reservoir_capacity:
          logits_reservoir.append(sampled_logits)
        else:
          slot = reservoir_rng.randrange(steps + 1)
          if slot < reservoir_capacity:
            logits_reservoir[slot] = sampled_logits
        if greedy:
          actions = logits.argmax(dim=-1)
        else:
          actions = torch.multinomial(
              torch.softmax(logits.float(), dim=-1), 1,
              generator=generator).squeeze(-1)
      active_actions = actions[active].cpu()
      action_counts += torch.bincount(
          active_actions, minlength=len(ACTION_NAMES))
      decisions += active_actions.numel()
      observations, _, terminals, _, infos = vecenv.step(
          actions.cpu().numpy())
      steps += 1
      state['done'] = torch.as_tensor(
          np.asarray(terminals), device=device)
      for info in infos:
        if 'curriculum_success' in info and len(rows) < episodes:
          rows.append(info)
  finally:
    policy.train(was_training)
  return _promotion_metrics(rows, action_counts, decisions,
                            logits_reservoir, recurrent_horizon, greedy,
                            aborted)


def _promotion_metrics(rows, action_counts, decisions, active_logits_rows,
                       recurrent_horizon, greedy, aborted):
  metrics = promotion_statistics(rows)
  diagnostics = policy_diagnostics(torch.cat(active_logits_rows))
  action_counts = action_counts.cpu()
  decisions = max(1, int(decisions))
  metrics.update({
      # 0: memory ran through whole episodes.
      'promotion_recurrent_horizon': float(recurrent_horizon or 0),
      'promotion_greedy': float(greedy),
      'promotion_aborted': float(aborted),
      'promotion_episodes': float(len(rows)),
      'promotion_mean_episode_length': sum(
          row['episode_length'] for row in rows) / len(rows),
      'promotion_possession_fraction': sum(
          row.get('possession_fraction', 0.0) for row in rows) / len(rows),
      'promotion_mean_ball_advance': sum(
          row.get('mean_ball_advance', 0.0) for row in rows) / len(rows),
      'promotion_max_action_fraction': (
          action_counts.max().item() / decisions),
      'promotion_shot_fraction': action_counts[SHOT_ACTION].item() / decisions,
      **{
          'promotion_{}'.format(name): value.item()
          for name, value in diagnostics.items()
      },
      **{
          'promotion_action_{}_fraction'.format(name):
          action_counts[index].item() / decisions
          for index, name in enumerate(ACTION_NAMES)
      },
  })
  return metrics


def _match_gumbel_noise(seed, match_ids, match_steps, rows, actions):
  """Gumbel noise that depends only on (seed, match, that match's step).

  Sampling with argmax(logits + noise) then gives every match the same
  actions no matter which other matches share its batch or in what order
  the matches report back.
  """
  noise = np.empty((len(match_ids), rows, actions), dtype=np.float32)
  for position, (match, step) in enumerate(zip(match_ids, match_steps)):
    uniform = np.random.default_rng(
        (int(seed), int(match), int(step))).random((rows, actions))
    noise[position] = -np.log(-np.log(np.clip(uniform, 1e-12, 1 - 1e-12)))
  return noise.reshape(len(match_ids) * rows, actions)


def evaluate_promotion_async(policy, vecenv, episodes, seed, device,
                             recurrent_horizon, greedy=False,
                             early_abort=None):
  """evaluate_promotion for a pool whose matches run at their own pace.

  Every match contributes its first ceil(episodes / matches) episodes and
  nothing else.  Taking the first `episodes` to finish overall would favour
  short episodes, and at the scoring levels a short episode is a goal.

  The result does not depend on timing: each match's engine, frozen
  defence and sampling noise are seeded per match, recurrent memory
  restarts every `recurrent_horizon` of a match's own steps (never, if
  None) and at episode ends, and everything reported (episodes, action
  statistics, the early abort) is taken from a fixed number of each match's
  first episodes.
  """
  if recurrent_horizon is not None and recurrent_horizon < 1:
    raise ValueError('recurrent_horizon must be positive or None')
  if getattr(vecenv, 'envs_per_worker', 1) != 1:
    raise ValueError('async promotion needs one match per worker, so that '
                     'match i is reset with seed + i')
  per_env = vecenv.driver_env.num_agents
  matches = vecenv.num_agents // per_env
  quota = -(-episodes // matches)
  abort_quota = (None if early_abort is None else
                 min(quota, -(-early_abort[0] // matches)))
  vecenv.async_reset(seed)
  hidden = policy.hidden_size
  agent_h = torch.zeros(vecenv.num_agents, hidden, device=device)
  agent_c = torch.zeros_like(agent_h)
  match_steps = np.zeros(matches, dtype=np.int64)
  # Per match: finished episodes, and the action counts and active logits
  # of each finished episode and of the one in progress.
  taken = [[] for _ in range(matches)]
  episode_actions = [[] for _ in range(matches)]
  episode_logits = [[] for _ in range(matches)]
  current_actions = np.zeros((matches, len(ACTION_NAMES)), dtype=np.int64)
  current_logits = [[] for _ in range(matches)]
  limit = quota
  aborted = False
  was_training = policy.training
  policy.eval()
  try:
    while any(len(rows) < quota for rows in taken):
      if abort_quota is not None and all(
          len(rows) >= abort_quota for rows in taken):
        first = [row for rows in taken for row in rows[:abort_quota]]
        rate = sum(float(row['curriculum_success'])
                   for row in first) / len(first)
        if rate < early_abort[1]:
          aborted = True
          limit = abort_quota
          break
        abort_quota = None
      observations, _, terminals, _, infos, agent_ids, _ = vecenv.recv()
      agent_ids = np.asarray(agent_ids)
      match_ids = agent_ids[::per_env] // per_env
      for info in infos:
        if 'curriculum_success' not in info:
          continue
        # reset(seed) gives match i the seed `seed + i`.
        match = int(info['env_seed']) - int(seed)
        if not 0 <= match < matches:
          raise RuntimeError('episode from an unknown match')
        taken[match].append(info)
        episode_actions[match].append(current_actions[match].copy())
        episode_logits[match].append(current_logits[match])
        current_actions[match] = 0
        current_logits[match] = []
      restart = (match_ids[match_steps[match_ids] % recurrent_horizon == 0]
                 if recurrent_horizon is not None else ())
      if len(restart):
        fresh = torch.as_tensor(
            (restart[:, None] * per_env + np.arange(per_env)).ravel(),
            device=device)
        agent_h[fresh] = 0
        agent_c[fresh] = 0
      index = torch.as_tensor(agent_ids, device=device)
      observation_tensor = torch.as_tensor(observations, device=device)
      active = observation_tensor.flatten(1).abs().sum(dim=-1) > 0
      state = {'lstm_h': agent_h[index], 'lstm_c': agent_c[index],
               'done': torch.as_tensor(np.asarray(terminals), device=device)}
      with torch.no_grad():
        logits, _ = policy.forward_eval(observation_tensor, state)
        agent_h[index] = state['lstm_h']
        agent_c[index] = state['lstm_c']
        if greedy:
          actions = logits.argmax(dim=-1)
        else:
          noise = torch.as_tensor(_match_gumbel_noise(
              seed, match_ids, match_steps[match_ids], per_env,
              logits.shape[-1]), device=device)
          actions = (logits.float() + noise).argmax(dim=-1)
      match_steps[match_ids] += 1
      active_rows = active.view(len(match_ids), per_env).cpu().numpy()
      match_actions = actions.view(len(match_ids), per_env).cpu().numpy()
      match_logits = logits.view(len(match_ids), per_env, -1).float().cpu()
      for position, match in enumerate(match_ids):
        if len(taken[match]) >= quota:
          continue
        rows = active_rows[position]
        current_actions[match] += np.bincount(
            match_actions[position][rows], minlength=len(ACTION_NAMES))
        current_logits[match].append(match_logits[position][rows])
      vecenv.send(actions.cpu().numpy())
  finally:
    policy.train(was_training)
  rows = [row for match in range(matches) for row in taken[match][:limit]]
  action_counts = torch.as_tensor(sum(
      (counts for match in range(matches)
       for counts in episode_actions[match][:limit]),
      np.zeros(len(ACTION_NAMES), dtype=np.int64)))
  active_logits_rows = [
      step for match in range(matches)
      for episode in episode_logits[match][:limit] for step in episode]
  decisions = max(1, int(action_counts.sum()))
  return _promotion_metrics(rows, action_counts, decisions,
                            active_logits_rows, recurrent_horizon, greedy,
                            aborted)


class GraphedActor:
  """forward_eval plus action sampling, replayed as one CUDA graph.

  The async collector always acts for the same number of rows, and one
  eager forward is ~35 small kernel launches for very little arithmetic.
  The graph reads the policy's parameters in place, so weight updates (the
  optimizer, or copy_ into the acting copy) need no recapture.  Outputs are
  overwritten by the next call and must be consumed before it.

  Sampling uses no RNG inside the graph: uniform noise is drawn eagerly and
  passed in, and the action is argmax(logits + Gumbel(noise)), which is an
  exact sample from softmax(logits).  A graph that consumes the default
  generator conflicts with the training graph that also does.
  """

  def __init__(self, policy, rows, observation_size, device, stream):
    hidden = policy.hidden_size
    self.policy = policy
    self.rows = rows
    self.observations = torch.zeros(rows, observation_size, device=device)
    self.hidden = torch.zeros(rows, hidden, device=device)
    self.cell = torch.zeros(rows, hidden, device=device)
    self.done = torch.zeros(rows, device=device)
    self.uniform = torch.full((rows, policy.actor.out_features), 0.5,
                              device=device)
    # Graphs cannot be captured on the default stream.
    stream = stream or torch.cuda.Stream(device=device)
    torch.cuda.current_stream().synchronize()
    with torch.cuda.stream(stream):
      for _ in range(3):
        self._act()
    stream.synchronize()
    self.graph = torch.cuda.CUDAGraph()
    # thread_local: the GPU filler thread keeps replaying its own graph.
    with torch.cuda.graph(self.graph, stream=stream,
                          capture_error_mode='thread_local'):
      self.outputs = self._act()

  def _act(self):
    state = {'lstm_h': self.hidden, 'lstm_c': self.cell, 'done': self.done}
    with torch.no_grad():
      logits, value = self.policy.forward_eval(self.observations, state)
      log_probabilities = torch.log_softmax(logits.float(), dim=-1)
      gumbel = -torch.log(-torch.log(self.uniform.clamp(1e-20, 1 - 1e-7)))
      action = (log_probabilities + gumbel).argmax(dim=-1)
      logprob = log_probabilities.gather(1, action[:, None]).squeeze(1)
    return action, logprob, value, state['lstm_h'], state['lstm_c']

  def __call__(self, observations, hidden, cell, done):
    self.observations.copy_(observations)
    self.hidden.copy_(hidden)
    self.cell.copy_(cell)
    self.done.copy_(done)
    self.uniform.uniform_()
    self.graph.replay()
    return self.outputs


class FootballPuffeRL(pufferl.PuffeRL):
  """PPO restricted to the agent rows the curriculum actually controls."""

  # Loss statistics accumulated on the GPU inside the minibatch loop and read
  # back once, when the epoch is logged.
  LOSS_NAMES = ('policy_loss', 'value_loss', 'entropy', 'gradient_norm',
                'old_approx_kl', 'approx_kl', 'clipfrac', 'importance',
                'minibatch_transitions')
  # Recurrent memory runs across rollout segments (see
  # _init_async_collection); off, it restarts at every segment.
  carry_memory = False

  def __init__(self, config, vecenv, policy, logger=None,
               log_interval_seconds=0.25, overlap_collection=False,
               graph_actor=False, graph_update=False, carry_memory=False):
    super().__init__(config, vecenv, policy, logger=logger)
    self.carry_memory = bool(carry_memory)
    if self.carry_memory and not getattr(vecenv, 'is_async', False):
      raise ValueError('carried memory needs --async-collection; PuffeRL\'s '
                       'own rollout restarts memory at every segment')
    # PufferLib's epoch-based scheduler is unused: train() sets the rate from
    # training progress (see cosine_learning_rate).
    self.async_collection = bool(getattr(vecenv, 'is_async', False))
    self.overlap_collection = bool(
        overlap_collection and self.async_collection and
        torch.device(config['device']).type == 'cuda')
    self.graph_actor = bool(
        graph_actor and self.async_collection and
        torch.device(config['device']).type == 'cuda')
    self.graphed_actor = None
    self._epoch_values = None
    # A captured update replays Adam as recorded, so Adam keeps its step
    # count on the GPU (capturable) and the learning rate lives in a GPU
    # tensor that set_learning_rate fills in place.
    self.graph_update = bool(
        graph_update and torch.device(config['device']).type == 'cuda')
    self._update_graph = None
    self._update_pool = None
    self._update_capacity = None
    self.update_captures = 0
    if self.graph_update:
      self.optimizer = torch.optim.Adam(
          self.uncompiled_policy.parameters(),
          lr=torch.tensor(float(config['learning_rate']),
                          device=config['device']),
          betas=(config['adam_beta1'], config['adam_beta2']),
          eps=config['adam_eps'], capturable=True)
    if self.async_collection:
      self._init_async_collection()
    self.optimizer_steps = 0
    self.active_steps = torch.zeros((), dtype=torch.long,
                                    device=config['device'])
    self.last_active_steps = 0
    self.last_active_log_time = time.time()
    self.promotion_metrics = {}
    # Diagnostics are only computed on epochs that are logged, so this also
    # sets how often the trainer pays for them.
    self.log_interval_seconds = float(log_interval_seconds)

  def _init_async_collection(self):
    """Buffers for collecting segments from envs that run at their own pace.

    PuffeRL's collector steps every env exactly `horizon` times per epoch in
    a fixed order, so one env in an engine reset holds up the rest.  Here
    each match fills its own `horizon`-step segment (22 agent rows) as fast
    as it runs, into blocks of a buffer with room for several segments per
    match.  An epoch trains on exactly one segment per match's worth of
    completed blocks; segments in progress, and any completed beyond that,
    carry over.  By default recurrent memory starts from zero at every
    segment and every episode end, exactly as in PuffeRL's rollout.  With
    carry_memory it clears only at episode ends: each segment continues its
    match's memory, and the memory it started from is kept per buffer row
    (segment_h, segment_c) for training.  Either way BPTT replays what the
    policy saw.

    With overlap on, collection continues while the update runs on the GPU:
    a copy of the policy acts on its own CUDA stream and is synced after
    every update, so an action is at most one update stale.  Its log-prob is
    stored with the transition, so PPO's importance ratio stays exact.
    """
    config = self.config
    device = config['device']
    self.agents_per_env = self.vecenv.driver_env.num_agents
    self.num_matches = self.total_agents // self.agents_per_env
    self.num_blocks = self.segments // self.agents_per_env
    self.blocks_per_epoch = self.num_matches
    needed = (3 if self.overlap_collection else 2) * self.num_matches
    if self.num_blocks < needed:
      raise ValueError('async collection needs {} segment blocks, the buffer '
                       'has {}'.format(needed, self.num_blocks))
    self.free_blocks = list(range(self.num_blocks))
    self.complete_blocks = []
    self.training_blocks = []
    self.match_block = np.full(self.num_matches, -1, dtype=np.int64)
    self.match_step = np.zeros(self.num_matches, dtype=np.int64)
    hidden = self.uncompiled_policy.hidden_size
    self.agent_h = torch.zeros(self.total_agents, hidden, device=device)
    self.agent_c = torch.zeros(self.total_agents, hidden, device=device)
    if self.carry_memory:
      # Per buffer row: the actor's memory when that row's segment began.
      self.segment_h = torch.zeros(self.segments, hidden, device=device)
      self.segment_c = torch.zeros(self.segments, hidden, device=device)
    self.row_training = torch.zeros(self.segments, dtype=torch.bool,
                                    device=device)
    self.agent_offsets = np.arange(self.agents_per_env)
    self.discarded_steps = 0
    self.dropped_segments = 0
    self.acting_policy = self.uncompiled_policy
    self.acting_stream = None
    if self.overlap_collection:
      import copy
      self.acting_policy = copy.deepcopy(self.uncompiled_policy)
      for parameter in self.acting_policy.parameters():
        parameter.requires_grad_(False)
      if torch.device(device).type == 'cuda':
        _, highest = torch.cuda.Stream.priority_range()
        self.acting_stream = torch.cuda.Stream(device=device,
                                               priority=highest)

  def _sync_acting_policy(self):
    """Copy the trained weights (and normalizer statistics) to the actor."""
    if self.acting_policy is self.uncompiled_policy:
      return
    main = torch.cuda.current_stream() if self.acting_stream else None
    if main is not None:
      # Nothing the actor has queued may read weights mid-copy.
      main.wait_stream(self.acting_stream)
    with torch.no_grad():
      for target, source in zip(self.acting_policy.state_dict().values(),
                                self.uncompiled_policy.state_dict().values()):
        target.copy_(source)
    if main is not None:
      self.acting_stream.wait_stream(main)

  def _workers_ready(self):
    """True if recv() would return a batch without waiting."""
    vecenv = self.vecenv
    if len(vecenv.ready_workers) >= vecenv.workers_per_batch:
      return True
    waiting = vecenv.waiting_workers
    if not waiting:
      return False
    ready = np.count_nonzero(
        vecenv.buf['semaphores'][waiting] >= pufferlib.vector.MAIN)
    return len(vecenv.ready_workers) + ready >= vecenv.workers_per_batch

  def _collect_batch(self):
    """Receive one batch of matches, act for them and store the step."""
    profile = self.profile
    epoch = self.epoch
    config = self.config
    device = config['device']
    horizon = config['bptt_horizon']
    per_env = self.agents_per_env

    profile('env', epoch)
    o, r, d, t, info, agent_ids, mask = self.vecenv.recv()

    profile('eval_misc', epoch)
    agent_ids = np.asarray(agent_ids)
    matches = agent_ids[::per_env] // per_env
    starting = matches[self.match_block[matches] < 0]
    fresh_matches = []
    for match in starting:
      if self.free_blocks:
        self.match_block[match] = self.free_blocks.pop()
        self.match_step[match] = 0
        fresh_matches.append(match)
    fresh_matches = np.asarray(fresh_matches, dtype=np.int64)
    recording = self.match_block[matches] >= 0
    self.discarded_steps += int((~recording).sum())
    self.global_step += int(mask.sum())

    stream = (torch.cuda.stream(self.acting_stream)
              if self.acting_stream is not None else contextlib.nullcontext())
    with stream:
      agent_index = torch.as_tensor(agent_ids, device=device)
      if len(fresh_matches):
        fresh = torch.as_tensor(
            (fresh_matches[:, None] * per_env +
             self.agent_offsets).ravel(), device=device)
        if self.carry_memory:
          # The segment continues its match's memory; keep where it started
          # so training can replay the segment from the same state.
          block_rows = torch.as_tensor(
              (self.match_block[fresh_matches][:, None] * per_env +
               self.agent_offsets).ravel(), device=device)
          self.segment_h[block_rows] = self.agent_h[fresh]
          self.segment_c[block_rows] = self.agent_c[fresh]
        else:
          # Every segment starts from zero memory, as PuffeRL's windows do.
          self.agent_h[fresh] = 0
          self.agent_c[fresh] = 0

      profile('eval_copy', epoch)
      o_device = torch.as_tensor(o).to(device)
      r = torch.as_tensor(r).to(device)
      d = torch.as_tensor(d).to(device)

      profile('eval_forward', epoch)
      with torch.no_grad(), self.amp_context:
        hidden = self.agent_h[agent_index]
        cell = self.agent_c[agent_index]
        if self.graph_actor and self.graphed_actor is None:
          self.graphed_actor = GraphedActor(
              self.acting_policy, len(agent_ids), o_device.shape[-1], device,
              self.acting_stream)
        if (self.graphed_actor is not None and
            self.graphed_actor.rows == len(agent_ids)):
          action, logprob, value, new_hidden, new_cell = self.graphed_actor(
              o_device, hidden, cell, d)
        else:
          state = {'lstm_h': hidden, 'lstm_c': cell, 'done': d}
          logits, value = self.acting_policy.forward_eval(o_device, state)
          action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
          new_hidden, new_cell = state['lstm_h'], state['lstm_c']
        # Rewards are stored unclipped.  Clipping to [-1, 1] cut the terminal
        # refund of potential-based shaping (up to 2 * ball scale), so the
        # shaping stopped telescoping and changed what is optimal.

      profile('eval_copy', epoch)
      with torch.no_grad():
        self.agent_h[agent_index] = new_hidden
        self.agent_c[agent_index] = new_cell
        if recording.any():
          # Only matches holding a block are stored; the rest (no free block
          # while an update runs) act, and start a fresh segment later.
          keep = np.repeat(recording, per_env)
          keep_rows = torch.as_tensor(np.flatnonzero(keep), device=device)
          rows = torch.as_tensor(
              (self.match_block[matches[recording]][:, None] * per_env +
               self.agent_offsets).ravel(), device=device)
          steps = torch.as_tensor(np.repeat(
              self.match_step[matches[recording]], per_env), device=device)
          # Indexed writes, unlike PuffeRL's slice writes, do not cast.
          self.observations[rows, steps] = o_device[keep_rows].to(
              self.observations.dtype)
          self.actions[rows, steps] = action[keep_rows].to(self.actions.dtype)
          self.logprobs[rows, steps] = logprob[keep_rows].to(
              self.logprobs.dtype)
          self.rewards[rows, steps] = r[keep_rows].to(self.rewards.dtype)
          self.terminals[rows, steps] = d[keep_rows].to(self.terminals.dtype)
          self.values[rows, steps] = value.flatten()[keep_rows].to(
              self.values.dtype)
        action = action.cpu().numpy()

    recorded = matches[recording]
    self.match_step[recorded] += 1
    for match in recorded[self.match_step[recorded] >= horizon]:
      self.complete_blocks.append(int(self.match_block[match]))
      self.match_block[match] = -1

    profile('eval_misc', epoch)
    for i in info:
      for k, v in pufferlib.unroll_nested_dict(i):
        if isinstance(v, np.ndarray):
          v = v.tolist()
        elif isinstance(v, (list, tuple)):
          self.stats[k].extend(v)
        else:
          self.stats[k].append(v)

    profile('env', epoch)
    self.vecenv.send(action)

  def _evaluate_async(self):
    profile = self.profile
    epoch = self.epoch
    profile('eval', epoch)
    profile('eval_misc', epoch, nest=True)
    # Last epoch's training blocks were trained on; recycle their rows.
    self.free_blocks.extend(self.training_blocks)
    self.training_blocks = []
    while len(self.complete_blocks) < self.blocks_per_epoch:
      self._collect_batch()
    profile.end()
    return self.stats

  def _claim_training_blocks(self):
    """Hand exactly one epoch of the newest completed segments to PPO.

    With overlap the environments can outrun the learner.  Older surplus
    segments are dropped rather than queued, so trained data is never more
    than about one update stale.
    """
    per_env = self.agents_per_env
    self.training_blocks = self.complete_blocks[-self.blocks_per_epoch:]
    surplus = self.complete_blocks[:-self.blocks_per_epoch]
    self.dropped_segments += len(surplus)
    self.free_blocks.extend(surplus)
    self.complete_blocks = []
    rows = (np.asarray(self.training_blocks)[:, None] * per_env +
            self.agent_offsets).ravel()
    self.training_rows = torch.as_tensor(rows, device=self.row_training.device)
    self.row_training.zero_()
    self.row_training[self.training_rows] = True
    if self.acting_stream is not None:
      # Everything the actor wrote must land before PPO reads it.
      torch.cuda.current_stream().wait_stream(self.acting_stream)

  def _collect_while(self, done):
    """Serve ready workers until `done()`; the update runs on the GPU."""
    while not done():
      if self._workers_ready():
        self._collect_batch()

  def _stage_epoch(self, rollout_values, advantages, returns, trainable):
    if getattr(self, '_epoch_values', None) is None:
      self._epoch_values = torch.zeros_like(rollout_values)
      self._epoch_advantages = torch.zeros_like(rollout_values)
      self._epoch_returns = torch.zeros_like(rollout_values)
      self._epoch_weight = torch.zeros_like(rollout_values)
      self._totals = torch.zeros(len(self.LOSS_NAMES),
                                 device=rollout_values.device)
    self._epoch_values.copy_(rollout_values)
    self._epoch_advantages.copy_(advantages)
    self._epoch_returns.copy_(returns)
    self._epoch_weight.copy_(trainable)
    self._totals.zero_()

  def _starting_memory(self, index):
    """The window-forward state that starts buffer rows `index` as collected.

    Empty when memory restarts at every segment (the window forward starts
    from zero by itself).  With carry_memory, the memory each row's segment
    started from plus the done flags that clear it where the segment opens
    on a new episode.  Shapes are fixed by `index`, so a captured update can
    read it.
    """
    if not self.carry_memory:
      return {}
    return {'done': self.terminals[index], 'lstm_h': self.segment_h[index],
            'lstm_c': self.segment_c[index]}

  def _minibatch_step(self, policy, index, pieces):
    """One PPO gradient step on the segments in `index`.

    Forward (BPTT over whole segments, using the precomputed episode_pieces
    layout of those segments), masked PPO losses, backward, gradient clip,
    Adam.  _update calls it once per update epoch with the whole rollout.
    """
    config = self.config
    clip = config['clip_coef']
    mask = self._epoch_weight[index]
    count = mask.sum()
    denominator = count.clamp_min(1)

    def masked_mean(values):
      return (values * mask).sum() / denominator

    logits, new_values = policy(
        self.observations[index],
        {'episode_pieces': pieces, **self._starting_memory(index)})
    _, new_logprobs, entropy = pufferlib.pytorch.sample_logits(
        logits, action=self.actions[index])
    new_logprobs = new_logprobs.view(mask.shape)
    entropy = entropy.view(mask.shape)
    old_values = self._epoch_values[index]
    mb_returns = self._epoch_returns[index]
    log_ratio = new_logprobs - self.logprobs[index]
    ratio = log_ratio.exp()

    mb_advantages = self._epoch_advantages[index].float()
    advantage_mean = masked_mean(mb_advantages)
    advantage_std = masked_mean(
        (mb_advantages - advantage_mean).square()).sqrt().clamp_min(1e-8)
    mb_advantages = (mb_advantages - advantage_mean) / advantage_std
    policy_loss = masked_mean(torch.max(
        -mb_advantages * ratio,
        -mb_advantages * ratio.clamp(1 - clip, 1 + clip)))
    clipped_values = old_values + (new_values - old_values).clamp(
        -config['vf_clip_coef'], config['vf_clip_coef'])
    value_loss = 0.5 * masked_mean(torch.max(
        (new_values - mb_returns) ** 2,
        (clipped_values - mb_returns) ** 2))
    entropy_loss = masked_mean(entropy)
    loss = (policy_loss + config['vf_coef'] * value_loss -
            config['ent_coef'] * entropy_loss)

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config['max_grad_norm'])
    self.optimizer.step()
    self.optimizer_steps += 1

    with torch.no_grad():
      self._totals += torch.stack((
          policy_loss, value_loss, entropy_loss, gradient_norm,
          masked_mean(-log_ratio),
          masked_mean((ratio - 1) - log_ratio),
          masked_mean(((ratio - 1).abs() > clip).float()),
          masked_mean(ratio),
          count)).detach()

  def _update(self, rows):
    """Train on this rollout: `update_epochs` steps over all its segments.

    `rows` are the rollout's buffer rows (a fixed number per epoch); rows
    and steps that are not trainable are masked out of every loss.  The
    minibatch is the whole rollout (~1000 segments at 46 matches, ~0.6 GB
    of activations), so update_epochs is exactly the number of optimizer
    steps per rollout.

    With graph_update, the step is a captured CUDA graph replayed
    update_epochs times.  The main thread also serves the environments while
    the update runs; launching every kernel from Python kept it busy ~40 ms
    per epoch (collection fell from ~280k to ~140k steps/s), and capturing
    a new graph each rollout still cost ~30 ms.  So the graph reads its rows
    and episode layout from fixed buffers, and the layout has a fixed
    capacity per step (episode_pieces pads it exactly): each rollout copies
    its layout in and replays.  The graph is captured again only when a
    rollout needs more slots than the capacity, or leaves over half of it
    unused.  Capturing does not execute, so the very first step runs
    eagerly to create Adam's state first.
    """
    done = self.terminals[rows]
    epochs = self.config['update_epochs']
    if not self.graph_update:
      pieces = episode_pieces(done)
      for _ in range(epochs):
        self._minibatch_step(self.policy, rows, pieces)
      return
    needed = episode_batch_sizes(done)
    capacity = self._update_capacity
    if (self._update_graph is None or
        rows.numel() != self._update_rows.numel() or
        bool((needed > capacity).any()) or
        2 * int(needed.sum()) < int(capacity.sum())):
      capacity = (torch.ceil(needed * 1.25 / 64) * 64).long().clamp_min(64)
      self._capture_update(rows, episode_pieces(done, capacity))
      first = self._update_warmup_steps
    else:
      order, inverse, _ = episode_pieces(done, capacity)
      self._update_rows.copy_(rows)
      self._update_order.copy_(order)
      self._update_inverse.copy_(inverse)
      first = 0
    for _ in range(first, epochs):
      self._update_graph.replay()
      self.optimizer_steps += 1

  def _capture_update(self, rows, pieces):
    """Capture one PPO step reading the fixed update buffers.

    The very first capture is preceded by one real, eager step (on a side
    stream) so Adam's state and the cuBLAS/cuDNN workspaces exist; that
    step counts as the rollout's first (_update_warmup_steps).  Later
    captures reuse the same memory pool.
    """
    policy = self.uncompiled_policy
    order, inverse, capacity = pieces
    self._update_rows = rows.clone()
    self._update_order = order.clone()
    self._update_inverse = inverse.clone()
    self._update_capacity = capacity
    static = (self._update_order, self._update_inverse, capacity)
    self._update_warmup_steps = 0
    if self._update_pool is None:
      side = torch.cuda.Stream()
      side.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(side):
        self._minibatch_step(policy, self._update_rows, static)
      torch.cuda.current_stream().wait_stream(side)
      self._update_pool = torch.cuda.graph_pool_handle()
      self._update_warmup_steps = 1
    # Gradients are allocated inside the graph's pool during capture.
    self.optimizer.zero_grad(set_to_none=True)
    self._update_graph = None
    graph = torch.cuda.CUDAGraph()
    steps = self.optimizer_steps
    # thread_local: the GPU filler thread keeps launching its own kernels.
    with torch.cuda.graph(graph, pool=self._update_pool,
                          capture_error_mode='thread_local'):
      self._minibatch_step(policy, self._update_rows, static)
    self.optimizer_steps = steps
    self._update_graph = graph
    self.update_captures += 1

  def evaluate(self):
    if self.async_collection:
      return self._evaluate_async()
    return super().evaluate()

  def record_promotion(self, level, metrics, advanced):
    """Keep the latest gate result, also keyed by its level, for the next log."""
    self.promotion_metrics = {
        'promotion_level': float(level),
        'promotion_advanced': float(advanced),
        **metrics,
        **level_entry_metrics(level, metrics),
    }

  def train(self):
    profile = self.profile
    epoch = self.epoch
    profile('train', epoch)
    profile('train_misc', epoch, nest=True)
    config = self.config
    device = config['device']
    done_training = (
        self.global_step >= config['total_timesteps'])
    log_epoch = (done_training or self.global_step == 0 or
                 time.time() > self.last_log_time + self.log_interval_seconds)
    losses = {}
    if self.async_collection:
      # Before any buffer read: waits for the actor's stream.
      self._claim_training_blocks()

    rollout_values = self.values.clone()
    advantages, returns, valid = generalized_advantages(
        rollout_values, self.rewards, self.terminals,
        config['gamma'], config['gae_lambda'])
    active = self.observations.flatten(2).abs().sum(dim=-1) > 0
    trainable = active & valid
    if self.async_collection:
      # Only this epoch's completed segments; the rest carry over.
      trainable &= self.row_training[:, None]
    segment_index = trainable.any(dim=1).nonzero().flatten()
    num_segments = int(segment_index.numel())
    if num_segments == 0:
      raise RuntimeError('rollout contains no controlled agents')
    num_minibatches = config['update_epochs']

    # Statistics come only from training rollouts, never from the held-out
    # promotion evaluation, so evaluation stays a clean measurement.
    self.uncompiled_policy.normalizer.update(self.observations[trainable])
    if log_epoch:
      losses['observation_scale_mean'] = float(
          self.uncompiled_policy.normalizer.var.sqrt().mean().item())
      losses['pre_update_explained_variance'] = explained_variance(
          rollout_values[trainable], returns[trainable])

    # Every loss is a masked mean over the (segment, step) grid, which is the
    # same number as indexing out the trainable rows first, without the
    # data-dependent shapes that force a GPU sync on every step.
    self._stage_epoch(rollout_values, advantages, returns, trainable)
    if config['anneal_lr']:
      set_learning_rate(self.optimizer, cosine_learning_rate(
          config['learning_rate'], self.global_step,
          config['total_timesteps']))
    # Always the epoch's whole set of rows (a fixed count, so a captured
    # update keeps its shapes); untrainable steps are masked in the loss.
    self._update(self.training_rows if self.async_collection else
                 torch.arange(self.segments, device=device))
    totals = self._totals

    if self.overlap_collection:
      # The update is queued on the GPU; keep the envs busy until it is done.
      finished = torch.cuda.Event()
      finished.record()
      self._collect_while(finished.query)
      self._sync_acting_policy()

    profile('train_misc', epoch)
    self.active_steps += trainable.sum()

    if log_epoch:
      for name, value in zip(self.LOSS_NAMES,
                             (totals / num_minibatches).tolist()):
        losses[name] = value
      active_steps = int(self.active_steps.item())
      now = time.time()
      reward_signs = torch.stack((
          (self.rewards[active] > 0).float().mean(),
          (self.rewards[active] < 0).float().mean(),
          advantages[trainable].mean(),
          advantages[trainable].std(unbiased=False),
          active.float().mean())).tolist()
      losses.update({
          'optimizer_steps': float(self.optimizer_steps),
          'update_captures': float(self.update_captures),
          'ppo_minibatches': float(num_minibatches),
          'active_agent_fraction': reward_signs[4],
          'active_transitions': float(trainable.sum().item()),
          'active_segments': float(num_segments),
          'advantage_mean': reward_signs[2],
          'advantage_std': reward_signs[3],
          'positive_reward_fraction': reward_signs[0],
          'negative_reward_fraction': reward_signs[1],
          'learning_rate': float(self.optimizer.param_groups[0]['lr']),
          'active_SPS': (
              (active_steps - self.last_active_steps) /
              max(1e-6, now - self.last_active_log_time)),
      })
      self.last_active_steps = active_steps
      self.last_active_log_time = now

      # Post-update health has to replay whole segments: a recurrent critic
      # scored from a zeroed hidden state is not the critic that acts.
      with torch.no_grad():
        index = segment_index[:min(64, num_segments)]
        sample_mask = trainable[index]
        post_logits, post_values = self.policy(
            self.observations[index],
            {'done': self.terminals[index], **self._starting_memory(index)})
        post_logits = post_logits.view(*sample_mask.shape, -1)[sample_mask]
        for name, value in policy_diagnostics(post_logits).items():
          losses[name] = value.item()
        losses['post_update_explained_variance'] = explained_variance(
            post_values[sample_mask], returns[index][sample_mask])
      losses['explained_variance'] = losses['pre_update_explained_variance']

      action_counts = torch.bincount(
          self.actions[active].long().flatten(),
          minlength=len(ACTION_NAMES)).float()
      action_fractions = (
          action_counts / action_counts.sum().clamp_min(1)).tolist()
      for index, name in enumerate(ACTION_NAMES):
        losses['action_{}_fraction'.format(name)] = action_fractions[index]
      losses['max_action_fraction'] = max(action_fractions)
      losses['shot_fraction'] = action_fractions[SHOT_ACTION]
      losses.update(self.promotion_metrics)

    profile.end()
    logs = None
    self.epoch += 1
    if log_epoch:
      self.losses = losses
      logs = self.mean_and_log()
      self.print_dashboard()
      self.stats = defaultdict(list)
      self.last_log_time = time.time()
      self.last_log_step = self.global_step
      profile.clear()
    if self.epoch % config['checkpoint_interval'] == 0 or done_training:
      self.save_checkpoint()
    return logs


def _base_config():
  import sys
  original_argv = sys.argv
  sys.argv = [original_argv[0]]
  try:
    return dict(pufferl.load_config('default')['train'])
  finally:
    sys.argv = original_argv


def _async_batch_workers(args, num_workers, promotion=False):
  flag = 'async_promotion' if promotion else 'async_collection'
  if not getattr(args, flag, False):
    return None
  if args.async_batch_workers is not None:
    return args.async_batch_workers
  return max(1, num_workers // 3)


def memory_window(args):
  """Steps the training rollout keeps recurrent memory before restarting it.

  The bptt horizon, since every rollout segment starts from zero memory, or
  None with --carry-lstm-memory, where memory clears only at episode ends.
  Evaluation replays a policy with the memory it was trained with.
  """
  return None if args.carry_lstm_memory else args.bptt_horizon


def _make_promotion_env(args, curriculum_level_value, frozen_defence_path=None):
  """Worker pool for gate evaluations at the given curriculum level.

  The frozen opponent keeps its memory the way this run trains unless
  --frozen-defence-horizon says otherwise (an opponent file trained by
  another run, where 0 means whole episodes).
  """
  frozen_defence_horizon = args.frozen_defence_horizon
  if frozen_defence_horizon is None:
    frozen_defence_horizon = memory_window(args)
  elif frozen_defence_horizon == 0:
    frozen_defence_horizon = None
  return make_vector_env(
      num_envs=args.promotion_workers, num_workers=args.promotion_workers,
      batch_size=args.promotion_workers, reserved_cpus=0,
      async_batch_workers=_async_batch_workers(
          args, args.promotion_workers, promotion=True),
      seed=args.seed + 1000000, env_name=args.env_name,
      frame_stack=args.frame_stack,
      curriculum_levels=args.curriculum_levels,
      curriculum_window=args.promotion_episodes + 1,  # never auto-advances
      curriculum_success_threshold=args.curriculum_success_threshold,
      attacker_only_levels=args.attacker_only_levels,
      curriculum_level_value=curriculum_level_value,
      sort_players=args.sort_players,
      curriculum_evaluation=True,
      observation=args.observation,
      frozen_defence_path=frozen_defence_path,
      frozen_defence_horizon=frozen_defence_horizon,
      # The gate scores on the same spawn distribution the policy trains on.
      spawn=args.spawn)


def rearm_for_reset(vecenv):
  """Let a synchronous Multiprocessing vecenv be reset() again.

  After a completed step() every worker is idle and its infos have already
  been read, but a worker that reported an episode end still has its
  semaphore at INFO.  PufferLib's async_reset flushes waiting workers by
  reading the pipe of every INFO worker a second time, which blocks forever;
  that was the promotion deadlock.  Nothing is in flight, so there is
  nothing to flush.
  """
  # An asynchronous pool has workers in flight; PufferLib's own flush waits
  # for them and reads each pending INFO exactly once.
  if hasattr(vecenv, 'waiting_workers') and not getattr(
      vecenv, 'is_async', False):
    vecenv.waiting_workers = []
    vecenv.ready_workers = []


_PROMOTION_POOLS = {}


def _promotion_pool(args, level_value, frozen_defence_path):
  """One persistent worker pool per opponent type (frozen or self-play).

  Building a pool starts 30 processes and a game engine in each, 10-20 s
  apiece, and the gate used to build three per check.
  """
  key = frozen_defence_path
  if key not in _PROMOTION_POOLS:
    _PROMOTION_POOLS[key] = _make_promotion_env(
        args, level_value, frozen_defence_path)
  return _PROMOTION_POOLS[key]


def close_promotion_pools():
  while _PROMOTION_POOLS:
    _, pool = _PROMOTION_POOLS.popitem()
    pool.close()


def _run_promotion(args, policy, level_value, level, device, episodes,
                   frozen_defence_path=None, greedy=False, early_abort=None):
  """One held-out evaluation.

  With --reuse-promotion-envs the worker pool persists across evaluations and
  each reset(seed) reseeds the engines in place, which replays exactly the
  episodes a freshly built pool would.  Otherwise a fresh pool is built and
  torn down every time.
  """
  reuse = getattr(args, 'reuse_promotion_envs', False)
  if reuse:
    promotion_env = _promotion_pool(args, level_value, frozen_defence_path)
    rearm_for_reset(promotion_env)
  else:
    promotion_env = _make_promotion_env(args, level_value, frozen_defence_path)
  evaluate = (evaluate_promotion_async
              if getattr(promotion_env, 'is_async', False)
              else evaluate_promotion)
  try:
    return evaluate(
        policy, promotion_env, episodes,
        args.seed + 1000000 + 10000 * level, device,
        recurrent_horizon=memory_window(args), greedy=greedy,
        early_abort=early_abort)
  finally:
    if not reuse:
      promotion_env.close()


def _prefixed(prefix, metrics):
  return {prefix + name: value for name, value in metrics.items()}


def level_entry_metrics(level, metrics):
  """Name a gate result by its level so each level gets its own progress line.

  The gate opponent is the policy as it was on entering the level (the one
  that cleared the level before; for level 0 the untrained policy, or the
  --frozen-defence-init file), fixed for as long as the level trains.  The
  score rate against it should therefore rise while a level is learning and
  go flat when it stalls; keying by level keeps each level's line separate
  in the logger.  The un-keyed rate is one continuous series across levels.
  """
  prefix = 'level_entry/level_{}/'.format(int(level))
  return {
      'level_entry/level': float(level),
      'level_entry/success_rate': metrics['promotion_success_rate'],
      prefix + 'success_rate': metrics['promotion_success_rate'],
      prefix + 'worst_two_template_success_rate':
          metrics['promotion_worst_two_template_success_rate'],
      prefix + 'episodes': metrics['promotion_episodes'],
  }


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-workers', type=int, default=30)
  parser.add_argument('--torch-threads', type=int, default=0,
                      help='threads for the update and the evaluation '
                           'forward; 0 leaves PyTorch alone. The launcher '
                           'pins OMP_NUM_THREADS=1 for the env workers, '
                           'which otherwise pins this process too')
  parser.add_argument('--total-timesteps', type=int, default=1_000_000_000)
  parser.add_argument('--env-name', default='11_vs_11_curriculum',
                      choices=('11_vs_11_curriculum', ADVANTAGE_ENV_NAME))
  parser.add_argument('--spawn', default='curriculum',
                      choices=('curriculum', 'uniform'),
                      help='curriculum: each level spawns its own scene; '
                           'uniform: ignore levels and place the ball and '
                           'players uniformly over the pitch, all onside '
                           '(advantage env only)')
  parser.add_argument('--curriculum-levels', type=int, default=None)
  parser.add_argument('--start-level', type=int, default=0,
                      help='begin at this curriculum level instead of 0')
  parser.add_argument('--curriculum-window', type=int, default=20)
  parser.add_argument('--curriculum-success-threshold', type=float, default=0.6)
  parser.add_argument('--attacker-only-levels', type=int, default=None)
  # Evaluation budget.  256 episodes every 25 epochs was ~40% of a ten-hour
  # job; 512 + 128 + 128 every 100 epochs against a frozen defence (slower per
  # episode, the defence is a network in every worker) came to ~50%.  This
  # schedule is 256 + 64 + 64 every 100 epochs, and a hopeless gate stops at
  # 64 episodes.
  parser.add_argument('--promotion-interval', type=int, default=100)
  parser.add_argument('--promotion-episodes', type=int, default=256)
  parser.add_argument('--promotion-min-episodes', type=int, default=0,
                      help='episodes before --promotion-early-abort-margin '
                           'may end an evaluation; 0 derives it from '
                           '--promotion-episodes and the template count')
  parser.add_argument('--promotion-workers', type=int, default=30)
  parser.add_argument('--promotion-worst-template-threshold', type=float,
                      default=0.4)
  parser.add_argument('--promotion-early-abort-margin', type=float,
                      default=0.2,
                      help='stop the gate evaluation after a quarter of its '
                           'episodes when success is this far below the '
                           'threshold; 0 disables')
  parser.add_argument('--frozen-defence-gate',
                      action=argparse.BooleanOptionalAction, default=True,
                      help='gate promotion against a frozen snapshot of the '
                           'policy taken on entering the level, instead of '
                           'the live self-play opponent')
  parser.add_argument('--frozen-defence-init', default=None,
                      help='start the gate opponent from this policy file '
                           'instead of the untrained policy, so every gate '
                           'evaluation on the starting level is scored '
                           'against the same fixed defence')
  parser.add_argument('--greedy-promotion-episodes', type=int, default=64,
                      help='extra argmax-action evaluation per promotion '
                           'check (diagnostic only); 0 disables')
  parser.add_argument('--selfplay-promotion-episodes', type=int, default=64,
                      help='extra live self-play evaluation per promotion '
                           'check when the gate is frozen-defence '
                           '(diagnostic only, comparable to older runs); '
                           '0 disables')
  # Timed promotion was added when no run could clear level 0, so that the
  # later levels were at least visited.  Once level 3 fell, two seeds rode the
  # 200-epoch timer from level 4 to level 20 with success at zero.  Mastery
  # is the goal, so every level is score-gated unless a job says otherwise.
  parser.add_argument('--scored-promotion-levels', type=int, default=None,
                      help='levels below this advance only on the score '
                           'gate; default: every level')
  parser.add_argument('--timed-promotion-epochs', type=int, default=200,
                      help='at or above --scored-promotion-levels, also '
                           'advance after this many epochs on a level')
  parser.add_argument('--anneal-lr', action=argparse.BooleanOptionalAction,
                      default=False)
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  # At 0.01 the entropy bonus matched the policy-gradient term in size and
  # every run stayed within 10% of a uniform policy after 3000+ epochs.
  parser.add_argument('--ent-coef', type=float, default=0.001)
  parser.add_argument('--vf-coef', type=float, default=0.5)
  parser.add_argument('--clip-coef', type=float, default=0.2)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--ball-potential', type=float, default=0.0,
                      help='potential-based shaping scale k: each side is '
                           'rewarded gamma*Phi(s\')-Phi(s) with Phi = k * '
                           '(ball x toward the goal it attacks, in [-1, 1]); '
                           'the other team gets exactly the negative, so '
                           'shaping is zero-sum and leaves the optimal '
                           'policy unchanged (Ng, Harada & Russell 1999); '
                           '0 disables')
  parser.add_argument('--gae-lambda', type=float, default=0.95)
  parser.add_argument('--player-potential', type=float, default=0.0,
                      help='potential-based shaping scale for the closest '
                           'teammate to the ball; added to '
                           'the ball potential with the same gamma and '
                           'zero terminal potential; 0 disables')
  parser.add_argument('--player-potential-fade-speed', type=float,
                      default=0.0,
                      help='fade the player potential out as the ball '
                           'speeds up, reaching 0 at this observed '
                           'ball-direction magnitude (dribble <~0.27, struck '
                           'ball ~1), so kicking costs no potential while a '
                           'settled loose ball still pulls players to it; '
                           'still strictly potential-based; 0 disables')
  parser.add_argument('--width-spawn-fraction', type=float, default=0.0,
                      help='share of training episodes (never gate '
                           'episodes) whose outfield players, except the '
                           'goal-side blockers, keep their level x but spawn '
                           'uniformly across the pitch width, so they must '
                           'go and get the ball (advantage env only)')
  parser.add_argument('--update-epochs', type=int, default=4,
                      help='optimizer steps per rollout; each step trains on '
                           'the whole rollout (the minibatch is the rollout)')
  parser.add_argument('--bptt-horizon', type=int, default=32)
  parser.add_argument('--carry-lstm-memory',
                      action=argparse.BooleanOptionalAction, default=False,
                      help='keep recurrent memory across rollout segments '
                           '(cleared only at episode ends) and train each '
                           'segment from the memory it started with, instead '
                           'of restarting memory every --bptt-horizon steps; '
                           'needs --async-collection')
  parser.add_argument('--frozen-defence-horizon', type=int, default=None,
                      help='steps the frozen gate opponent keeps memory '
                           'before restarting it (0: whole episodes); by '
                           'default the same as this run trains with')
  parser.add_argument('--hidden-size', type=int, default=256)
  parser.add_argument('--observation', default='simple115',
                      choices=('simple115', 'entities'),
                      help='simple115: egocentric simple115v2; entities: '
                           'context, ball and per-player rows with absolute '
                           'positions, ball owner, keepers, offside, '
                           'sticky buttons and steps left '
                           '(gfootball/env/entity_observation.py)')
  parser.add_argument('--network', default='mlp',
                      choices=('mlp', 'transformer'),
                      help='trunk before the LSTM: a dense MLP over the '
                           'whole row, or attention over the players and '
                           'ball (needs --observation entities)')
  parser.add_argument('--frame-stack', type=int, default=1, choices=(1, 4))
  parser.add_argument('--sort-players',
                      action=argparse.BooleanOptionalAction, default=True,
                      help='order other players by distance from the '
                           'controlled player')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
  parser.add_argument('--data-dir', default='experiments/football-ppo')
  parser.add_argument('--wandb', action=argparse.BooleanOptionalAction,
                      default=True)
  parser.add_argument('--wandb-project', default='google-football-fast-rl')
  parser.add_argument('--wandb-group', default='self-play-lstm-ppo')
  parser.add_argument('--wandb-tag', default=None)
  # Throughput.  None of these change what PPO computes.
  parser.add_argument('--envs-per-worker', type=int, default=1,
                      help='matches stepped by each worker process')
  parser.add_argument('--async-collection',
                      action=argparse.BooleanOptionalAction, default=True,
                      help='collect training segments from whichever '
                           'matches are ready, so engine resets never stall '
                           'other matches')
  parser.add_argument('--async-promotion',
                      action=argparse.BooleanOptionalAction, default=True,
                      help='run promotion evaluation asynchronously too; it '
                           'stays deterministic (per-match episode quotas '
                           'and per-match sampling noise)')
  parser.add_argument('--gpu-filler', action=argparse.BooleanOptionalAction,
                      default=True,
                      help='keep reported GPU utilization up with small '
                           'side-stream matmuls inside the trainer (Torch '
                           'warns below 75%%); replaces the external '
                           'heartbeat, which time-sliced the GPU')
  parser.add_argument('--gpu-filler-kind', default='sleep',
                      choices=('sleep',),
                      help='sleep: a single-block spin kernel on one SM.  '
                           'Graph-replaying fillers are not offered: they '
                           'advance the CUDA generator and crash alongside '
                           'the captured update and actor graphs')
  parser.add_argument('--gpu-filler-matrix-size', type=int, default=1024,
                      help='accepted for older launchers; the spin filler '
                           'has no matrices')
  parser.add_argument('--overlap-collection',
                      action=argparse.BooleanOptionalAction, default=True,
                      help='keep collecting while the PPO update runs on the '
                           'GPU, acting with a copy of the policy synced after '
                           'every update (needs --async-collection)')
  parser.add_argument('--graph-update', action=argparse.BooleanOptionalAction,
                      default=True,
                      help='capture one whole-rollout PPO step (forward, '
                           'loss, backward, clip, Adam) as a CUDA graph per '
                           'rollout and replay it update_epochs times, so the '
                           'main thread keeps serving the environments')
  parser.add_argument('--graph-actor', action=argparse.BooleanOptionalAction,
                      default=True,
                      help='replay the acting forward and sampling as one '
                           'CUDA graph during async collection')
  parser.add_argument('--diagnostic-promotion-every', type=int, default=4,
                      help='run the greedy and self-play diagnostic '
                           'evaluations on every Nth promotion check, and on '
                           'any check that passes the gate; they never decide '
                           'promotion')
  parser.add_argument('--async-batch-workers', type=int, default=None,
                      help='workers per policy batch with --async-collection '
                           '(default: a third of the workers)')
  parser.add_argument('--compile', action=argparse.BooleanOptionalAction,
                      default=False,
                      help='torch.compile the BPTT forward pass')
  parser.add_argument('--compile-mode', default='max-autotune-no-cudagraphs')
  parser.add_argument('--log-interval-seconds', type=float, default=0.25,
                      help='log and compute diagnostics at most this often')
  parser.add_argument('--reuse-promotion-envs',
                      action=argparse.BooleanOptionalAction, default=True,
                      help='keep the promotion worker pools alive between '
                           'checks instead of rebuilding three per check')
  parser.add_argument('--benchmark-epochs', type=int, default=0,
                      help='time this many training epochs after a warmup, '
                           'print BENCHMARK json and exit; skips promotion')
  return parser


def _buffer_segments_per_agent(args):
  """Rollout rows per agent: 1 in lockstep, more when matches run ahead."""
  if not getattr(args, 'async_collection', False):
    return 1
  return 3 if getattr(args, 'overlap_collection', False) else 2


def build_config(args, num_agents):
  """One rollout segment per agent per epoch keeps BPTT aligned with rollout.

  `num_agents` must count every agent in every match the vecenv holds, not
  just the ones a single receive delivers.  PufferLib allocates the rollout
  buffer as segments x horizon and refuses `num_agents > segments`
  (pufferl.py: "Total agents N <= segments M"), because with more matches
  than workers any match may be the one that reports next.  Sizing this from
  a single batch is what made --envs-per-worker 2 abort at startup.
  """
  horizon = args.bptt_horizon
  config = _base_config()
  config.update({
      'adam_beta1': 0.9,
      'adam_beta2': 0.999,
      'adam_eps': 1e-5,
      'anneal_lr': args.anneal_lr,
      # Async collection keeps room for two segments per agent so matches
      # that run ahead keep writing while slow ones finish theirs.
      'batch_size': num_agents * horizon * _buffer_segments_per_agent(args),
      'bptt_horizon': horizon,
      'checkpoint_interval': 200,
      'clip_coef': args.clip_coef,
      'compile': args.compile,
      'compile_mode': args.compile_mode,
      'compile_fullgraph': False,
      'cpu_offload': False,
      'data_dir': os.path.abspath(args.data_dir),
      'device': args.device,
      'ent_coef': args.ent_coef,
      'env': 'gfootball',
      'gae_lambda': args.gae_lambda,
      'gamma': args.gamma,
      'learning_rate': args.learning_rate,
      'max_grad_norm': 0.5,
      # The minibatch is the whole epoch: one segment per agent.
      'minibatch_size': num_agents * horizon,
      'max_minibatch_size': num_agents * horizon,
      'optimizer': 'adam',
      'precision': 'float32',
      'seed': args.seed,
      'torch_deterministic': False,
      'total_timesteps': args.total_timesteps,
      'update_epochs': args.update_epochs,
      'use_rnn': True,
      'vf_clip_coef': 0.2,
      'vf_coef': args.vf_coef,
  })
  for unused in ('critic_learning_rate', 'prio_alpha', 'prio_beta0',
                 'vtrace_c_clip', 'vtrace_rho_clip'):
    config.pop(unused, None)
  if config['total_timesteps'] < config['batch_size']:
    raise ValueError('total_timesteps must cover at least one rollout batch')
  return config


def benchmark(trainer, epochs, warmup=5):
  """Steady-state rollout and update time per epoch, promotion excluded."""
  def synchronize():
    # Only the trainer's stream: the GPU filler has its own.
    if trainer.config['device'] == 'cuda':
      torch.cuda.current_stream().synchronize()

  for _ in range(warmup):
    trainer.evaluate()
    trainer.train()
  rollout = update = 0.0
  steps = trainer.global_step
  for _ in range(epochs):
    synchronize()
    started = time.perf_counter()
    trainer.evaluate()
    synchronize()
    middle = time.perf_counter()
    trainer.train()
    synchronize()
    rollout += middle - started
    update += time.perf_counter() - middle
  steps = trainer.global_step - steps
  trained = epochs * (
      trainer.blocks_per_epoch * trainer.agents_per_env
      if trainer.async_collection else trainer.segments
  ) * trainer.config['bptt_horizon']
  return {
      'trained_samples': trained,
      'trained_sps': trained / (rollout + update),
      'epochs': epochs,
      'agent_steps': steps,
      'rollout_seconds': rollout,
      'update_seconds': update,
      'sps': steps / (rollout + update),
      'rollout_sps': steps / rollout,
      'update_fraction': update / (rollout + update),
      'optimizer_steps': trainer.optimizer_steps,
  }


def main():
  args = build_parser().parse_args()
  advantage_schedule = args.env_name == ADVANTAGE_ENV_NAME
  if args.curriculum_levels is None:
    args.curriculum_levels = (
        ADVANTAGE_LEVELS if advantage_schedule else TOTAL_LEVELS)
  if args.attacker_only_levels is None:
    # The advantage schedule has every player active from level 0, so there
    # is no attacker-only prefix to configure.
    args.attacker_only_levels = 0 if advantage_schedule else ATTACKER_ONLY_LEVELS
  if args.scored_promotion_levels is None:
    args.scored_promotion_levels = args.curriculum_levels
  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA training requested but no GPU is visible')
  os.makedirs(args.data_dir, exist_ok=True)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  num_envs = args.num_workers * args.envs_per_worker
  env = make_vector_env(
      num_envs=num_envs, num_workers=args.num_workers, batch_size=num_envs,
      async_batch_workers=_async_batch_workers(args, args.num_workers),
      reserved_cpus=0, seed=args.seed,
      env_name=args.env_name, frame_stack=args.frame_stack,
      curriculum_levels=args.curriculum_levels,
      curriculum_window=args.curriculum_window,
      curriculum_success_threshold=args.curriculum_success_threshold,
      attacker_only_levels=args.attacker_only_levels,
      sort_players=args.sort_players,
      centralized_curriculum=True,
      observation=args.observation,
      # Shaping is a training signal only.  Promotion is judged on goals, so
      # the evaluation envs are built without it.
      ball_potential_scale=args.ball_potential,
      player_potential_scale=args.player_potential,
      player_potential_fade_speed=args.player_potential_fade_speed,
      potential_gamma=args.gamma,
      spawn=args.spawn,
      # A training distribution only: the gate keeps the level's own scenes.
      width_spawn_fraction=args.width_spawn_fraction)
  if not 0 <= args.start_level < args.curriculum_levels:
    raise ValueError('start-level must be inside the curriculum')
  # The launcher exports OMP_NUM_THREADS=1 so the env workers do not each
  # spawn a thread per core and thrash.  PyTorch reads that at import, which
  # also pins the update and the evaluation forward to a single core, and
  # those are the bulk of the job.  Raise it here, after the workers exist
  # (they pin themselves to one thread) so only this process widens.
  if args.torch_threads > 0:
    torch.set_num_threads(args.torch_threads)
  env.curriculum_level_value.value = args.start_level
  config = build_config(args, env.num_agents)
  print(json.dumps({
      'config': config,
      'curriculum_levels': args.curriculum_levels,
      'curriculum_success_threshold': args.curriculum_success_threshold,
      'curriculum_window': args.curriculum_window,
      'attacker_only_levels': args.attacker_only_levels,
      'frame_stack': args.frame_stack,
      'hidden_size': args.hidden_size,
      'observation': args.observation,
      'network': args.network,
      'env_name': args.env_name,
      'num_workers': args.num_workers,
      'envs_per_worker': args.envs_per_worker,
      'compile': args.compile,
      'sort_players': args.sort_players,
      'start_level': args.start_level,
      'promotion_interval': args.promotion_interval,
      'promotion_episodes': args.promotion_episodes,
      'promotion_min_episodes': promotion_min_episodes(args),
      'promotion_workers': args.promotion_workers,
      'promotion_worst_template_threshold': (
          args.promotion_worst_template_threshold),
      'promotion_early_abort_margin': args.promotion_early_abort_margin,
      'ball_potential': args.ball_potential,
      'player_potential': args.player_potential,
      'player_potential_fade_speed': args.player_potential_fade_speed,
      'spawn': args.spawn,
      'width_spawn_fraction': args.width_spawn_fraction,
      'frozen_defence_gate': args.frozen_defence_gate,
      'greedy_promotion_episodes': args.greedy_promotion_episodes,
      'selfplay_promotion_episodes': args.selfplay_promotion_episodes,
      'envs_per_worker': args.envs_per_worker,
      'async_collection': args.async_collection,
      'overlap_collection': args.overlap_collection,
      'diagnostic_promotion_every': args.diagnostic_promotion_every,
      'compile': args.compile,
      'reuse_promotion_envs': args.reuse_promotion_envs,
      'carry_lstm_memory': args.carry_lstm_memory,
      'frozen_defence_horizon': args.frozen_defence_horizon,
  }, sort_keys=True), flush=True)

  policy = FootballPolicy(env, hidden_size=args.hidden_size,
                          network=args.network).to(args.device)
  logger = None
  if args.wandb:
    logger = pufferl.WandbLogger({
        'wandb_project': args.wandb_project,
        'wandb_group': args.wandb_group,
        'tag': args.wandb_tag,
    })
  trainer = FootballPuffeRL(
      config, env, policy, logger=logger,
      log_interval_seconds=args.log_interval_seconds,
      overlap_collection=args.overlap_collection,
      graph_actor=args.graph_actor,
      graph_update=args.graph_update,
      carry_memory=args.carry_lstm_memory)
  filler = None
  if args.gpu_filler and args.device == 'cuda':
    from gfootball.examples.gpu_filler import GpuFiller
    filler = GpuFiller(matrix_size=args.gpu_filler_matrix_size,
                       kind=args.gpu_filler_kind).start()
  if args.benchmark_epochs:
    try:
      print('BENCHMARK ' + json.dumps(
          benchmark(trainer, args.benchmark_epochs), sort_keys=True),
          flush=True)
    except BaseException:
      import traceback
      traceback.print_exc()
      os._exit(1)
    # PufferLib's Multiprocessing close can hang at interpreter exit once the
    # workers are gone; a benchmark has nothing to save.
    os._exit(0)
  level_entry_epoch = 0
  promotion_checks = 0
  # The gate opponent is the policy as it was on entering the level.  Against
  # the live self-play opponent, "attackers score 60%" was a moving target:
  # every improvement in attack was matched by the same network's defence.
  frozen_defence_path = None
  if args.frozen_defence_gate:
    frozen_defence_path = os.path.join(args.data_dir, 'frozen_defence.pt')
    if args.frozen_defence_init:
      shutil.copyfile(args.frozen_defence_init, frozen_defence_path)
    else:
      save_policy_snapshot(policy, frozen_defence_path)
  early_abort = None
  if args.promotion_early_abort_margin > 0:
    early_abort = (
        promotion_min_episodes(args),
        args.curriculum_success_threshold - args.promotion_early_abort_margin)
  try:
    while trainer.global_step < config['total_timesteps']:
      if trainer.epoch % args.promotion_interval == 0:
        level = env.curriculum_level_value.value
        started = time.time()
        metrics = _run_promotion(
            args, trainer.uncompiled_policy, env.curriculum_level_value,
            level, args.device, args.promotion_episodes,
            frozen_defence_path=frozen_defence_path, early_abort=early_abort)
        scored_gate = promotion_passes(
            metrics, args.curriculum_success_threshold,
            args.promotion_worst_template_threshold)
        diagnostics = (
            scored_gate or args.diagnostic_promotion_every <= 1 or
            promotion_checks % args.diagnostic_promotion_every == 0)
        promotion_checks += 1
        if diagnostics and args.greedy_promotion_episodes > 0:
          metrics.update(_prefixed('greedy_', _run_promotion(
              args, trainer.uncompiled_policy, env.curriculum_level_value,
              level, args.device, args.greedy_promotion_episodes,
              frozen_defence_path=frozen_defence_path, greedy=True)))
        if (diagnostics and frozen_defence_path and
            args.selfplay_promotion_episodes > 0):
          metrics.update(_prefixed('selfplay_', _run_promotion(
              args, trainer.uncompiled_policy, env.curriculum_level_value,
              level, args.device, args.selfplay_promotion_episodes)))
        metrics['promotion_wall_seconds'] = time.time() - started
        epochs_here = trainer.epoch - level_entry_epoch
        # The later levels are too hard to gate on mastery, so they advance on
        # a schedule instead; a level still promotes early if it is mastered.
        timed_gate = (level >= args.scored_promotion_levels and
                      epochs_here >= args.timed_promotion_epochs)
        advanced = (level < args.curriculum_levels - 1 and
                    (scored_gate or timed_gate))
        if advanced:
          env.curriculum_level_value.value = level + 1
          level_entry_epoch = trainer.epoch
          if frozen_defence_path:
            # The next level's opponent is the policy that just cleared this
            # one.  Keep a per-level copy so any level can be re-evaluated.
            save_policy_snapshot(trainer.uncompiled_policy, os.path.join(
                args.data_dir, 'frozen_defence_level{}.pt'.format(level + 1)))
            save_policy_snapshot(
                trainer.uncompiled_policy, frozen_defence_path)
        trainer.record_promotion(level, metrics, advanced)
        print('PROMOTION {}'.format(json.dumps({
            'level': level, 'advanced': advanced,
            'gate': 'frozen' if frozen_defence_path else 'selfplay',
            'reason': ('scored' if scored_gate else
                       'timed' if timed_gate else 'none'),
            'epochs_on_level': epochs_here,
            'optimizer_steps': trainer.optimizer_steps, **metrics,
        }, sort_keys=True)), flush=True)
      trainer.evaluate()
      trainer.train()
  finally:
    if filler is not None:
      filler.stop()
    close_promotion_pools()
    model_path = trainer.close()
    if logger is not None:
      logger.close(model_path)
    print('Saved model: {}'.format(model_path), flush=True)


if __name__ == '__main__':
  main()
