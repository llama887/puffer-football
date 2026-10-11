#!/usr/bin/env python3
"""Recurrent PPO self-play training on ADR scenes (gfootball/adr.py).

One shared trunk feeds an LSTM; the actor and critic read the same recurrent
state and train under a single optimizer.  The vector env always exposes 22
agent rows, but some are hidden (a frozen probe defence, the first step of a
scene with held directions), so every loss is masked down to the rows the
learner actually played.
"""

import argparse
import contextlib
from collections import defaultdict
import json
import math
import os
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

import pufferlib
from pufferlib import pufferl
import pufferlib.pytorch

from gfootball import adr
from gfootball.env.defence import IDLE as IDLE_DEFENCE
from gfootball.env.defence import RANDOM as RANDOM_DEFENCE
from gfootball.env.puffer_env import make_vector_env
from gfootball.env import football_action_set
# Re-exported: scripts and tests import the policy from this module.
from gfootball.env.puffer_policy import (  # noqa: F401
    FootballPolicy, PolicyPlayer, episode_batch_sizes, episode_pieces,
    save_policy_snapshot)


ACTION_NAMES = tuple(
    str(action) for action in football_action_set.action_set_dict['default'])
SHOT_ACTION = ACTION_NAMES.index('shot')

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


def _evaluation_metrics(rows, action_counts, decisions, active_logits_rows,
                        recurrent_horizon):
  diagnostics = policy_diagnostics(torch.cat(active_logits_rows))
  action_counts = action_counts.cpu()
  decisions = max(1, int(decisions))
  return {
      'success_rate': sum(float(row['success']) for row in rows) / len(rows),
      # 0: memory ran through whole episodes.
      'recurrent_horizon': float(recurrent_horizon or 0),
      'episodes': float(len(rows)),
      'mean_episode_length': sum(
          row['episode_length'] for row in rows) / len(rows),
      'possession_fraction': sum(
          row.get('possession_fraction', 0.0) for row in rows) / len(rows),
      'mean_ball_advance': sum(
          row.get('mean_ball_advance', 0.0) for row in rows) / len(rows),
      'max_action_fraction': action_counts.max().item() / decisions,
      'shot_fraction': action_counts[SHOT_ACTION].item() / decisions,
      **{name: value.item() for name, value in diagnostics.items()},
      **{
          'action_{}_fraction'.format(name):
          action_counts[index].item() / decisions
          for index, name in enumerate(ACTION_NAMES)
      },
  }


def evaluate_episodes(policy, vecenv, episodes, seed, device,
                      recurrent_horizon):
  """Play `episodes` held-out episodes on a pool whose matches run at their
  own pace, and summarize them (_evaluation_metrics).

  Every match contributes its first ceil(episodes / matches) episodes and
  nothing else.  Taking the first `episodes` to finish overall would favour
  short episodes, and a short episode is usually a goal.

  The result does not depend on timing: each match's engine, defence and
  scene draws are seeded per match (reset(seed) gives match i the seed
  seed + i, which its episode infos report as env_seed), the policy plays
  through a PolicyPlayer (memory window `recurrent_horizon` per match,
  sampling noise per match and step), and everything reported is taken from
  a fixed number of each match's first episodes.
  """
  if getattr(vecenv, 'envs_per_worker', 1) != 1:
    raise ValueError('async evaluation needs one match per worker, so that '
                     'match i is reset with seed + i')
  per_env = vecenv.driver_env.num_agents
  matches = vecenv.num_agents // per_env
  quota = -(-episodes // matches)
  player = PolicyPlayer(policy, matches, per_env, recurrent_horizon, seed,
                        device=device)
  vecenv.async_reset(seed)
  # Per match: finished episodes, and the action counts and active logits
  # of each finished episode and of the one in progress.
  taken = [[] for _ in range(matches)]
  episode_actions = [[] for _ in range(matches)]
  episode_logits = [[] for _ in range(matches)]
  current_actions = np.zeros((matches, len(ACTION_NAMES)), dtype=np.int64)
  current_logits = [[] for _ in range(matches)]
  was_training = policy.training
  policy.eval()
  try:
    while any(len(rows) < quota for rows in taken):
      observations, _, terminals, _, infos, agent_ids, _ = vecenv.recv()
      agent_ids = np.asarray(agent_ids)
      match_ids = agent_ids[::per_env] // per_env
      for info in infos:
        if 'success' not in info:
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
      active = np.abs(np.asarray(observations)).reshape(
          len(agent_ids), -1).sum(axis=-1) > 0
      actions, logits = player.act(observations, match_ids, terminals)
      active_rows = active.reshape(len(match_ids), per_env)
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
  rows = [row for match in range(matches) for row in taken[match][:quota]]
  action_counts = torch.as_tensor(sum(
      (counts for match in range(matches)
       for counts in episode_actions[match][:quota]),
      np.zeros(len(ACTION_NAMES), dtype=np.int64)))
  active_logits_rows = [
      step for match in range(matches)
      for episode in episode_logits[match][:quota] for step in episode]
  decisions = max(1, int(action_counts.sum()))
  return _evaluation_metrics(rows, action_counts, decisions,
                             active_logits_rows, recurrent_horizon)


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


class SegmentCollector:
  """Collects `horizon`-step segments from matches that run at their own pace.

  PuffeRL's collector steps every env exactly `horizon` times per epoch in a
  fixed order, so one env in an engine reset holds up the rest.  Here each
  match fills its own `horizon`-step segment (one row per agent) as fast as
  it runs, into blocks of the rollout buffers with room for several segments
  per match.  An epoch trains on exactly one segment per match's worth of
  completed blocks; segments in progress, and any completed beyond that,
  carry over.  By default recurrent memory starts from zero at every segment
  and every episode end, exactly as in PuffeRL's rollout.  With carry_memory
  it clears only at episode ends: each segment continues its match's memory,
  and the memory it started from is kept per buffer row for training
  (starting_memory).  Either way BPTT replays what the policy saw.

  With overlap, collection continues while the update runs on the GPU
  (collect_during_update): a copy of the policy acts on its own CUDA stream
  and is synced after every update, so an action is at most one update
  stale.  Its log-prob is stored with the transition, so PPO's importance
  ratio stays exact.  With graph_actor the acting forward replays as one CUDA graph.

  `rollout` holds the buffers (observations, actions, logprobs, rewards,
  terminals, values; one row per agent segment); `on_batch(infos, steps)`
  receives every batch's episode infos and its agent steps; `profile(name)`
  times the phases.
  """

  def __init__(self, vecenv, policy, rollout, horizon, device,
               carry_memory=False, overlap=False, graph_actor=False,
               amp_context=None, on_batch=None, profile=None):
    self.vecenv = vecenv
    self.policy = policy
    self.rollout = rollout
    self.horizon = int(horizon)
    self.device = device
    self.carry_memory = bool(carry_memory)
    self.overlap = bool(overlap)
    self.graph_actor = bool(graph_actor)
    self._amp_context = amp_context or contextlib.nullcontext
    self._on_batch = on_batch or (lambda infos, steps: None)
    self._profile = profile or (lambda name: None)
    self.agents_per_env = vecenv.driver_env.num_agents
    total_agents = vecenv.num_agents
    segments = rollout.observations.shape[0]
    self.num_matches = total_agents // self.agents_per_env
    self.num_blocks = segments // self.agents_per_env
    self.blocks_per_epoch = self.num_matches
    needed = (3 if self.overlap else 2) * self.num_matches
    if self.num_blocks < needed:
      raise ValueError('async collection needs {} segment blocks, the buffer '
                       'has {}'.format(needed, self.num_blocks))
    self.free_blocks = list(range(self.num_blocks))
    self.complete_blocks = []
    self.training_blocks = []
    self.match_block = np.full(self.num_matches, -1, dtype=np.int64)
    self.match_step = np.zeros(self.num_matches, dtype=np.int64)
    hidden = policy.hidden_size
    self.agent_h = torch.zeros(total_agents, hidden, device=device)
    self.agent_c = torch.zeros(total_agents, hidden, device=device)
    if self.carry_memory:
      # Per buffer row: the actor's memory when that row's segment began.
      self.segment_h = torch.zeros(segments, hidden, device=device)
      self.segment_c = torch.zeros(segments, hidden, device=device)
    # Per buffer row: whether it belongs to the epoch being trained.
    self.row_training = torch.zeros(segments, dtype=torch.bool,
                                    device=device)
    self.agent_offsets = np.arange(self.agents_per_env)
    self.discarded_steps = 0
    self.dropped_segments = 0
    self.graphed_actor = None
    self.acting_policy = policy
    self.acting_stream = None
    if self.overlap:
      import copy
      self.acting_policy = copy.deepcopy(policy)
      for parameter in self.acting_policy.parameters():
        parameter.requires_grad_(False)
      if torch.device(device).type == 'cuda':
        _, highest = torch.cuda.Stream.priority_range()
        self.acting_stream = torch.cuda.Stream(device=device,
                                               priority=highest)

  def collect_epoch(self):
    """Recycle the rows last trained on; collect one epoch of segments."""
    self.free_blocks.extend(self.training_blocks)
    self.training_blocks = []
    while len(self.complete_blocks) < self.blocks_per_epoch:
      self._collect_batch()

  def collect_during_update(self):
    """With overlap: keep the environments busy until the update queued on
    the GPU has finished, then give the actor the new weights.  Without
    overlap there is nothing to do."""
    if not self.overlap:
      return
    finished = torch.cuda.Event()
    finished.record()
    while not finished.query():
      if self._workers_ready():
        self._collect_batch()
    self._sync_acting_policy()

  def claim_training_rows(self):
    """Hand exactly one epoch of the newest completed segments to PPO.

    Returns their buffer rows (a fixed count) and a per-row mask of them;
    the mask is reused, so it holds only until the next claim.  Call it
    before reading the buffers: it waits for the actor's stream.
    With overlap the environments can outrun the learner: older surplus
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
    rows = torch.as_tensor(rows, device=self.row_training.device)
    self.row_training.zero_()
    self.row_training[rows] = True
    if self.acting_stream is not None:
      # Everything the actor wrote must land before PPO reads it.
      torch.cuda.current_stream().wait_stream(self.acting_stream)
    return rows, self.row_training

  def _sync_acting_policy(self):
    """Copy the trained weights (and normalizer statistics) to the actor."""
    if self.acting_policy is self.policy:
      return
    main = torch.cuda.current_stream() if self.acting_stream else None
    if main is not None:
      # Nothing the actor has queued may read weights mid-copy.
      main.wait_stream(self.acting_stream)
    with torch.no_grad():
      for target, source in zip(self.acting_policy.state_dict().values(),
                                self.policy.state_dict().values()):
        target.copy_(source)
    if main is not None:
      self.acting_stream.wait_stream(main)

  def starting_memory(self, index):
    """The window-forward state that starts buffer rows `index` as collected.

    Empty when memory restarts at every segment (the window forward starts
    from zero by itself).  With carry_memory, the memory each row's segment
    started from plus the done flags that clear it where the segment opens
    on a new episode.  Shapes are fixed by `index`, so a captured update can
    read it.
    """
    if not self.carry_memory:
      return {}
    return {'done': self.rollout.terminals[index],
            'lstm_h': self.segment_h[index], 'lstm_c': self.segment_c[index]}

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
    profile = self._profile
    device = self.device
    horizon = self.horizon
    per_env = self.agents_per_env
    rollout = self.rollout

    profile('env')
    o, r, d, t, info, agent_ids, mask = self.vecenv.recv()

    profile('eval_misc')
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

      profile('eval_copy')
      o_device = torch.as_tensor(o).to(device)
      r = torch.as_tensor(r).to(device)
      d = torch.as_tensor(d).to(device)

      profile('eval_forward')
      with torch.no_grad(), self._amp_context():
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

      profile('eval_copy')
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
          rollout.observations[rows, steps] = o_device[keep_rows].to(
              rollout.observations.dtype)
          rollout.actions[rows, steps] = action[keep_rows].to(
              rollout.actions.dtype)
          rollout.logprobs[rows, steps] = logprob[keep_rows].to(
              rollout.logprobs.dtype)
          rollout.rewards[rows, steps] = r[keep_rows].to(rollout.rewards.dtype)
          rollout.terminals[rows, steps] = d[keep_rows].to(
              rollout.terminals.dtype)
          rollout.values[rows, steps] = value.flatten()[keep_rows].to(
              rollout.values.dtype)
        action = action.cpu().numpy()

    recorded = matches[recording]
    self.match_step[recorded] += 1
    for match in recorded[self.match_step[recorded] >= horizon]:
      self.complete_blocks.append(int(self.match_block[match]))
      self.match_block[match] = -1

    profile('eval_misc')
    self._on_batch(info, int(mask.sum()))

    profile('env')
    self.vecenv.send(action)


class PpoUpdate:
  """PPO steps over the trainable rows of a rollout, eager or as a CUDA graph.

  update() stages the epoch's values, advantages, returns and trainable mask
  and takes update_epochs steps, each over the whole rollout (~1000 segments
  at 46 matches, ~0.6 GB of activations), so update_epochs is exactly the
  number of optimizer steps per rollout.  Rows and steps that are not
  trainable are masked out of every loss.  `starting_memory(index)` gives
  the state each row's segment started from (empty when memory restarts at
  every segment).

  With graph, the step is a captured CUDA graph replayed update_epochs
  times.  The main thread also serves the environments while the update
  runs; launching every kernel from Python kept it busy ~40 ms per epoch
  (collection fell from ~280k to ~140k steps/s), and capturing a new graph
  each rollout still cost ~30 ms.  So the graph reads its rows and episode
  layout from fixed buffers, and the layout has a fixed capacity per step
  (episode_pieces pads it exactly): each rollout copies its layout in and
  replays.  The graph is captured again only when a rollout needs more
  slots than the capacity, or leaves over half of it unused.  Capturing does
  not execute, so the very first step runs eagerly to create Adam's state
  first.  The optimizer must then be capturable, with its learning rate in
  a GPU tensor (set_learning_rate fills it in place).
  """

  # Loss statistics accumulated on the GPU inside the step and read back
  # once, when the epoch is logged.
  LOSS_NAMES = ('policy_loss', 'value_loss', 'entropy', 'gradient_norm',
                'old_approx_kl', 'approx_kl', 'clipfrac', 'importance',
                'minibatch_transitions')

  def __init__(self, policy, optimizer, rollout, config, graph=False,
               starting_memory=None, capture_policy=None):
    self.policy = policy
    self.optimizer = optimizer
    self.rollout = rollout
    self.config = config
    self.graph = bool(graph)
    self._starting_memory = starting_memory or (lambda index: {})
    # The graph captures the uncompiled module (torch.compile adds nothing
    # inside a CUDA graph and does not capture cleanly).
    self._capture_policy = capture_policy or policy
    self.optimizer_steps = 0
    self.captures = 0
    self.totals = None
    self._epoch_values = None
    self._update_graph = None
    self._update_pool = None
    self._update_capacity = None

  def update(self, rows, values, advantages, returns, trainable):
    """Train on buffer rows `rows`; returns the loss totals (LOSS_NAMES),
    summed over the update epochs."""
    self._stage(values, advantages, returns, trainable)
    done = self.rollout.terminals[rows]
    epochs = self.config['update_epochs']
    if not self.graph:
      pieces = episode_pieces(done)
      for _ in range(epochs):
        self._step(self.policy, rows, pieces)
      return self.totals.clone()
    needed = episode_batch_sizes(done)
    capacity = self._update_capacity
    if (self._update_graph is None or
        rows.numel() != self._update_rows.numel() or
        bool((needed > capacity).any()) or
        2 * int(needed.sum()) < int(capacity.sum())):
      capacity = (torch.ceil(needed * 1.25 / 64) * 64).long().clamp_min(64)
      self._capture(rows, episode_pieces(done, capacity))
      first = self._warmup_steps
    else:
      order, inverse, _ = episode_pieces(done, capacity)
      self._update_rows.copy_(rows)
      self._update_order.copy_(order)
      self._update_inverse.copy_(inverse)
      first = 0
    for _ in range(first, epochs):
      self._update_graph.replay()
      self.optimizer_steps += 1
    return self.totals.clone()

  def _stage(self, values, advantages, returns, trainable):
    """Copy the epoch into fixed buffers, which a captured step reads."""
    if self._epoch_values is None:
      self._epoch_values = torch.zeros_like(values)
      self._epoch_advantages = torch.zeros_like(values)
      self._epoch_returns = torch.zeros_like(values)
      self._epoch_weight = torch.zeros_like(values)
      self.totals = torch.zeros(len(self.LOSS_NAMES), device=values.device)
    self._epoch_values.copy_(values)
    self._epoch_advantages.copy_(advantages)
    self._epoch_returns.copy_(returns)
    self._epoch_weight.copy_(trainable)
    self.totals.zero_()

  def _step(self, policy, index, pieces):
    """One PPO gradient step on the segments in `index`.

    Forward (BPTT over whole segments, using the precomputed episode_pieces
    layout of those segments), masked PPO losses, backward, gradient clip,
    Adam.
    """
    config = self.config
    rollout = self.rollout
    clip = config['clip_coef']
    mask = self._epoch_weight[index]
    count = mask.sum()
    denominator = count.clamp_min(1)

    def masked_mean(values):
      return (values * mask).sum() / denominator

    logits, new_values = policy(
        rollout.observations[index],
        {'episode_pieces': pieces, **self._starting_memory(index)})
    _, new_logprobs, entropy = pufferlib.pytorch.sample_logits(
        logits, action=rollout.actions[index])
    new_logprobs = new_logprobs.view(mask.shape)
    entropy = entropy.view(mask.shape)
    old_values = self._epoch_values[index]
    mb_returns = self._epoch_returns[index]
    log_ratio = new_logprobs - rollout.logprobs[index]
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
      self.totals += torch.stack((
          policy_loss, value_loss, entropy_loss, gradient_norm,
          masked_mean(-log_ratio),
          masked_mean((ratio - 1) - log_ratio),
          masked_mean(((ratio - 1).abs() > clip).float()),
          masked_mean(ratio),
          count)).detach()

  def _capture(self, rows, pieces):
    """Capture one PPO step reading the fixed update buffers.

    The very first capture is preceded by one real, eager step (on a side
    stream) so Adam's state and the cuBLAS/cuDNN workspaces exist; that
    step counts as the rollout's first (_warmup_steps).  Later captures
    reuse the same memory pool.
    """
    policy = self._capture_policy
    order, inverse, capacity = pieces
    self._update_rows = rows.clone()
    self._update_order = order.clone()
    self._update_inverse = inverse.clone()
    self._update_capacity = capacity
    static = (self._update_order, self._update_inverse, capacity)
    self._warmup_steps = 0
    if self._update_pool is None:
      side = torch.cuda.Stream()
      side.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(side):
        self._step(policy, self._update_rows, static)
      torch.cuda.current_stream().wait_stream(side)
      self._update_pool = torch.cuda.graph_pool_handle()
      self._warmup_steps = 1
    # Gradients are allocated inside the graph's pool during capture.
    self.optimizer.zero_grad(set_to_none=True)
    self._update_graph = None
    graph = torch.cuda.CUDAGraph()
    steps = self.optimizer_steps
    # thread_local: the GPU filler thread keeps launching its own kernels.
    with torch.cuda.graph(graph, pool=self._update_pool,
                          capture_error_mode='thread_local'):
      self._step(policy, self._update_rows, static)
    self.optimizer_steps = steps
    self._update_graph = graph
    self.captures += 1


class FootballPuffeRL(pufferl.PuffeRL):
  """PPO self-play: a SegmentCollector feeds a PpoUpdate.

  PuffeRL owns the rollout buffers, the optimizer, logging and checkpoints;
  the collector fills the buffers from matches running at their own pace
  (the vecenv must be asynchronous), and the update trains on the rows the
  learner actually played.
  """

  def __init__(self, config, vecenv, policy, logger=None,
               log_interval_seconds=0.25, overlap_collection=False,
               graph_actor=False, graph_update=False, carry_memory=False):
    super().__init__(config, vecenv, policy, logger=logger)
    if not getattr(vecenv, 'is_async', False):
      raise ValueError('FootballPuffeRL collects asynchronously; build the '
                       'vecenv with async_batch_workers')
    cuda = torch.device(config['device']).type == 'cuda'
    # PufferLib's epoch-based scheduler is unused: train() sets the rate from
    # training progress (see cosine_learning_rate).  A captured update
    # replays Adam as recorded, so Adam keeps its step count on the GPU
    # (capturable) and the learning rate lives in a GPU tensor that
    # set_learning_rate fills in place.
    graph_update = bool(graph_update and cuda)
    if graph_update:
      self.optimizer = torch.optim.Adam(
          self.uncompiled_policy.parameters(),
          lr=torch.tensor(float(config['learning_rate']),
                          device=config['device']),
          betas=(config['adam_beta1'], config['adam_beta2']),
          eps=config['adam_eps'], capturable=True)
    # The ADR randomizer that records probe results (set by main).
    self.adr = None
    # PuffeRL's rollout buffers, all the collector and the update share.
    rollout = SimpleNamespace(
        observations=self.observations, actions=self.actions,
        logprobs=self.logprobs, rewards=self.rewards,
        terminals=self.terminals, values=self.values)
    self.collector = SegmentCollector(
        vecenv, self.uncompiled_policy, rollout, config['bptt_horizon'],
        config['device'], carry_memory=carry_memory,
        overlap=overlap_collection and cuda,
        graph_actor=graph_actor and cuda, amp_context=lambda: self.amp_context,
        on_batch=self._record_batch,
        profile=lambda name: self.profile(name, self.epoch))
    self.ppo = PpoUpdate(
        self.policy, self.optimizer, rollout, config, graph=graph_update,
        starting_memory=self.collector.starting_memory,
        capture_policy=self.uncompiled_policy)
    self.active_steps = torch.zeros((), dtype=torch.long,
                                    device=config['device'])
    self.last_active_steps = 0
    self.last_active_log_time = time.time()
    self.evaluation_metrics = {}
    # Diagnostics are only computed on epochs that are logged, so this also
    # sets how often the trainer pays for them.
    self.log_interval_seconds = float(log_interval_seconds)

  def _record_batch(self, infos, steps):
    """Count a collected batch's agent steps; log its finished episodes."""
    self.global_step += steps
    if self.adr is not None:
      record_adr_probes(self.adr, infos)
    for i in infos:
      for k, v in pufferlib.unroll_nested_dict(i):
        if isinstance(v, np.ndarray):
          v = v.tolist()
        elif isinstance(v, (list, tuple)):
          self.stats[k].extend(v)
        else:
          self.stats[k].append(v)

  def evaluate(self):
    profile = self.profile
    profile('eval', self.epoch)
    profile('eval_misc', self.epoch, nest=True)
    self.collector.collect_epoch()
    profile.end()
    return self.stats

  def train(self):
    profile = self.profile
    epoch = self.epoch
    profile('train', epoch)
    profile('train_misc', epoch, nest=True)
    config = self.config
    collector = self.collector
    done_training = (
        self.global_step >= config['total_timesteps'])
    log_epoch = (done_training or self.global_step == 0 or
                 time.time() > self.last_log_time + self.log_interval_seconds)
    losses = {}
    # Before any buffer read: waits for the actor's stream.
    rows, training_rows = collector.claim_training_rows()

    rollout_values = self.values.clone()
    advantages, returns, valid = generalized_advantages(
        rollout_values, self.rewards, self.terminals,
        config['gamma'], config['gae_lambda'])
    active = self.observations.flatten(2).abs().sum(dim=-1) > 0
    # Only this epoch's completed segments; the rest carry over.
    trainable = active & valid & training_rows[:, None]
    segment_index = trainable.any(dim=1).nonzero().flatten()
    num_segments = int(segment_index.numel())
    if num_segments == 0:
      raise RuntimeError('rollout contains no controlled agents')
    num_minibatches = config['update_epochs']

    # Statistics come only from training rollouts, never from the held-out
    # evaluation, so evaluation stays a clean measurement.
    self.uncompiled_policy.normalizer.update(self.observations[trainable])
    if log_epoch:
      losses['observation_scale_mean'] = float(
          self.uncompiled_policy.normalizer.var.sqrt().mean().item())
      losses['pre_update_explained_variance'] = explained_variance(
          rollout_values[trainable], returns[trainable])

    if config['anneal_lr']:
      set_learning_rate(self.optimizer, cosine_learning_rate(
          config['learning_rate'], self.global_step,
          config['total_timesteps']))
    # Every loss is a masked mean over the (segment, step) grid of the
    # epoch's whole set of rows (a fixed count, so a captured update keeps
    # its shapes), the same number as indexing out the trainable rows first
    # without the data-dependent shapes that force a GPU sync on every step.
    totals = self.ppo.update(rows, rollout_values, advantages, returns,
                             trainable)

    collector.collect_during_update()

    profile('train_misc', epoch)
    self.active_steps += trainable.sum()

    if log_epoch:
      for name, value in zip(PpoUpdate.LOSS_NAMES,
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
          'optimizer_steps': float(self.ppo.optimizer_steps),
          'update_captures': float(self.ppo.captures),
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
            {'done': self.terminals[index],
             **collector.starting_memory(index)})
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
      losses.update(self.evaluation_metrics)

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


def _async_batch_workers(args, num_workers):
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


def defence_memory_window(args):
  """Steps a snapshot defence keeps memory before restarting it, wherever it
  plays (training probes and evaluation pools alike).

  --frozen-defence-horizon, for a snapshot trained with another window (0:
  whole episodes); by default memory_window, the way this run trains.
  """
  if args.frozen_defence_horizon is None:
    return memory_window(args)
  return args.frozen_defence_horizon or None


class Evaluator:
  """Scores a policy on fixed ADR test scenes against a chosen defence.

  A test difficulty d draws every setting from its adr set_difficulty range,
  so the scenes depend on d and the seed alone, never on the ranges training
  has reached, and the scores compare across runs.  The defence is None (the
  policy itself), 'random' (uniform random actions) or a snapshot path.

  One worker pool per defence is built on first use and kept until close():
  building one starts a game engine in every worker, 10-20 s apiece.  Each
  pool draws its scenes from its own shared ADR ranges, set to the requested
  difficulty before every evaluation, and every evaluation reseeds the pool
  in place, which replays exactly the episodes a fresh pool would.  Pools
  play without shaping and without probes.  A snapshot defence keeps
  memory for defence_memory_window steps.  Its arguments are checked when it
  is built, so main builds it before any env worker starts.
  """

  def __init__(self, args, make_pool=make_vector_env):
    # Pools run asynchronously, so at least one worker must be left out of
    # every batch (PufferLib needs 0 < batch workers < workers).
    if args.evaluation_workers < 2:
      raise ValueError('--evaluation-workers must be at least 2')
    self.difficulties = [
        float(x) for x in args.evaluation_difficulties.split(',') if x]
    self._args = args
    self._make_pool = make_pool
    self._pools = {}
    if any(not 0 <= difficulty <= 1 for difficulty in self.difficulties):
      raise ValueError('--evaluation-difficulties must lie in [0, 1]')
    self._memory_window = memory_window(args)
    self._defence_window = defence_memory_window(args)

  def score(self, policy, difficulty, defence=None):
    """_evaluation_metrics of at least --evaluation-episodes episodes at
    `difficulty` against `defence` (ceil(episodes / workers) per match)."""
    args = self._args
    if defence not in self._pools:
      shared = adr.shared_ranges()
      workers = args.evaluation_workers
      self._pools[defence] = (self._make_pool(
          num_envs=workers, num_workers=workers, batch_size=workers,
          # Its own batch size, not --async-batch-workers (training's).
          reserved_cpus=0, async_batch_workers=max(1, workers // 3),
          seed=args.seed + 1000000, frame_stack=args.frame_stack,
          sort_players=args.sort_players, observation=args.observation,
          frozen_defence_path=defence,
          frozen_defence_horizon=self._defence_window,
          adr_shared=shared, adr_probe_fraction=0.0),
          adr.AutomaticDomainRandomization(shared=shared))
    pool, ranges = self._pools[defence]
    # The workers read these ranges at every episode reset.
    ranges.set_difficulty(difficulty)
    return evaluate_episodes(
        policy, pool, args.evaluation_episodes,
        args.seed + 1000000 + int(round(1000 * difficulty)), args.device,
        recurrent_horizon=self._memory_window)

  def baseline(self, policy):
    """Goal rate against the base random policy at every test difficulty.

    The opponent and the scenes never change, so the numbers compare across
    runs and over time.
    """
    return {'vs_random/difficulty_{}/success_rate'.format(difficulty):
            self.score(policy, difficulty, RANDOM_DEFENCE)['success_rate']
            for difficulty in self.difficulties}

  def close(self):
    while self._pools:
      _, (pool, _) = self._pools.popitem()
      pool.close()


def record_adr_probes(randomizer, infos):
  """Feed every finished ADR probe episode in `infos` to the randomizer.

  A probe's info names the setting and edge it pinned and whether the
  attack scored against the probe defence; other episodes are skipped.
  Returns how many probes were recorded.
  """
  recorded = 0
  for info in infos:
    if info.get('adr_probe'):
      randomizer.record(int(info['adr_parameter']), int(info['adr_edge']),
                        info['success'])
      recorded += 1
  return recorded


def refresh_adr_probe_defence(args, evaluator, policy, probe_path):
  """Replace the ADR probe defence with the learner once it outgrows it.

  ADR probes judge each range edge by how often the attack scores against
  the probe defence.  Random defenders cannot sense defensive settings
  (blockers, formation), and a fixed snapshot stops sensing them once the
  learner's own defence is stronger, so the ranges saturate.  This plays the
  learner's attack against the probe defence and against its own
  (self-play) defence on the same scenes, at each of the evaluator's test
  difficulties; when the probe defence concedes at least
  --adr-probe-refresh-margin more, the learner's current weights overwrite
  `probe_path` (atomically: env workers reload it at their next probe
  episode).

  Returns metrics: both success rates, averaged over the difficulties, and
  whether the defence was replaced.
  """
  rates = {'vs_probe': [], 'vs_self': []}
  for difficulty in evaluator.difficulties:
    for name, defence in (('vs_probe', probe_path), ('vs_self', None)):
      rates[name].append(
          evaluator.score(policy, difficulty, defence)['success_rate'])
  vs_probe, vs_self = float(np.mean(rates['vs_probe'])), float(
      np.mean(rates['vs_self']))
  replaced = vs_probe - vs_self >= args.adr_probe_refresh_margin
  if replaced:
    save_policy_snapshot(policy, probe_path)
  return {'adr_probe/vs_probe': vs_probe, 'adr_probe/vs_self': vs_self,
          'adr_probe/replaced': float(replaced)}


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-workers', type=int, default=30)
  parser.add_argument('--torch-threads', type=int, default=0,
                      help='threads for the update and the evaluation '
                           'forward; 0 leaves PyTorch alone. The launcher '
                           'pins OMP_NUM_THREADS=1 for the env workers, '
                           'which otherwise pins this process too')
  parser.add_argument('--total-timesteps', type=int, default=1_000_000_000)
  # Evaluation every --evaluation-interval epochs: the ADR ranges, the
  # random-defender score at each test difficulty and, with a snapshot probe
  # defence, the refresh check.
  parser.add_argument('--evaluation-interval', type=int, default=100)
  parser.add_argument('--evaluation-workers', type=int, default=30,
                      help='worker processes per evaluation pool (at least '
                           '2; one pool per defence stays alive)')
  parser.add_argument('--evaluation-difficulties', default='0.25,0.5,1.0',
                      help='test difficulties (adr set_difficulty: every '
                           'setting from its start value out to this share '
                           'of the way to its limits) at which every check '
                           'scores the policy against random defenders; '
                           'empty disables')
  parser.add_argument('--evaluation-episodes', type=int, default=64,
                      help='at least this many episodes per difficulty '
                           'and defence: ceil(episodes / workers) per match')
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
  parser.add_argument('--adr-probe-fraction', type=float, default=0.2,
                      help='share of episodes that probe one setting edge '
                           'against the probe defence')
  parser.add_argument('--adr-probe-defence', default=RANDOM_DEFENCE,
                      help="who defends ADR probes: 'random' (uniform "
                           "random defenders), 'idle' (no action) or a "
                           'policy snapshot, copied to '
                           '<data-dir>/adr_probe_defence.pt and replaced by '
                           'the learner when it outgrows it '
                           '(refresh_adr_probe_defence)')
  parser.add_argument('--adr-probe-refresh-interval', type=int, default=1000,
                      help='epochs between checks that may replace a '
                           'snapshot probe defence; 0 never replaces it')
  parser.add_argument('--adr-probe-refresh-margin', type=float, default=0.1,
                      help='replace the probe defence once the learner '
                           'scores at least this much more against it than '
                           'against its own defence')
  parser.add_argument('--adr-buffer-size', type=int, default=64,
                      help='probe results per edge before it may move')
  parser.add_argument('--adr-low-threshold', type=float, default=0.3,
                      help='probe success at or below this shrinks the edge')
  parser.add_argument('--adr-high-threshold', type=float, default=0.6,
                      help='probe success at or above this grows the edge')
  parser.add_argument('--update-epochs', type=int, default=4,
                      help='optimizer steps per rollout; each step trains on '
                           'the whole rollout (the minibatch is the rollout)')
  parser.add_argument('--bptt-horizon', type=int, default=32)
  parser.add_argument('--carry-lstm-memory',
                      action=argparse.BooleanOptionalAction, default=False,
                      help='keep recurrent memory across rollout segments '
                           '(cleared only at episode ends) and train each '
                           'segment from the memory it started with, instead '
                           'of restarting memory every --bptt-horizon '
                           'steps')
  parser.add_argument('--frozen-defence-horizon', type=int, default=None,
                      help='steps a frozen evaluation or probe opponent '
                           'keeps memory before restarting it (0: whole '
                           'episodes); by default the same as this run trains '
                           'with')
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
  parser.add_argument('--gpu-filler', action=argparse.BooleanOptionalAction,
                      default=True,
                      help='keep reported GPU utilization up with small '
                           'side-stream matmuls inside the trainer (Torch '
                           'warns below 75%%); replaces the external '
                           'heartbeat, which time-sliced the GPU')
  parser.add_argument('--overlap-collection',
                      action=argparse.BooleanOptionalAction, default=True,
                      help='keep collecting while the PPO update runs on the '
                           'GPU, acting with a copy of the policy synced after '
                           'every update')
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
  parser.add_argument('--async-batch-workers', type=int, default=None,
                      help='workers per collected batch (default: a third '
                           'of the workers); matches run at their own pace, '
                           'so engine resets never stall the others')
  parser.add_argument('--compile', action=argparse.BooleanOptionalAction,
                      default=False,
                      help='torch.compile the BPTT forward pass')
  parser.add_argument('--compile-mode', default='max-autotune-no-cudagraphs')
  parser.add_argument('--log-interval-seconds', type=float, default=0.25,
                      help='log and compute diagnostics at most this often')
  parser.add_argument('--benchmark-epochs', type=int, default=0,
                      help='time this many training epochs after a warmup, '
                           'print BENCHMARK json and exit; skips evaluation')
  return parser


def _buffer_segments_per_agent(args):
  """Rollout rows per agent: matches run ahead of the epoch being trained.

  Room for the epoch being trained and one segment in progress per match,
  plus, with overlap, the segments completed while the update runs.
  """
  return 3 if getattr(args, 'overlap_collection', False) else 2


def build_config(args, num_agents):
  """Rollout buffers of whole segments per agent keep BPTT aligned with the
  rollout: an epoch trains one segment per agent, and the buffer holds two
  or three per agent (_buffer_segments_per_agent).

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
  """Steady-state rollout and update time per epoch, evaluation excluded."""
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
  collector = trainer.collector
  trained = (epochs * collector.blocks_per_epoch * collector.agents_per_env *
             trainer.config['bptt_horizon'])
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
      'optimizer_steps': trainer.ppo.optimizer_steps,
  }


def main():
  args = build_parser().parse_args()
  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA training requested but no GPU is visible')
  os.makedirs(args.data_dir, exist_ok=True)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # Built first: it checks its arguments before any env worker starts.
  evaluator = Evaluator(args)
  # ADR ranges live in shared memory: env workers sample from them, the
  # trainer moves them from probe results.
  adr_shared = adr.shared_ranges()
  adr_probe_defence = args.adr_probe_defence
  snapshot_probe = adr_probe_defence not in (RANDOM_DEFENCE, IDLE_DEFENCE)
  if snapshot_probe:
    # Refreshes are checked on the evaluation epochs only, on the evaluation
    # difficulties.
    if args.adr_probe_refresh_interval % args.evaluation_interval:
      raise ValueError('--adr-probe-refresh-interval must be a multiple '
                       'of --evaluation-interval')
    if args.adr_probe_refresh_interval > 0 and not evaluator.difficulties:
      raise ValueError('refreshing a snapshot probe defence needs '
                       '--evaluation-difficulties (or '
                       '--adr-probe-refresh-interval 0)')
    # A copy, so a refresh replaces it without touching the original.
    adr_probe_defence = os.path.join(args.data_dir, 'adr_probe_defence.pt')
    shutil.copyfile(args.adr_probe_defence, adr_probe_defence)
  num_envs = args.num_workers * args.envs_per_worker
  env = make_vector_env(
      num_envs=num_envs, num_workers=args.num_workers, batch_size=num_envs,
      async_batch_workers=_async_batch_workers(args, args.num_workers),
      reserved_cpus=0, seed=args.seed, frame_stack=args.frame_stack,
      sort_players=args.sort_players, observation=args.observation,
      # Shaping is a training signal only.  Evaluation is judged on goals,
      # so the evaluation envs are built without it.
      ball_potential_scale=args.ball_potential,
      player_potential_scale=args.player_potential,
      player_potential_fade_speed=args.player_potential_fade_speed,
      potential_gamma=args.gamma,
      adr_shared=adr_shared,
      adr_probe_fraction=args.adr_probe_fraction,
      adr_probe_defence=adr_probe_defence,
      frozen_defence_horizon=defence_memory_window(args))
  # The launcher exports OMP_NUM_THREADS=1 so the env workers do not each
  # spawn a thread per core and thrash.  PyTorch reads that at import, which
  # also pins the update and the evaluation forward to a single core, and
  # those are the bulk of the job.  Raise it here, after the workers exist
  # (they pin themselves to one thread) so only this process widens.
  if args.torch_threads > 0:
    torch.set_num_threads(args.torch_threads)
  config = build_config(args, env.num_agents)
  print(json.dumps({
      'config': config,
      'frame_stack': args.frame_stack,
      'hidden_size': args.hidden_size,
      'observation': args.observation,
      'network': args.network,
      'num_workers': args.num_workers,
      'envs_per_worker': args.envs_per_worker,
      'compile': args.compile,
      'sort_players': args.sort_players,
      'evaluation_interval': args.evaluation_interval,
      'evaluation_workers': args.evaluation_workers,
      'evaluation_difficulties': args.evaluation_difficulties,
      'evaluation_episodes': args.evaluation_episodes,
      'ball_potential': args.ball_potential,
      'player_potential': args.player_potential,
      'player_potential_fade_speed': args.player_potential_fade_speed,
      'adr_probe_fraction': args.adr_probe_fraction,
      'adr_buffer_size': args.adr_buffer_size,
      'adr_thresholds': [args.adr_low_threshold, args.adr_high_threshold],
      'adr_probe_defence': args.adr_probe_defence,
      'adr_probe_refresh_interval': args.adr_probe_refresh_interval,
      'adr_probe_refresh_margin': args.adr_probe_refresh_margin,
      'overlap_collection': args.overlap_collection,
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
  # Probe results arrive in the episode infos the collector hands over.
  trainer.adr = adr.AutomaticDomainRandomization(
      shared=adr_shared, buffer_size=args.adr_buffer_size,
      low_threshold=args.adr_low_threshold,
      high_threshold=args.adr_high_threshold)
  filler = None
  if args.gpu_filler and args.device == 'cuda':
    from gfootball.examples.gpu_filler import GpuFiller
    filler = GpuFiller().start()
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
  try:
    while trainer.global_step < config['total_timesteps']:
      if trainer.epoch % args.evaluation_interval == 0:
        started = time.time()
        metrics = trainer.adr.metrics()
        metrics.update(evaluator.baseline(trainer.uncompiled_policy))
        if (snapshot_probe and trainer.epoch > 0 and
            args.adr_probe_refresh_interval > 0 and
            trainer.epoch % args.adr_probe_refresh_interval == 0):
          metrics.update(refresh_adr_probe_defence(
              args, evaluator, trainer.uncompiled_policy, adr_probe_defence))
          if metrics['adr_probe/replaced']:
            # Keep every probe defence the run has used.
            shutil.copyfile(adr_probe_defence, os.path.join(
                args.data_dir,
                'adr_probe_defence_epoch{}.pt'.format(trainer.epoch)))
        metrics['evaluation_wall_seconds'] = time.time() - started
        trainer.evaluation_metrics = metrics
        print('ADR {}'.format(json.dumps(
            {'epoch': trainer.epoch,
             'optimizer_steps': trainer.ppo.optimizer_steps,
             **metrics}, sort_keys=True)), flush=True)
      trainer.evaluate()
      trainer.train()
  finally:
    if filler is not None:
      filler.stop()
    # The model is saved first: closing the evaluation pools can hang.
    try:
      model_path = trainer.close()
      if logger is not None:
        logger.close(model_path)
      print('Saved model: {}'.format(model_path), flush=True)
    finally:
      evaluator.close()


if __name__ == '__main__':
  main()
