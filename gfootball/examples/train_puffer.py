#!/usr/bin/env python3
"""Recurrent PPO self-play training for the football curriculum.

One shared trunk feeds an LSTM; the actor and critic read the same recurrent
state and train under a single optimizer.  The vector env always exposes 22
agent rows, but the curriculum only controls a few of them at the early
levels, so every loss is masked down to the rows that are actually playing.
"""

import argparse
from collections import defaultdict
import json
import math
import os
import time

import numpy as np
import torch

import pufferlib
from pufferlib import pufferl
import pufferlib.pytorch

from gfootball.env.puffer_env import make_vector_env
from gfootball.env import football_action_set
from gfootball.curriculum import (
    ADVANTAGE_ENV_NAME, ADVANTAGE_LEVELS, ATTACKER_ONLY_LEVELS,
    SPAWN_TEMPLATE_COUNT, TOTAL_LEVELS)


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
  metrics = {
      'promotion_success_rate': success_rate,
      'promotion_worst_template_success_rate': (
          min(template_rates.values()) if template_rates else 0.0),
      'promotion_templates_covered': float(len(template_rates)),
  }
  metrics.update({
      'promotion_template_{}_success_rate'.format(template):
      template_rates.get(template, 0.0)
      for template in range(SPAWN_TEMPLATE_COUNT)
  })
  return metrics


def promotion_passes(metrics, success_threshold, worst_template_threshold):
  return (
      metrics['promotion_templates_covered'] == SPAWN_TEMPLATE_COUNT and
      metrics['promotion_success_rate'] >= success_threshold and
      metrics['promotion_worst_template_success_rate'] >=
      worst_template_threshold)


def evaluate_promotion(policy, vecenv, episodes, seed, device, recurrent_horizon):
  """Evaluate with the same recurrent windows as PufferLib training rollouts."""
  if recurrent_horizon < 1:
    raise ValueError('recurrent_horizon must be positive')
  observations, _ = vecenv.reset(seed=seed)
  generator = torch.Generator(device=device).manual_seed(seed)
  state = {'lstm_h': None, 'lstm_c': None, 'done': None}
  rows = []
  action_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
  active_logits_rows = []
  decisions = 0
  steps = 0
  was_training = policy.training
  policy.eval()
  try:
    while len(rows) < episodes:
      # PuffeRL.evaluate starts every rollout window from zero memory. Keep
      # this clock independent of episode resets, just like the collector.
      if steps % recurrent_horizon == 0:
        state['lstm_h'] = state['lstm_c'] = None
      observation_tensor = torch.as_tensor(observations, device=device)
      active = observation_tensor.flatten(1).abs().sum(dim=-1) > 0
      with torch.no_grad():
        logits, _ = policy.forward_eval(observation_tensor, state)
        active_logits_rows.append(logits[active].float().cpu())
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
  metrics = promotion_statistics(rows)
  diagnostics = policy_diagnostics(torch.cat(active_logits_rows))
  metrics.update({
      'promotion_recurrent_horizon': float(recurrent_horizon),
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


class RunningNormalizer(torch.nn.Module):
  """Per-feature running standardization of observations.

  simple115v2 is wildly unbalanced for this task: measured on the curriculum,
  the twenty-one other players' relative positions carry ~8x the magnitude and
  ~5x the variance of the ball's relative position, which is the one feature
  that actually matters.  Standardizing each feature puts them on equal footing
  so the first layer does not have to learn a 8x weight ratio to compensate.
  """

  def __init__(self, size, epsilon=1e-4, clip=10.0):
    super().__init__()
    self.clip = clip
    self.register_buffer('mean', torch.zeros(size))
    self.register_buffer('var', torch.ones(size))
    self.register_buffer('count', torch.full((), float(epsilon)))

  @torch.no_grad()
  def update(self, observations):
    """Chan et al. parallel variance update from one rollout."""
    batch = observations.reshape(-1, observations.shape[-1]).float()
    if batch.shape[0] < 2:
      return
    batch_count = batch.new_tensor(float(batch.shape[0]))
    batch_mean = batch.mean(0)
    batch_var = batch.var(0, unbiased=False)
    delta = batch_mean - self.mean
    total = self.count + batch_count
    combined = (self.var * self.count + batch_var * batch_count +
                delta.square() * self.count * batch_count / total)
    self.mean.copy_(self.mean + delta * batch_count / total)
    self.var.copy_(combined / total)
    self.count.copy_(total)

  def forward(self, observations):
    normalized = (observations - self.mean) * torch.rsqrt(self.var + 1e-8)
    return normalized.clamp(-self.clip, self.clip)


class FootballPolicy(torch.nn.Module):
  """Shared trunk into an LSTM, then an actor head and a value head."""

  is_continuous = False

  def __init__(self, env, hidden_size=256):
    super().__init__()
    observation_size = int(np.prod(env.single_observation_space.shape))
    self.hidden_size = hidden_size
    self.normalizer = RunningNormalizer(observation_size)
    self.encoder = torch.nn.Sequential(
        pufferlib.pytorch.layer_init(
            torch.nn.Linear(observation_size, hidden_size)),
        torch.nn.ReLU(),
        pufferlib.pytorch.layer_init(
            torch.nn.Linear(hidden_size, hidden_size)),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(hidden_size),
    )
    self.cell = torch.nn.LSTMCell(hidden_size, hidden_size)
    for name, parameter in self.cell.named_parameters():
      if 'bias' in name:
        torch.nn.init.constant_(parameter, 0)
      else:
        torch.nn.init.orthogonal_(parameter, 1.0)
    self.actor = pufferlib.pytorch.layer_init(
        torch.nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
    self.critic = pufferlib.pytorch.layer_init(
        torch.nn.Linear(hidden_size, 1), std=1.0)

  def _recurrent_state(self, state, rows, reference):
    hidden = state.get('lstm_h')
    cell = state.get('lstm_c')
    if hidden is None or cell is None:
      hidden = reference.new_zeros(rows, self.hidden_size, dtype=torch.float32)
      cell = torch.zeros_like(hidden)
    return hidden.float(), cell.float()

  @staticmethod
  def _reset_finished(hidden, cell, done):
    """Zero the recurrent state of any row whose episode just ended."""
    if done is None:
      return hidden, cell
    keep = (~done.bool()).to(hidden.dtype).unsqueeze(-1)
    return hidden * keep, cell * keep

  def forward_eval(self, observations, state):
    """Advance one environment step for every agent row."""
    observations = observations.float()
    hidden, cell = self._recurrent_state(
        state, observations.shape[0], observations)
    hidden, cell = self._reset_finished(hidden, cell, state.get('done'))
    encoded = self.encoder(self.normalizer(observations))
    hidden, cell = self.cell(encoded, (hidden, cell))
    state['lstm_h'] = hidden
    state['lstm_c'] = cell
    return self.actor(hidden), self.critic(hidden).squeeze(-1)

  def forward(self, observations, state=None):
    """Backprop through time over a (segments, horizon) minibatch."""
    if observations.dim() == 2:
      return self.forward_eval(observations, dict(state or {}))
    state = dict(state or {})
    segments, horizon = observations.shape[:2]
    observations = observations.float()
    encoded = self.encoder(self.normalizer(
        observations.reshape(segments * horizon, -1))).view(
            segments, horizon, self.hidden_size)
    hidden, cell = self._recurrent_state(state, segments, observations)
    done = state.get('done')
    outputs = []
    for step in range(horizon):
      hidden, cell = self._reset_finished(
          hidden, cell, None if done is None else done[:, step])
      hidden, cell = self.cell(encoded[:, step], (hidden, cell))
      outputs.append(hidden)
    hidden = torch.stack(outputs, dim=1).reshape(
        segments * horizon, self.hidden_size)
    state['lstm_h'] = hidden.detach()
    state['lstm_c'] = cell.detach()
    return self.actor(hidden), self.critic(hidden).view(segments, horizon)


class FootballPuffeRL(pufferl.PuffeRL):
  """PPO restricted to the agent rows the curriculum actually controls."""

  def __init__(self, config, vecenv, policy, logger=None):
    super().__init__(config, vecenv, policy, logger=logger)
    self.optimizer_steps = 0
    self.active_steps = 0
    self.last_active_steps = 0
    self.last_active_log_time = time.time()
    self.promotion_metrics = {}

  def record_promotion(self, level, metrics, advanced):
    self.promotion_metrics = {
        'promotion_level': float(level),
        'promotion_advanced': float(advanced),
        **metrics,
    }

  def train(self):
    profile = self.profile
    epoch = self.epoch
    profile('train', epoch)
    profile('train_misc', epoch, nest=True)
    config = self.config
    device = config['device']
    losses = defaultdict(float)

    rollout_values = self.values.clone()
    advantages, returns, valid = generalized_advantages(
        rollout_values, self.rewards, self.terminals,
        config['gamma'], config['gae_lambda'])
    active = self.observations.flatten(2).abs().sum(dim=-1) > 0
    trainable = active & valid
    segment_index = trainable.any(dim=1).nonzero().flatten()
    num_segments = int(segment_index.numel())
    if num_segments == 0:
      raise RuntimeError('rollout contains no controlled agents')
    segments_per_minibatch = min(self.minibatch_segments, num_segments)
    num_minibatches = max(1, math.ceil(
        config['update_epochs'] * num_segments / segments_per_minibatch))

    # Statistics come only from training rollouts, never from the held-out
    # promotion evaluation, so evaluation stays a clean measurement.
    self.uncompiled_policy.normalizer.update(self.observations[trainable])
    losses['observation_scale_mean'] = float(
        self.uncompiled_policy.normalizer.var.sqrt().mean().item())
    losses['pre_update_explained_variance'] = explained_variance(
        rollout_values[trainable], returns[trainable])

    for _ in range(num_minibatches):
      profile('train_copy', epoch)
      order = torch.randperm(num_segments, device=device)
      index = segment_index[order[:segments_per_minibatch]]
      mask = trainable[index]
      observations = self.observations[index]
      actions = self.actions[index]

      profile('train_forward', epoch)
      logits, new_values = self.policy(
          observations, {'done': self.terminals[index]})
      _, new_logprobs, entropy = pufferlib.pytorch.sample_logits(
          logits, action=actions)

      profile('train_misc', epoch)
      new_logprobs = new_logprobs.view(mask.shape)[mask]
      entropy = entropy.view(mask.shape)[mask]
      old_logprobs = self.logprobs[index][mask]
      old_values = rollout_values[index][mask]
      mb_returns = returns[index][mask]
      mb_values = new_values[mask]

      log_ratio = new_logprobs - old_logprobs
      ratio = log_ratio.exp()
      with torch.no_grad():
        losses['old_approx_kl'] += (-log_ratio).mean().item()
        losses['approx_kl'] += ((ratio - 1) - log_ratio).mean().item()
        losses['clipfrac'] += (
            (ratio - 1).abs() > config['clip_coef']).float().mean().item()
        losses['importance'] += ratio.mean().item()

      mb_advantages = normalize_advantages(advantages[index][mask])
      policy_loss = torch.max(
          -mb_advantages * ratio,
          -mb_advantages * ratio.clamp(
              1 - config['clip_coef'], 1 + config['clip_coef'])).mean()
      clipped_values = old_values + (mb_values - old_values).clamp(
          -config['vf_clip_coef'], config['vf_clip_coef'])
      value_loss = 0.5 * torch.max(
          (mb_values - mb_returns) ** 2,
          (clipped_values - mb_returns) ** 2).mean()
      entropy_loss = entropy.mean()
      loss = (policy_loss + config['vf_coef'] * value_loss -
              config['ent_coef'] * entropy_loss)

      profile('learn', epoch)
      self.optimizer.zero_grad()
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
          self.policy.parameters(), config['max_grad_norm'])
      self.optimizer.step()
      self.optimizer_steps += 1

      losses['policy_loss'] += policy_loss.item()
      losses['value_loss'] += value_loss.item()
      losses['entropy'] += entropy_loss.item()
      losses['gradient_norm'] += gradient_norm.item()
      losses['minibatch_transitions'] += float(mask.sum().item())

    profile('train_misc', epoch)
    for name in ('policy_loss', 'value_loss', 'entropy', 'gradient_norm',
                 'old_approx_kl', 'approx_kl', 'clipfrac', 'importance',
                 'minibatch_transitions'):
      losses[name] /= num_minibatches

    if config['anneal_lr']:
      self.scheduler.step()

    active_transitions = int(trainable.sum().item())
    self.active_steps += active_transitions
    now = time.time()
    losses.update({
        'optimizer_steps': float(self.optimizer_steps),
        'ppo_minibatches': float(num_minibatches),
        'active_agent_fraction': active.float().mean().item(),
        'active_transitions': float(active_transitions),
        'active_segments': float(num_segments),
        'advantage_mean': advantages[trainable].mean().item(),
        'advantage_std': advantages[trainable].std(unbiased=False).item(),
        'positive_reward_fraction': (
            self.rewards[active] > 0).float().mean().item(),
        'negative_reward_fraction': (
            self.rewards[active] < 0).float().mean().item(),
        'learning_rate': self.optimizer.param_groups[0]['lr'],
        'active_SPS': (
            (self.active_steps - self.last_active_steps) /
            max(1e-6, now - self.last_active_log_time)),
    })
    self.last_active_steps = self.active_steps
    self.last_active_log_time = now

    # Post-update health has to replay whole segments: a recurrent critic
    # scored from a zeroed hidden state is not the critic that acts.
    with torch.no_grad():
      index = segment_index[:min(64, num_segments)]
      sample_mask = trainable[index]
      post_logits, post_values = self.policy(
          self.observations[index], {'done': self.terminals[index]})
      post_logits = post_logits.view(*sample_mask.shape, -1)[sample_mask]
      for name, value in policy_diagnostics(post_logits).items():
        losses[name] = value.item()
      losses['post_update_explained_variance'] = explained_variance(
          post_values[sample_mask], returns[index][sample_mask])
    losses['explained_variance'] = losses['pre_update_explained_variance']

    active_actions = self.actions[active]
    action_fractions = [
        (active_actions == index).float().mean().item()
        for index in range(len(ACTION_NAMES))]
    for index, name in enumerate(ACTION_NAMES):
      losses['action_{}_fraction'.format(name)] = action_fractions[index]
    losses['max_action_fraction'] = max(action_fractions)
    losses['shot_fraction'] = action_fractions[SHOT_ACTION]
    losses.update(self.promotion_metrics)

    profile.end()
    logs = None
    self.epoch += 1
    done_training = self.global_step >= config['total_timesteps']
    if (done_training or self.global_step == 0 or
        time.time() > self.last_log_time + 0.25):
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


def _make_promotion_env(args, curriculum_level_value):
  return make_vector_env(
      num_envs=args.promotion_workers, num_workers=args.promotion_workers,
      batch_size=args.promotion_workers, reserved_cpus=0,
      seed=args.seed + 1000000, env_name=args.env_name,
      frame_stack=args.frame_stack,
      curriculum_levels=args.curriculum_levels,
      curriculum_window=args.promotion_episodes + 1,
      curriculum_success_threshold=args.curriculum_success_threshold,
      attacker_only_levels=args.attacker_only_levels,
      curriculum_level_value=curriculum_level_value,
      sort_players=args.sort_players,
      curriculum_evaluation=True)


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-workers', type=int, default=30)
  parser.add_argument('--total-timesteps', type=int, default=1_000_000_000)
  parser.add_argument('--env-name', default='11_vs_11_curriculum',
                      choices=('11_vs_11_curriculum', ADVANTAGE_ENV_NAME))
  parser.add_argument('--curriculum-levels', type=int, default=None)
  parser.add_argument('--start-level', type=int, default=0,
                      help='begin at this curriculum level instead of 0')
  parser.add_argument('--curriculum-window', type=int, default=20)
  parser.add_argument('--curriculum-success-threshold', type=float, default=0.6)
  parser.add_argument('--attacker-only-levels', type=int, default=None)
  parser.add_argument('--promotion-interval', type=int, default=50)
  parser.add_argument('--promotion-episodes', type=int, default=256)
  parser.add_argument('--promotion-workers', type=int, default=30)
  parser.add_argument('--promotion-worst-template-threshold', type=float,
                      default=0.4)
  parser.add_argument('--scored-promotion-levels', type=int, default=4,
                      help='levels below this advance only on the score gate')
  parser.add_argument('--timed-promotion-epochs', type=int, default=200,
                      help='at or above --scored-promotion-levels, also '
                           'advance after this many epochs on a level')
  parser.add_argument('--anneal-lr', action=argparse.BooleanOptionalAction,
                      default=False)
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  parser.add_argument('--ent-coef', type=float, default=0.01)
  parser.add_argument('--vf-coef', type=float, default=0.5)
  parser.add_argument('--clip-coef', type=float, default=0.2)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--gae-lambda', type=float, default=0.95)
  parser.add_argument('--update-epochs', type=int, default=4)
  parser.add_argument('--bptt-horizon', type=int, default=32)
  parser.add_argument('--minibatch-segments', type=int, default=16)
  parser.add_argument('--hidden-size', type=int, default=256)
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
  return parser


def build_config(args, num_agents):
  """One rollout segment per agent per epoch keeps BPTT aligned with rollout."""
  horizon = args.bptt_horizon
  config = _base_config()
  config.update({
      'adam_beta1': 0.9,
      'adam_beta2': 0.999,
      'adam_eps': 1e-5,
      'anneal_lr': args.anneal_lr,
      'batch_size': num_agents * horizon,
      'bptt_horizon': horizon,
      'checkpoint_interval': 200,
      'clip_coef': args.clip_coef,
      'compile': False,
      'cpu_offload': False,
      'data_dir': os.path.abspath(args.data_dir),
      'device': args.device,
      'ent_coef': args.ent_coef,
      'env': 'gfootball',
      'gae_lambda': args.gae_lambda,
      'gamma': args.gamma,
      'learning_rate': args.learning_rate,
      'max_grad_norm': 0.5,
      'minibatch_size': args.minibatch_segments * horizon,
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
  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA training requested but no GPU is visible')
  os.makedirs(args.data_dir, exist_ok=True)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  env = make_vector_env(
      num_envs=args.num_workers, num_workers=args.num_workers,
      batch_size=args.num_workers, reserved_cpus=0, seed=args.seed,
      env_name=args.env_name, frame_stack=args.frame_stack,
      curriculum_levels=args.curriculum_levels,
      curriculum_window=args.curriculum_window,
      curriculum_success_threshold=args.curriculum_success_threshold,
      attacker_only_levels=args.attacker_only_levels,
      sort_players=args.sort_players,
      centralized_curriculum=True)
  if not 0 <= args.start_level < args.curriculum_levels:
    raise ValueError('start-level must be inside the curriculum')
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
      'minibatch_segments': args.minibatch_segments,
      'env_name': args.env_name,
      'num_workers': args.num_workers,
      'sort_players': args.sort_players,
      'start_level': args.start_level,
      'promotion_interval': args.promotion_interval,
      'promotion_episodes': args.promotion_episodes,
      'promotion_workers': args.promotion_workers,
      'promotion_worst_template_threshold': (
          args.promotion_worst_template_threshold),
  }, sort_keys=True), flush=True)

  policy = FootballPolicy(env, hidden_size=args.hidden_size).to(args.device)
  logger = None
  if args.wandb:
    logger = pufferl.WandbLogger({
        'wandb_project': args.wandb_project,
        'wandb_group': args.wandb_group,
        'tag': args.wandb_tag,
    })
  trainer = FootballPuffeRL(config, env, policy, logger=logger)
  level_entry_epoch = 0
  try:
    while trainer.global_step < config['total_timesteps']:
      if trainer.epoch % args.promotion_interval == 0:
        level = env.curriculum_level_value.value
        # Build a fresh promotion env per evaluation.  Reusing one deadlocks:
        # evaluate_promotion returns as soon as it has enough episodes, so the
        # vecenv is left mid-flight with outstanding send/recv pairs and the
        # next reset() never completes.
        promotion_env = _make_promotion_env(args, env.curriculum_level_value)
        try:
          metrics = evaluate_promotion(
              trainer.uncompiled_policy, promotion_env,
              args.promotion_episodes,
              args.seed + 1000000 + 10000 * level, args.device,
              recurrent_horizon=args.bptt_horizon)
        finally:
          promotion_env.close()
        scored_gate = promotion_passes(
            metrics, args.curriculum_success_threshold,
            args.promotion_worst_template_threshold)
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
        trainer.record_promotion(level, metrics, advanced)
        print('PROMOTION {}'.format(json.dumps({
            'level': level, 'advanced': advanced,
            'reason': ('scored' if scored_gate else
                       'timed' if timed_gate else 'none'),
            'epochs_on_level': epochs_here,
            'optimizer_steps': trainer.optimizer_steps, **metrics,
        }, sort_keys=True)), flush=True)
      trainer.evaluate()
      trainer.train()
  finally:
    model_path = trainer.close()
    if logger is not None:
      logger.close(model_path)
    print('Saved model: {}'.format(model_path), flush=True)


if __name__ == '__main__':
  main()
