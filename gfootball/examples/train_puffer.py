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
# Re-exported: scripts and tests import the policy from this module.
from gfootball.env.puffer_policy import (  # noqa: F401
    FootballPolicy, RunningNormalizer, save_policy_snapshot)
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


def evaluate_promotion(policy, vecenv, episodes, seed, device,
                       recurrent_horizon, greedy=False, early_abort=None):
  """Evaluate with the same recurrent windows as PufferLib training rollouts.

  `greedy` takes the argmax action instead of sampling, which shows whether
  the network has actually learned a shot underneath a near-uniform
  distribution.  `early_abort` is an optional `(after_episodes, min_rate)`
  pair: once that many episodes are in, a success rate below the floor ends
  the evaluation early, since it cannot pass the gate and the remaining
  episodes would only refine a number nobody acts on.
  """
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
      if steps % recurrent_horizon == 0:
        state['lstm_h'] = state['lstm_c'] = None
      observation_tensor = torch.as_tensor(observations, device=device)
      active = observation_tensor.flatten(1).abs().sum(dim=-1) > 0
      with torch.no_grad():
        logits, _ = policy.forward_eval(observation_tensor, state)
        active_logits_rows.append(logits[active].float().cpu())
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
  metrics = promotion_statistics(rows)
  diagnostics = policy_diagnostics(torch.cat(active_logits_rows))
  metrics.update({
      'promotion_recurrent_horizon': float(recurrent_horizon),
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


def _make_promotion_env(args, curriculum_level_value, frozen_defence_path=None):
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
      curriculum_evaluation=True,
      frozen_defence_path=frozen_defence_path,
      frozen_defence_horizon=args.bptt_horizon)


def _run_promotion(args, policy, level_value, level, device, episodes,
                   frozen_defence_path=None, greedy=False, early_abort=None):
  """One held-out evaluation on a fresh promotion env.

  A fresh env per evaluation is deliberate.  Reusing one deadlocks:
  evaluate_promotion returns as soon as it has enough episodes, so the vecenv
  is left mid-flight with outstanding send/recv pairs and the next reset()
  never completes.
  """
  promotion_env = _make_promotion_env(args, level_value, frozen_defence_path)
  try:
    return evaluate_promotion(
        policy, promotion_env, episodes,
        args.seed + 1000000 + 10000 * level, device,
        recurrent_horizon=args.bptt_horizon, greedy=greedy,
        early_abort=early_abort)
  finally:
    promotion_env.close()


def _prefixed(prefix, metrics):
  return {prefix + name: value for name, value in metrics.items()}


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
  # Promotion evaluation used to run 256 episodes every 25 epochs, which was
  # about 40% of a ten-hour job.  Larger and rarer is cheaper and less noisy.
  parser.add_argument('--promotion-interval', type=int, default=100)
  parser.add_argument('--promotion-episodes', type=int, default=512)
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
  parser.add_argument('--greedy-promotion-episodes', type=int, default=128,
                      help='extra argmax-action evaluation per promotion '
                           'check (diagnostic only); 0 disables')
  parser.add_argument('--selfplay-promotion-episodes', type=int, default=128,
                      help='extra live self-play evaluation per promotion '
                           'check when the gate is frozen-defence '
                           '(diagnostic only, comparable to older runs); '
                           '0 disables')
  parser.add_argument('--scored-promotion-levels', type=int, default=4,
                      help='levels below this advance only on the score gate')
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
      'promotion_early_abort_margin': args.promotion_early_abort_margin,
      'frozen_defence_gate': args.frozen_defence_gate,
      'greedy_promotion_episodes': args.greedy_promotion_episodes,
      'selfplay_promotion_episodes': args.selfplay_promotion_episodes,
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
  # The gate opponent is the policy as it was on entering the level.  Against
  # the live self-play opponent, "attackers score 60%" was a moving target:
  # every improvement in attack was matched by the same network's defence.
  frozen_defence_path = None
  if args.frozen_defence_gate:
    frozen_defence_path = os.path.join(args.data_dir, 'frozen_defence.pt')
    save_policy_snapshot(policy, frozen_defence_path)
  early_abort = None
  if args.promotion_early_abort_margin > 0:
    early_abort = (
        max(SPAWN_TEMPLATE_COUNT * 4, args.promotion_episodes // 4),
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
        if args.greedy_promotion_episodes > 0:
          metrics.update(_prefixed('greedy_', _run_promotion(
              args, trainer.uncompiled_policy, env.curriculum_level_value,
              level, args.device, args.greedy_promotion_episodes,
              frozen_defence_path=frozen_defence_path, greedy=True)))
        if frozen_defence_path and args.selfplay_promotion_episodes > 0:
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
    model_path = trainer.close()
    if logger is not None:
      logger.close(model_path)
    print('Saved model: {}'.format(model_path), flush=True)


if __name__ == '__main__':
  main()
