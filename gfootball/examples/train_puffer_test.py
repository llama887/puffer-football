"""Correctness checks for the recurrent football PPO loop."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium
import numpy as np
import torch

from gfootball.examples.train_puffer import (
    ACTION_NAMES, SHOT_ACTION, FootballPolicy, FootballPuffeRL, GraphedActor,
    build_config, build_parser,
    evaluate_promotion, evaluate_promotion_async, explained_variance,
    generalized_advantages, normalize_advantages,
    policy_diagnostics, promotion_passes, promotion_statistics)


def _env(observation_size=115):
  return SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-1, high=1, shape=(observation_size,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(len(ACTION_NAMES)))


def test_shot_action_index_matches_action_set():
  assert ACTION_NAMES[SHOT_ACTION] == 'shot'


def test_gae_uses_next_step_rewards_and_respects_episode_boundaries():
  values = torch.zeros(1, 8)
  rewards = torch.zeros(1, 8)
  rewards[0, 3] = 1
  rewards[0, 6] = 1
  terminals = torch.zeros(1, 8)
  terminals[0, 3] = 1
  terminals[0, 6] = 1

  advantages, returns, valid = generalized_advantages(
      values, rewards, terminals, gamma=0.5, gae_lambda=1)

  expected = [[0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 0.0, 0.0]]
  assert advantages.tolist() == expected
  assert returns.tolist() == expected
  assert valid.tolist() == [[True, True, True, True, True, True, True, False]]
  normalized = normalize_advantages(advantages[valid])
  assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
  assert torch.isclose(
      normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_explained_variance_is_one_for_a_perfect_critic():
  targets = torch.tensor([0.0, 0.5, 1.0])
  assert explained_variance(targets, targets) == 1
  assert math.isnan(explained_variance(torch.ones(3), torch.ones(3)))


def test_bptt_forward_matches_stepwise_rollout_with_episode_resets():
  """Training and rollout must see the same recurrent state."""
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16).eval()
  segments, horizon = 3, 7
  observations = torch.randn(segments, horizon, 115)
  done = torch.zeros(segments, horizon)
  done[0, 3] = 1
  done[2, 1] = 1
  done[2, 5] = 1

  with torch.no_grad():
    logits, values = policy(observations, {'done': done})
    state = {'lstm_h': None, 'lstm_c': None, 'done': None}
    stepwise_logits, stepwise_values = [], []
    for step in range(horizon):
      state['done'] = done[:, step]
      step_logits, step_values = policy.forward_eval(
          observations[:, step], state)
      stepwise_logits.append(step_logits)
      stepwise_values.append(step_values)

  stepwise_logits = torch.stack(stepwise_logits, dim=1).reshape(
      segments * horizon, -1)
  stepwise_values = torch.stack(stepwise_values, dim=1)
  assert torch.allclose(logits, stepwise_logits, atol=1e-5)
  assert torch.allclose(values, stepwise_values, atol=1e-5)


def test_promotion_matches_training_across_rollout_and_episode_boundaries():
  """Promotion must replay the same recurrent windows as PufferLib rollout."""
  for horizon in (3, 32):
    torch.manual_seed(7)
    steps = 2 * horizon + 5
    observations = torch.randn(1, steps + 1, 115)
    done = torch.zeros(1, steps + 1, dtype=torch.bool)
    done[0, [2, horizon + 2, steps]] = True

    class RecordingPolicy(FootballPolicy):
      def forward_eval(self, observations, state):
        logits, values = super().forward_eval(observations, state)
        self.recorded_logits.append(logits.clone())
        self.recorded_values.append(values.clone())
        return logits, values

    class TrajectoryEnv:
      def reset(self, seed):
        self.step_index = 0
        self.episode_length = 0
        return observations[:, 0].numpy(), []

      def step(self, actions):
        self.step_index += 1
        self.episode_length += 1
        terminals = done[:, self.step_index].numpy()
        infos = []
        if terminals[0]:
          infos.append({'curriculum_success': 1.0,
                        'curriculum_template': 0,
                        'episode_length': self.episode_length})
          self.episode_length = 0
        return (observations[:, self.step_index].numpy(), np.zeros(1),
                terminals, np.zeros(1, dtype=bool), infos)

    policy = RecordingPolicy(_env(), hidden_size=16)
    policy.recorded_logits, policy.recorded_values = [], []
    expected_logits, expected_values = [], []
    with torch.no_grad():
      # PufferLib starts each rollout window from zero, regardless of which
      # episodes ended inside it. PPO replays these same windows from zero.
      for start in range(0, steps, horizon):
        stop = min(start + horizon, steps)
        logits, values = policy(observations[:, start:stop],
                                {'done': done[:, start:stop]})
        expected_logits.append(logits)
        expected_values.append(values.flatten())
    metrics = evaluate_promotion(policy, TrajectoryEnv(), 3, 17, 'cpu', horizon)
    assert policy.training  # Evaluation restores the caller's mode.
    assert metrics['promotion_episodes'] == 3
    assert metrics['promotion_recurrent_horizon'] == horizon
    assert torch.allclose(torch.cat(policy.recorded_logits),
                          torch.cat(expected_logits), atol=1e-6)
    assert torch.allclose(torch.cat(policy.recorded_values),
                          torch.cat(expected_values), atol=1e-6)


class _ScriptedEnv:
  """One agent row; every episode lasts `length` steps and ends as told."""

  def __init__(self, length, successes):
    self.length = length
    self.successes = list(successes)
    self.actions = []

  def reset(self, seed):
    self.step_index = 0
    self.episode = 0
    return np.random.RandomState(seed).randn(1, 115).astype(np.float32), []

  def step(self, actions):
    self.actions.append(int(np.asarray(actions).reshape(-1)[0]))
    self.step_index += 1
    infos = []
    terminal = self.step_index % self.length == 0
    if terminal:
      success = self.successes[self.episode % len(self.successes)]
      infos.append({'curriculum_success': float(success),
                    'curriculum_template': self.episode % 8,
                    'episode_length': self.length})
      self.episode += 1
    observation = np.random.RandomState(self.step_index).randn(
        1, 115).astype(np.float32)
    return (observation, np.zeros(1), np.array([terminal]),
            np.zeros(1, dtype=bool), infos)


def test_greedy_promotion_takes_the_argmax_action():
  torch.manual_seed(3)
  policy = FootballPolicy(_env(), hidden_size=16)
  # Make the actor decisive so argmax and sampling visibly differ.
  with torch.no_grad():
    policy.actor.weight.mul_(50)
  env = _ScriptedEnv(length=4, successes=[1])
  logits_seen = []
  original = policy.forward_eval

  def recording_forward_eval(observations, state):
    logits, values = original(observations, state)
    logits_seen.append(logits.clone())
    return logits, values

  policy.forward_eval = recording_forward_eval
  metrics = evaluate_promotion(policy, env, 2, 5, 'cpu', 4, greedy=True)
  assert metrics['promotion_greedy'] == 1.0
  assert metrics['promotion_aborted'] == 0.0
  expected = [int(logits.argmax(dim=-1)[0]) for logits in logits_seen]
  assert env.actions == expected


def test_promotion_early_abort_stops_hopeless_evaluations():
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16)
  hopeless = _ScriptedEnv(length=2, successes=[0])
  metrics = evaluate_promotion(
      policy, hopeless, 40, 1, 'cpu', 8, early_abort=(6, 0.4))
  assert metrics['promotion_aborted'] == 1.0
  assert metrics['promotion_episodes'] == 6
  assert metrics['promotion_success_rate'] == 0.0
  assert not promotion_passes(metrics, 0.6, 0.4)

  promising = _ScriptedEnv(length=2, successes=[1, 1, 0])
  metrics = evaluate_promotion(
      policy, promising, 12, 1, 'cpu', 8, early_abort=(6, 0.4))
  assert metrics['promotion_aborted'] == 0.0
  assert metrics['promotion_episodes'] == 12


def test_defaults_lower_entropy_and_spend_less_on_evaluation():
  args = build_parser().parse_args(['--device', 'cpu'])
  assert args.ent_coef == 0.001
  assert args.promotion_interval == 100
  assert args.promotion_episodes == 256
  assert args.frozen_defence_gate is True
  assert args.greedy_promotion_episodes > 0
  # Per 100 epochs the original schedule ran 4 x 256 held-out episodes and
  # the first frozen-gate schedule 512 + 128 + 128, which measured at half
  # the job.  The gate plus both diagnostics must stay well under either.
  assert (args.promotion_episodes + args.greedy_promotion_episodes +
          args.selfplay_promotion_episodes) <= 2 * 256


def test_every_level_is_score_gated_by_default():
  args = build_parser().parse_args(['--device', 'cpu'])
  assert args.scored_promotion_levels is None
  args = build_parser().parse_args(
      ['--device', 'cpu', '--scored-promotion-levels', '4'])
  assert args.scored_promotion_levels == 4


def test_gate_averages_the_two_weakest_templates():
  episodes = []
  for template in range(8):
    rate = 0.3 if template == 7 else 0.7
    episodes.extend({
        'curriculum_template': template,
        'curriculum_success': float(index < rate * 10),
    } for index in range(10))
  metrics = promotion_statistics(episodes)
  assert math.isclose(metrics['promotion_worst_template_success_rate'], 0.3)
  assert math.isclose(
      metrics['promotion_worst_two_template_success_rate'], 0.5)
  # One weak template no longer blocks promotion on its own ...
  assert promotion_passes(metrics, 0.6, 0.4)
  # ... but a genuine hole in two templates still does.
  episodes[-20:] = ({
      'curriculum_template': 6 + (index >= 10),
      'curriculum_success': float(index % 10 < 2),
  } for index in range(20))
  metrics = promotion_statistics(episodes)
  assert math.isclose(
      metrics['promotion_worst_two_template_success_rate'], 0.2)
  assert not promotion_passes(metrics, 0.6, 0.4)


def test_episode_end_clears_recurrent_memory():
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16).eval()
  observation = torch.randn(2, 115)

  with torch.no_grad():
    fresh = policy.forward_eval(
        observation, {'lstm_h': None, 'lstm_c': None, 'done': None})[0]
    state = {'lstm_h': None, 'lstm_c': None, 'done': None}
    policy.forward_eval(torch.randn(2, 115), state)
    policy.forward_eval(torch.randn(2, 115), state)
    carried = policy.forward_eval(observation, dict(state, done=None))[0]
    reset = policy.forward_eval(
        observation, dict(state, done=torch.ones(2)))[0]

  assert torch.allclose(reset, fresh, atol=1e-6)
  assert not torch.allclose(carried, fresh, atol=1e-4)


def test_policy_shapes_and_gradients_flow_to_every_head():
  policy = FootballPolicy(_env(), hidden_size=16)
  observations = torch.randn(4, 5, 115)
  logits, values = policy(observations, {'done': torch.zeros(4, 5)})
  assert logits.shape == (20, len(ACTION_NAMES))
  assert values.shape == (4, 5)
  (logits.square().mean() + values.square().mean()).backward()
  named = dict(policy.named_parameters())
  for name in ('actor.weight', 'critic.weight', 'cell.weight_ih',
               'encoder.0.weight'):
    assert named[name].grad is not None
    assert named[name].grad.abs().sum() > 0


def test_masked_rows_are_excluded_from_the_loss():
  """Zero-padded agent rows must not contribute a single loss term."""
  observations = torch.zeros(4, 5, 115)
  observations[1] = torch.randn(5, 115)
  observations[3, :2] = torch.randn(2, 115)
  active = observations.flatten(2).abs().sum(dim=-1) > 0
  valid = torch.ones(4, 5, dtype=torch.bool)
  valid[:, -1] = False
  trainable = active & valid

  assert trainable[0].sum() == 0
  assert trainable[1].tolist() == [True, True, True, True, False]
  assert trainable[3].tolist() == [True, True, False, False, False]
  assert trainable.any(dim=1).nonzero().flatten().tolist() == [1, 3]


def test_policy_diagnostics_distinguish_uniform_and_collapsed_policies():
  uniform = policy_diagnostics(torch.zeros(8, 19))
  collapsed = policy_diagnostics(torch.tensor([[20.0] + [0.0] * 18]))

  assert math.isclose(uniform['policy_entropy_fraction'].item(), 1.0,
                      rel_tol=1e-6)
  assert math.isclose(uniform['policy_max_probability'].item(), 1 / 19,
                      rel_tol=1e-6)
  assert collapsed['policy_entropy_fraction'].item() < 0.01
  assert collapsed['policy_max_probability'].item() > 0.99


def test_promotion_requires_overall_and_every_heldout_template():
  episodes = []
  for template in range(8):
    episodes.extend({
        'curriculum_template': template,
        'curriculum_success': float(success),
    } for success in ([1] * 7 + [0] * 3))
  metrics = promotion_statistics(episodes)
  assert promotion_passes(metrics, 0.6, 0.4)
  assert metrics['promotion_template_0_success_rate'] == 0.7

  episodes[-10:] = ({
      'curriculum_template': 7,
      'curriculum_success': 0.0,
  } for _ in range(10))
  metrics = promotion_statistics(episodes)
  assert not promotion_passes(metrics, 0.6, 0.4)


def test_config_satisfies_pufferlib_batching_constraints():
  args = build_parser().parse_args(['--device', 'cpu', '--no-async-collection'])
  num_agents = 660
  config = build_config(args, num_agents)
  horizon = config['bptt_horizon']
  segments = config['batch_size'] // horizon

  assert config['use_rnn'] is True
  assert config['batch_size'] % horizon == 0
  assert config['minibatch_size'] % horizon == 0
  assert config['minibatch_size'] <= config['batch_size']
  assert config['minibatch_size'] <= config['max_minibatch_size']
  # PufferLib requires at least one buffer row per agent.
  assert segments >= num_agents
  # One segment per agent per epoch keeps rollout and BPTT state aligned.
  assert segments == num_agents


def test_shaping_controls_share_discount_and_stay_out_of_promotion():
  """CLI scales stay opt-in and promotion construction receives neither."""
  from gfootball.examples.train_puffer import _make_promotion_env
  defaults = build_parser().parse_args([])
  assert defaults.ball_potential == defaults.player_potential == 0
  args = build_parser().parse_args([
      '--ball-potential', '1', '--player-potential', '0.3', '--gamma', '0.997'])
  assert args.player_potential == 0.3
  assert build_config(args, 660)['gamma'] == 0.997
  with patch('gfootball.examples.train_puffer.make_vector_env') as make:
    _make_promotion_env(args, SimpleNamespace(value=4))
  assert 'ball_potential_scale' not in make.call_args.kwargs
  assert 'player_potential_scale' not in make.call_args.kwargs


def test_update_epochs_cover_the_active_data_at_least_once():
  for num_segments in (1, 7, 30, 60, 660):
    for requested in (1, 16, 64):
      segments_per_minibatch = min(requested, num_segments)
      minibatches = max(1, math.ceil(
          4 * num_segments / segments_per_minibatch))
      sampled = minibatches * segments_per_minibatch
      assert sampled >= 4 * num_segments


class _AsyncScriptedPool:
  """Matches of one agent each, returned in whatever order they finish.

  Match m's episodes last `lengths[m]` steps and succeed iff `successes[m]`.
  recv() hands back the `batch` matches that are furthest behind in time,
  like a pool whose fast matches finish more steps.
  """

  def __init__(self, lengths, successes, batch=2):
    self.lengths, self.successes, self.batch = lengths, successes, batch
    self.num_agents = len(lengths)
    self.driver_env = SimpleNamespace(num_agents=1)

  def async_reset(self, seed):
    self.seed = seed
    self.clock = np.zeros(self.num_agents)
    self.steps = np.zeros(self.num_agents, dtype=int)
    self.pending = []

  def recv(self):
    # A short-episode match steps twice as fast in wall time.
    order = np.argsort(self.clock, kind='stable')[:self.batch]
    self.pending = order
    infos = []
    terminals = np.zeros(len(order), dtype=bool)
    for position, match in enumerate(order):
      if self.steps[match] and self.steps[match] % self.lengths[match] == 0:
        terminals[position] = True
        infos.append({'curriculum_success': float(self.successes[match]),
                      'curriculum_template': int(match) % 8,
                      'episode_length': int(self.lengths[match]),
                      'env_seed': float(self.seed + match)})
    observations = np.ones((len(order), 115), dtype=np.float32)
    return (observations, np.zeros(len(order)), terminals,
            np.zeros(len(order), dtype=bool), infos, order.copy(),
            np.ones(len(order), dtype=bool))

  def send(self, actions):
    for match in self.pending:
      self.steps[match] += 1
      self.clock[match] += self.lengths[match] / 10


def test_async_promotion_takes_a_fixed_quota_from_every_match():
  """Short (successful) episodes must not crowd out long (failed) ones."""
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16)
  # Half the matches score in 2 steps, half fail after 20.
  lengths = [2, 2, 20, 20]
  pool = _AsyncScriptedPool(lengths, successes=[1, 1, 0, 0])
  metrics = evaluate_promotion_async(policy, pool, 8, 3, 'cpu', 4)
  assert metrics['promotion_episodes'] == 8
  assert metrics['promotion_success_rate'] == 0.5


def test_async_promotion_does_not_depend_on_batching_or_timing():
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16)
  lengths = [3, 5, 7, 11, 13]
  for early_abort in (None, (10, 0.9)):
    results = []
    for batch in (1, 2, 4):
      pool = _AsyncScriptedPool(lengths, successes=[1, 0, 1, 0, 1],
                                batch=batch)
      results.append(evaluate_promotion_async(
          policy, pool, 20, 9, 'cpu', 4, early_abort=early_abort))
    assert results[0]['promotion_aborted'] == float(early_abort is not None)
    # Identical up to float summation order in the averaged diagnostics.
    for other in results[1:]:
      assert other.keys() == results[0].keys()
      for name, value in results[0].items():
        assert math.isclose(other[name], value, rel_tol=1e-6), name


def test_masked_losses_equal_the_indexed_losses():
  """The trainer's masked means are the same numbers as indexing first."""
  torch.manual_seed(0)
  mask = torch.rand(6, 5) < 0.6
  weight = mask.float()
  advantages = torch.randn(6, 5)
  ratio = torch.rand(6, 5) + 0.5

  def masked_mean(values):
    return (values * weight).sum() / weight.sum()

  indexed = normalize_advantages(advantages[mask])
  expected = torch.max(-indexed * ratio[mask],
                       -indexed * ratio[mask].clamp(0.8, 1.2)).mean()
  mean = masked_mean(advantages)
  std = masked_mean((advantages - mean).square()).sqrt()
  normalized = (advantages - mean) / std
  actual = masked_mean(torch.max(-normalized * ratio,
                                 -normalized * ratio.clamp(0.8, 1.2)))
  assert torch.isclose(actual, expected, atol=1e-6)


def test_async_collection_sizes_the_rollout_buffer():
  args = build_parser().parse_args(['--device', 'cpu'])
  assert args.async_collection and args.async_promotion
  assert args.overlap_collection
  config = build_config(args, 660)
  # Room for the epoch being trained, one segment in progress per match,
  # and segments completed while the update runs.
  assert config['batch_size'] // config['bptt_horizon'] == 3 * 660
  args = build_parser().parse_args(['--device', 'cpu',
                                    '--no-overlap-collection'])
  config = build_config(args, 660)
  assert config['batch_size'] // config['bptt_horizon'] == 2 * 660



def test_graphed_actor_matches_the_eager_forward():
  """Same value, memory and log-prob as forward_eval; skipped without CUDA."""
  if not torch.cuda.is_available():
    return
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=32).cuda()
  actor = GraphedActor(policy, 44, 115, 'cuda', None)
  for _ in range(2):
    observations = torch.randn(44, 115, device='cuda')
    hidden = torch.randn(44, 32, device='cuda')
    cell = torch.randn(44, 32, device='cuda')
    done = (torch.rand(44, device='cuda') < 0.3).float()
    action, logprob, value, new_hidden, new_cell = (
        t.clone() for t in actor(observations, hidden, cell, done))
    state = {'lstm_h': hidden, 'lstm_c': cell, 'done': done}
    with torch.no_grad():
      logits, expected_value = policy.forward_eval(observations, state)
    expected_logprob = torch.log_softmax(logits.float(), -1).gather(
        1, action.long().view(-1, 1)).squeeze(1)
    assert torch.allclose(value, expected_value, atol=1e-5)
    assert torch.allclose(new_hidden, state['lstm_h'], atol=1e-5)
    assert torch.allclose(new_cell, state['lstm_c'], atol=1e-5)
    assert torch.allclose(logprob, expected_logprob, atol=1e-5)
    # Parameters updated in place are picked up without recapturing.
    with torch.no_grad():
      for parameter in policy.parameters():
        parameter.mul_(1.1)


def test_graphed_update_matches_the_eager_minibatch_loop():
  """Same parameters and loss statistics after several epochs; needs CUDA."""
  if not torch.cuda.is_available():
    return
  import copy
  matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = False
  try:
    segments, horizon, device = 48, 8, 'cuda'
    torch.manual_seed(0)
    base = FootballPolicy(_env(), hidden_size=16).to(device)
    config = dict(clip_coef=0.2, vf_clip_coef=0.2, vf_coef=0.5,
                  ent_coef=0.001, max_grad_norm=0.5, device=device)
    buffers = dict(
        observations=torch.randn(segments, horizon, 115, device=device),
        terminals=(torch.rand(segments, horizon, device=device) < 0.1).float(),
        actions=torch.randint(0, 19, (segments, horizon), device=device),
        logprobs=-3 * torch.rand(segments, horizon, device=device))
    epoch = [torch.randn(segments, horizon, device=device) for _ in range(3)]
    trainable = torch.rand(segments, horizon, device=device) < 0.9
    segment_index = torch.arange(segments, device=device)
    results = []
    for graph in (False, True):
      trainer = FootballPuffeRL.__new__(FootballPuffeRL)
      trainer.config = config
      for name, value in buffers.items():
        setattr(trainer, name, value.clone())
      trainer.uncompiled_policy = trainer.policy = copy.deepcopy(base)
      trainer.optimizer = torch.optim.Adam(
          trainer.policy.parameters(), lr=3e-4, eps=1e-5, capturable=graph)
      trainer.optimizer_steps = 0
      trainer.graph_update = graph
      trainer._update_graph = trainer._update_graph_rows = None
      trainer._epoch_values = None
      for step in range(3):
        trainer._stage_epoch(*epoch, trainable)
        torch.manual_seed(100 + step)
        if graph:
          trainer._update_graphed(segment_index, 16, 6)
        else:
          orders = torch.argsort(torch.rand(6, segments, device=device), dim=1)
          for index in segment_index[orders[:, :16]]:
            trainer._minibatch_step(trainer.policy, index)
      results.append((torch.cat([p.detach().flatten()
                                 for p in trainer.policy.parameters()]),
                      trainer._totals.clone(), trainer.optimizer_steps))
    (eager, eager_totals, eager_steps), (graphed, graph_totals, graph_steps) = (
        results)
    assert eager_steps == graph_steps == 18
    assert torch.allclose(eager, graphed, atol=1e-5)
    assert torch.allclose(eager_totals, graph_totals, atol=1e-5)
  finally:
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32

if __name__ == '__main__':
  for name, test in sorted(dict(globals()).items()):
    if name.startswith('test_'):
      test()
      print('ok', name)
