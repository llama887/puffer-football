"""Correctness checks for the recurrent football PPO loop."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium
import numpy as np
import torch

from gfootball.env.puffer_policy import (
    hidden_size_from_state_dict, save_policy_snapshot, upgrade_state_dict)
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
  """Training and rollout must see the same recurrent state and gradients.

  The window forward splits segments at episode ends; this pins it to the
  one-step acting forward for resets at the first step, back to back, at the
  last step and on every step, for outputs and for every parameter gradient.
  """
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16).eval()
  segments, horizon = 6, 7
  observations = torch.randn(segments, horizon, 115)
  done = torch.zeros(segments, horizon)
  done[0, 3] = 1
  done[2, 1] = 1
  done[2, 5] = 1
  done[3, 0] = 1
  done[4, 2] = done[4, 3] = 1
  done[4, horizon - 1] = 1
  done[5] = 1

  def gradients_of(logits, values):
    policy.zero_grad()
    (logits.square().mean() + values.square().mean()).backward()
    return {name: parameter.grad.clone()
            for name, parameter in policy.named_parameters()}

  logits, values = policy(observations, {'done': done})
  gradients = gradients_of(logits, values)
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
  stepwise_gradients = gradients_of(stepwise_logits, stepwise_values)

  assert torch.allclose(logits, stepwise_logits, atol=1e-5)
  assert torch.allclose(values, stepwise_values, atol=1e-5)
  for name, gradient in gradients.items():
    assert torch.allclose(gradient, stepwise_gradients[name], atol=1e-6), name


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
  for name in ('actor.weight', 'critic.weight', 'rnn.weight_ih_l0',
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


def test_spawn_mode_defaults_to_curriculum_and_reaches_the_gate():
  """The gate must score on the same spawn distribution the policy trains on."""
  from gfootball.examples.train_puffer import _make_promotion_env
  assert build_parser().parse_args([]).spawn == 'curriculum'
  args = build_parser().parse_args(['--spawn', 'uniform'])
  with patch('gfootball.examples.train_puffer.make_vector_env') as make:
    _make_promotion_env(args, SimpleNamespace(value=0))
  assert make.call_args.kwargs['spawn'] == 'uniform'


def test_learning_rate_follows_training_progress():
  """Cosine decay by agent steps done, reaching zero exactly at the end.

  PufferLib sized its cosine schedule in epochs as total_timesteps //
  batch_size, but batch_size is buffer room (2x for async collection, 3x
  with overlap) while epochs consume one segment per agent, and overlap
  drops surplus segments, so no epoch count is right.  A 1B-step run hit
  zero at ~500M and, the cosine being periodic, climbed back to the full
  rate by 1B.  Tying the rate to steps done fixes it for any batching.
  """
  from gfootball.examples.train_puffer import cosine_learning_rate
  total = 1_000_000
  rates = [cosine_learning_rate(1e-3, step, total)
           for step in range(0, 1_300_001, 1_000)]
  assert rates[0] == 1e-3
  assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))
  assert abs(cosine_learning_rate(1e-3, total // 2, total) - 5e-4) < 1e-12
  assert cosine_learning_rate(1e-3, total, total) == 0.0
  assert cosine_learning_rate(1e-3, 3 * total, total) == 0.0


def test_gpu_filler_accepts_the_trainer_call_and_only_spins():
  """main() builds GpuFiller(matrix_size=..., kind=args.gpu_filler_kind).

  Only the spin-kernel filler is offered: a filler that replays a CUDA graph
  advances the global CUDA generator and crashed alongside captured graphs.
  """
  from gfootball.examples.gpu_filler import GpuFiller
  args = build_parser().parse_args([])
  assert args.gpu_filler_kind == 'sleep'
  GpuFiller(matrix_size=args.gpu_filler_matrix_size, kind=args.gpu_filler_kind)
  try:
    GpuFiller(kind='matmul')
  except ValueError:
    pass
  else:
    raise AssertionError('a graph-replaying filler must be rejected')
  try:
    build_parser().parse_args(['--gpu-filler-kind', 'matmul'])
  except SystemExit:
    pass
  else:
    raise AssertionError('--gpu-filler-kind matmul must not be accepted')


def test_gate_score_is_logged_per_level_against_the_level_entry_policy():
  """Each gate result is also logged under its own level's name.

  The gate opponent is the policy as it was on entering the level, fixed for
  as long as the level trains, so each level's line should rise while that
  level is learning and go flat when it stalls.
  """
  from gfootball.examples.train_puffer import FootballPuffeRL
  trainer = object.__new__(FootballPuffeRL)
  trainer.record_promotion(5, {
      'promotion_success_rate': 0.4,
      'promotion_worst_two_template_success_rate': 0.2,
      'promotion_episodes': 256.0,
      'promotion_mean_episode_length': 90.0}, advanced=False)
  logged = trainer.promotion_metrics
  assert logged['promotion_success_rate'] == 0.4
  assert logged['level_entry/success_rate'] == 0.4
  assert logged['level_entry/level_5/success_rate'] == 0.4
  assert logged['level_entry/level_5/worst_two_template_success_rate'] == 0.2
  assert logged['level_entry/level_5/episodes'] == 256.0
  assert not any(key.startswith('level_entry/level_4') for key in logged)
  assert not hasattr(build_parser().parse_args([]), 'initial_opponent_interval')


def test_padded_episode_layout_gives_the_same_window_forward():
  """Extra packed slots per step change nothing but the capacity.

  A captured update keeps one packed layout for many rollouts by giving
  every step more slots than it needs; the spare slots read a zero row and
  their outputs are dropped, so outputs and gradients must not move.
  """
  from gfootball.env.puffer_policy import (
      episode_batch_sizes, episode_pieces)
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16)
  observations = torch.randn(5, 7, 115)
  done = torch.zeros(5, 7)
  done[0, 3] = done[2, 1] = done[2, 5] = done[3, 0] = 1
  done[4, 2] = done[4, 3] = 1

  def run(state):
    policy.zero_grad()
    logits, values = policy(observations, state)
    (logits.square().mean() + values.square().mean()).backward()
    return [logits.detach(), values.detach()] + [
        parameter.grad.clone() for parameter in policy.parameters()]

  needed = episode_batch_sizes(done)
  assert needed.tolist() == [10, 8, 6, 5, 2, 2, 2]
  exact = run({'done': done})
  padded = run({'episode_pieces': episode_pieces(done, needed + 3)})
  for expected, actual in zip(exact, padded):
    assert torch.allclose(expected, actual, atol=1e-6)


def test_every_update_epoch_is_one_step_over_the_whole_rollout():
  """A gradient step trains on every active transition of the rollout.

  The minibatch is the whole rollout (it fits on the GPU many times over),
  so update_epochs is exactly the number of optimizer steps per rollout.
  """
  torch.manual_seed(0)
  segments, horizon = 12, 8
  trainer = FootballPuffeRL.__new__(FootballPuffeRL)
  trainer.config = dict(clip_coef=0.2, vf_clip_coef=0.2, vf_coef=0.5,
                        ent_coef=0.001, max_grad_norm=0.5, device='cpu',
                        update_epochs=3)
  trainer.observations = torch.randn(segments, horizon, 115)
  trainer.terminals = (torch.rand(segments, horizon) < 0.1).float()
  trainer.actions = torch.randint(0, 19, (segments, horizon))
  trainer.logprobs = -3 * torch.rand(segments, horizon)
  trainer.uncompiled_policy = trainer.policy = FootballPolicy(
      _env(), hidden_size=16)
  trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=3e-4)
  trainer.optimizer_steps = 0
  trainer.graph_update = False
  trainer._epoch_values = None
  trainable = torch.rand(segments, horizon) < 0.7
  trainable[4] = False
  segment_index = trainable.any(dim=1).nonzero().flatten()
  trainer._stage_epoch(*(torch.randn(segments, horizon) for _ in range(3)),
                       trainable)

  trainer._update(segment_index)

  transitions = trainer._totals[
      FootballPuffeRL.LOSS_NAMES.index('minibatch_transitions')]
  assert trainer.optimizer_steps == 3
  assert transitions == 3 * trainable.sum()
  assert '--minibatch-segments' not in build_parser().format_help()


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
  from gfootball.env import entity_observation as entity
  for network, size in (('mlp', 115), ('transformer', entity.SIZE)):
    _check_graphed_actor(network, size)


def _check_graphed_actor(network, size):
  torch.manual_seed(0)
  policy = FootballPolicy(_env(size), hidden_size=32, network=network).cuda()
  actor = GraphedActor(policy, 44, size, 'cuda', None)
  for _ in range(2):
    observations = (_entity_rows(44, absent_opponents=2).cuda()
                    if network == 'transformer' else
                    torch.randn(44, size, device='cuda'))
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


def test_graphed_update_matches_the_eager_update():
  """One graph per rollout, replayed update_epochs times; needs CUDA.

  Every rollout has its own episode layout, so the graph is captured again
  for each one; the learning rate is annealed between rollouts through a
  tensor the graph reads.  Parameters, loss totals and step counts must
  equal the eager update, and a zero rate must leave the weights alone.
  """
  if not torch.cuda.is_available():
    return
  import copy
  from gfootball.examples.train_puffer import set_learning_rate
  matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
  cudnn_tf32 = torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  try:
    segments, horizon, device = 48, 8, 'cuda'
    torch.manual_seed(0)
    base = FootballPolicy(_env(), hidden_size=16).to(device)
    config = dict(clip_coef=0.2, vf_clip_coef=0.2, vf_coef=0.5,
                  ent_coef=0.001, max_grad_norm=0.5, device=device,
                  update_epochs=3)
    observations = torch.randn(segments, horizon, 115, device=device)
    actions = torch.randint(0, 19, (segments, horizon), device=device)
    logprobs = -3 * torch.rand(segments, horizon, device=device)
    epoch = [torch.randn(segments, horizon, device=device) for _ in range(3)]
    trainable = torch.rand(segments, horizon, device=device) < 0.9
    segment_index = torch.arange(segments, device=device)
    layouts = [(torch.rand(segments, horizon, device=device) < rate).float()
               for rate in (0.1, 0.1, 0.3)]
    schedule = (3e-4, 1.5e-4, 0.0)
    results = []
    for graph in (False, True):
      trainer = FootballPuffeRL.__new__(FootballPuffeRL)
      trainer.config = config
      trainer.observations = observations.clone()
      trainer.actions = actions.clone()
      trainer.logprobs = logprobs.clone()
      trainer.uncompiled_policy = trainer.policy = copy.deepcopy(base)
      rate = torch.tensor(schedule[0], device=device) if graph else schedule[0]
      trainer.optimizer = torch.optim.Adam(
          trainer.policy.parameters(), lr=rate, eps=1e-5, capturable=graph)
      trainer.optimizer_steps = 0
      trainer.graph_update = graph
      trainer._update_graph = trainer._update_pool = None
      trainer._update_capacity = None
      trainer.update_captures = 0
      trainer._epoch_values = None
      totals = []
      for terminals, learning_rate in zip(layouts, schedule):
        trainer.terminals = terminals
        set_learning_rate(trainer.optimizer, learning_rate)
        trainer._stage_epoch(*epoch, trainable)
        before = torch.cat([p.detach().flatten().clone()
                            for p in trainer.policy.parameters()])
        trainer._update(segment_index)
        totals.append(trainer._totals.clone())
      final = torch.cat([p.detach().flatten()
                         for p in trainer.policy.parameters()])
      results.append((final, before, torch.stack(totals),
                      trainer.optimizer_steps))
      captures = trainer.update_captures
    (eager, _, eager_totals, eager_steps), (graphed, graphed_before,
                                           graph_totals, graph_steps) = results
    assert eager_steps == graph_steps == 9
    # The second layout fits the first's capacity; the third needs more.
    assert captures == 2
    assert torch.allclose(eager, graphed, atol=1e-5)
    assert torch.allclose(eager_totals, graph_totals, atol=1e-5)
    assert torch.equal(graphed, graphed_before)
  finally:
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _entity_rows(rows, absent_opponents=0):
  """Random entity rows with the flags a real row has."""
  from gfootball.env import entity_observation as entity
  observations = torch.randn(rows, entity.SIZE)
  players = observations[:, entity.PLAYERS_START:].view(
      rows, entity.PLAYERS, entity.PLAYER_FEATURES)
  present = entity.PLAYER_FEATURE_INDEX['present']
  players[..., present] = 1
  if absent_opponents:
    players[:, -absent_opponents:] = 0
  return observations


def test_transformer_policy_matches_stepwise_rollout():
  """The attention encoder keeps training and acting on the same numbers."""
  from gfootball.env import entity_observation as entity
  torch.manual_seed(0)
  policy = FootballPolicy(_env(entity.SIZE), hidden_size=16,
                          network='transformer').eval()
  segments, horizon = 3, 5
  observations = _entity_rows(segments * horizon, absent_opponents=2).view(
      segments, horizon, -1)
  done = torch.zeros(segments, horizon)
  done[1, 2] = 1
  with torch.no_grad():
    logits, values = policy(observations, {'done': done})
    state = {'lstm_h': None, 'lstm_c': None, 'done': None}
    stepwise = []
    for step in range(horizon):
      state['done'] = done[:, step]
      stepwise.append(policy.forward_eval(observations[:, step], state)[0])
  stepwise = torch.stack(stepwise, dim=1).reshape(segments * horizon, -1)
  assert torch.allclose(logits, stepwise, atol=1e-5)
  assert values.shape == (segments, horizon)


def test_transformer_ignores_absent_players_and_their_order():
  """Absent players change nothing; nor does the order of the others."""
  from gfootball.env import entity_observation as entity
  torch.manual_seed(0)
  policy = FootballPolicy(_env(entity.SIZE), hidden_size=16,
                          network='transformer').eval()
  observations = _entity_rows(4, absent_opponents=3)
  players = slice(entity.PLAYERS_START, None)
  with torch.no_grad():
    reference = policy.forward_eval(observations, {})[0]
    garbage = observations.clone()
    view = garbage[:, players].view(4, entity.PLAYERS, entity.PLAYER_FEATURES)
    view[:, -3:, :entity.PLAYER_FEATURE_INDEX['present']] = torch.randn(
        4, 3, entity.PLAYER_FEATURE_INDEX['present'])
    shuffled = observations.clone()
    view = shuffled[:, players].view(4, entity.PLAYERS, entity.PLAYER_FEATURES)
    order = torch.cat([torch.tensor([0]), 1 + torch.randperm(entity.PLAYERS - 1)])
    view.copy_(view[:, order].clone())
    # The normalizer is per feature slot, so it must be the identity here.
    assert policy.normalizer.count < 1
    assert torch.allclose(policy.forward_eval(garbage, {})[0], reference,
                          atol=1e-5)
    assert torch.allclose(policy.forward_eval(shuffled, {})[0], reference,
                          atol=1e-5)


def test_frozen_snapshots_reload_as_the_network_they_were_saved_from():
  import tempfile
  from gfootball.env import entity_observation as entity
  from gfootball.env.puffer_policy import load_frozen_policy
  torch.manual_seed(0)
  for network, size in (('mlp', 115), ('transformer', entity.SIZE)):
    policy = FootballPolicy(_env(size), hidden_size=16, network=network).eval()
    with tempfile.TemporaryDirectory() as directory:
      path = directory + '/policy.pt'
      save_policy_snapshot(policy, path)
      restored = load_frozen_policy(path, _env(size))
    observations = (_entity_rows(3) if network == 'transformer'
                    else torch.randn(3, size))
    with torch.no_grad():
      assert torch.allclose(restored.forward_eval(observations, {})[0],
                            policy.forward_eval(observations, {})[0])


def test_transformer_network_needs_entity_observations():
  args = build_parser().parse_args(
      ['--network', 'transformer', '--observation', 'entities'])
  assert args.network == 'transformer' and args.observation == 'entities'
  assert build_parser().parse_args([]).network == 'mlp'
  try:
    FootballPolicy(_env(115), hidden_size=16, network='transformer')
  except ValueError:
    pass
  else:
    raise AssertionError('a transformer over simple115 must be rejected')


def test_lstm_cell_checkpoints_still_load():
  """Runs from the LSTMCell era wrote cell.* keys; they must still load.

  Every checkpoint and level-entry snapshot before the packed cuDNN forward
  holds the recurrence as LSTMCell parameters, the same tensors nn.LSTM
  names rnn.*_l0.
  """
  policy = FootballPolicy(_env(), hidden_size=16)
  legacy = {}
  for key, value in policy.state_dict().items():
    if key.startswith('rnn.'):
      key = key.replace('rnn.', 'cell.').removesuffix('_l0')
    legacy[key] = value
  assert 'cell.weight_hh' in legacy and 'rnn.weight_hh_l0' not in legacy
  assert hidden_size_from_state_dict(legacy) == 16
  for checkpoint in (legacy, policy.state_dict()):
    restored = FootballPolicy(_env(), hidden_size=16)
    restored.load_state_dict(upgrade_state_dict(checkpoint))
    for key, value in policy.state_dict().items():
      assert torch.equal(restored.state_dict()[key], value), key


if __name__ == '__main__':
  # At the end of the module so every test above is defined when it runs.
  for name, test in sorted(dict(globals()).items()):
    if name.startswith('test_'):
      test()
      print('ok', name)
