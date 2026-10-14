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
    ACTION_NAMES, SHOT_ACTION, FootballPolicy, GraphedActor, PpoUpdate,
    SegmentCollector,
    Evaluator, build_config, build_parser, defence_memory_window,
    evaluate_episodes,
    explained_variance, generalized_advantages, normalize_advantages,
    policy_diagnostics)


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


def test_evaluation_matches_training_across_rollout_and_episode_boundaries():
  """Evaluation must replay the same recurrent windows as PufferLib rollout."""
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

    class TrajectoryPool:
      """One match replaying `observations`, episodes ending at `done`."""
      num_agents = 1
      driver_env = SimpleNamespace(num_agents=1)

      def async_reset(self, seed):
        self.seed, self.step_index, self.episode_length = seed, 0, 0

      def recv(self):
        terminals = done[:, self.step_index].numpy()
        infos = []
        if terminals[0]:
          infos.append({'success': 1.0, 'env_seed': float(self.seed),
                        'episode_length': self.episode_length})
          self.episode_length = 0
        return (observations[:, self.step_index].numpy(), np.zeros(1),
                terminals, np.zeros(1, dtype=bool), infos, np.zeros(1, int),
                np.ones(1, dtype=bool))

      def send(self, actions):
        self.step_index += 1
        self.episode_length += 1

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
    metrics = evaluate_episodes(policy, TrajectoryPool(), 3, 17, 'cpu',
                                horizon)
    assert policy.training  # Evaluation restores the caller's mode.
    assert metrics['episodes'] == 3
    assert metrics['recurrent_horizon'] == horizon
    # The pool also hands over the frame after the last episode end.
    assert len(policy.recorded_logits) == steps + 1
    assert torch.allclose(torch.cat(policy.recorded_logits)[:steps],
                          torch.cat(expected_logits), atol=1e-6)
    assert torch.allclose(torch.cat(policy.recorded_values)[:steps],
                          torch.cat(expected_values), atol=1e-6)


def test_defaults_lower_entropy_and_spend_less_on_evaluation():
  args = build_parser().parse_args(['--device', 'cpu'])
  assert args.ent_coef == 0.001
  assert args.evaluation_interval == 100
  # Per 100 epochs the level gate ran up to 384 held-out episodes, which
  # measured at nearly half the job; a check (one random-defender score per
  # difficulty) must stay well under that.
  difficulties = args.evaluation_difficulties.split(',')
  assert len(difficulties) * args.evaluation_episodes <= 256


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


def test_config_satisfies_pufferlib_batching_constraints():
  args = build_parser().parse_args(['--device', 'cpu'])
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
  # Whole segments per agent keep rollout and BPTT state aligned.
  assert segments % num_agents == 0


class _RecordingPools:
  """A pool factory for Evaluator that records how each pool was built."""

  def __init__(self):
    self.built = []

  def __call__(self, **kwargs):
    self.built.append(kwargs)
    return SimpleNamespace(kwargs=kwargs, close=lambda: None)


def test_shaping_controls_share_discount_and_stay_out_of_evaluation():
  """CLI scales stay opt-in and evaluation pools receive neither."""
  defaults = build_parser().parse_args([])
  assert defaults.ball_potential == defaults.player_potential == 0
  args = build_parser().parse_args([
      '--ball-potential', '1', '--player-potential', '0.3', '--gamma', '0.997'])
  assert args.player_potential == 0.3
  assert build_config(args, 660)['gamma'] == 0.997
  pools = _RecordingPools()
  with patch('gfootball.examples.train_puffer.evaluate_episodes'):
    Evaluator(args, make_pool=pools).score(None, 0.5)
  assert 'ball_potential_scale' not in pools.built[0]
  assert 'player_potential_scale' not in pools.built[0]


def test_loose_ball_shaping_is_opt_in_and_training_only():
  """The player-potential fade defaults off and changes shaping only, which
  evaluation never uses."""
  defaults = build_parser().parse_args([])
  assert defaults.player_potential_fade_speed == 0
  args = build_parser().parse_args([
      '--player-potential', '0.3', '--player-potential-fade-speed', '0.5'])
  assert args.player_potential_fade_speed == 0.5
  pools = _RecordingPools()
  with patch('gfootball.examples.train_puffer.evaluate_episodes'):
    Evaluator(args, make_pool=pools).score(None, 0.5)
  assert 'player_potential_fade_speed' not in pools.built[0]


def test_probe_results_move_the_shared_ranges():
  """Probe results in episode infos are recorded at their setting's edge
  (other episodes are ignored)."""
  from gfootball import adr
  from gfootball.examples.train_puffer import record_adr_probes
  defaults = build_parser().parse_args([])
  assert defaults.adr_probe_fraction == 0.2
  assert defaults.adr_buffer_size == 64
  assert (defaults.adr_low_threshold, defaults.adr_high_threshold) == (0.3, 0.6)
  randomizer = adr.AutomaticDomainRandomization(buffer_size=2)
  index = adr.PARAMETER_INDEX['fetch_distance']
  infos = [
      {'adr_probe': 1.0, 'adr_parameter': float(index),
       'adr_edge': float(adr.UPPER), 'success': 1.0},
      {'adr_probe': 0.0, 'adr_parameter': -1.0, 'adr_edge': -1.0,
       'success': 1.0},
      {'success': 0.0},
      {'adr_probe': 1.0, 'adr_parameter': float(index),
       'adr_edge': float(adr.UPPER), 'success': 1.0},
  ]
  assert record_adr_probes(randomizer, infos) == 2
  low, high = randomizer.ranges()[index]
  assert high > adr.PARAMETERS[index].start and low == adr.PARAMETERS[index].start


def test_evaluation_scores_on_fixed_adr_difficulty_scenes():
  """The evaluator scores on ADR scenes at fixed test difficulties (adr
  set_difficulty), never on the ranges training has reached and never with
  probes, one seed per difficulty, and keeps one pool per defence."""
  from gfootball import adr
  from gfootball.examples import train_puffer
  assert build_parser().parse_args([]).evaluation_difficulties == '0.25,0.5,1.0'
  seen = []

  def fake_evaluate(policy, pool, episodes, seed, device, recurrent_horizon):
    ranges = adr.AutomaticDomainRandomization(
        shared=pool.kwargs['adr_shared']).ranges()
    seen.append((pool.kwargs, seed, episodes))
    return {'success_rate': float(ranges[adr.PARAMETER_INDEX['blockers'], 1])}

  args = build_parser().parse_args(['--evaluation-difficulties', '0.5,1.0'])
  pools = _RecordingPools()
  evaluator = Evaluator(args, make_pool=pools)
  with patch.object(train_puffer, 'evaluate_episodes', fake_evaluate):
    for _ in range(2):
      metrics = evaluator.baseline(policy=None)
      evaluator.score(None, 0.5)
  blockers = adr.PARAMETERS[adr.PARAMETER_INDEX['blockers']]
  assert metrics == {
      'vs_random/difficulty_0.5/success_rate':
          blockers.start + 0.5 * (blockers.maximum - blockers.start),
      'vs_random/difficulty_1.0/success_rate': blockers.maximum}
  # One pool for the random defence and one for self-play, built once.
  assert [kwargs['frozen_defence_path'] for kwargs in pools.built] == [
      'random', None]
  for kwargs, seed, episodes in seen:
    assert kwargs['adr_probe_fraction'] == 0.0
    assert episodes == args.evaluation_episodes
  assert seen[0][1] != seen[1][1]
  # Same difficulty, same scenes, whatever the defence (the probe refresh
  # compares defences on them) and on every check.
  assert seen[0][1] == seen[2][1] == seen[3][1]


def test_adr_probe_defence_is_replaced_only_when_clearly_weaker_than_self_play():
  """The learner attacks the probe defence and its own defence on the same
  scenes; only when the probe defence concedes at least
  --adr-probe-refresh-margin more does the learner's current policy replace
  it on disk."""
  import tempfile
  from gfootball.env.puffer_policy import load_frozen_policy
  from gfootball.examples import train_puffer
  defaults = build_parser().parse_args([])
  assert defaults.adr_probe_refresh_interval == 1000
  assert defaults.adr_probe_refresh_margin == 0.1
  torch.manual_seed(0)
  old = FootballPolicy(_env(), hidden_size=16).eval()
  learner = FootballPolicy(_env(), hidden_size=16).eval()
  args = build_parser().parse_args([])
  observations = torch.randn(4, 115)
  for vs_probe, vs_self, replaced in ((0.50, 0.45, False), (0.60, 0.45, True)):
    with tempfile.TemporaryDirectory() as directory:
      path = directory + '/adr_probe_defence.pt'
      save_policy_snapshot(old, path)

      class FakeEvaluator:
        difficulties = [0.5, 1.0]

        def score(self, policy, difficulty, defence=None):
          return {'success_rate': vs_probe if defence == path else vs_self}

      metrics = train_puffer.refresh_adr_probe_defence(
          args, FakeEvaluator(), learner, path)
      expected = learner if replaced else old
      restored = load_frozen_policy(path, _env())
      with torch.no_grad():
        assert torch.allclose(restored.forward_eval(observations, {})[0],
                              expected.forward_eval(observations, {})[0])
      assert metrics['adr_probe/replaced'] == float(replaced)
      assert math.isclose(metrics['adr_probe/vs_probe'], vs_probe)
      assert math.isclose(metrics['adr_probe/vs_self'], vs_self)


def test_adr_probe_defence_starts_from_the_learner_or_a_file():
  """'self' starts the probe defence as the learner's own starting weights,
  so a run needs no probe file and the refresh grows the defence from
  there; a given file is copied instead, and the bot, random or idle write
  nothing (they are never refreshed)."""
  import os
  import tempfile
  from gfootball.env.puffer_policy import load_frozen_policy
  from gfootball.examples import train_puffer
  torch.manual_seed(0)
  learner = FootballPolicy(_env(), hidden_size=16).eval()
  other = FootballPolicy(_env(), hidden_size=16).eval()
  observations = torch.randn(4, 115)
  with tempfile.TemporaryDirectory() as directory:
    given = directory + '/given.pt'
    save_policy_snapshot(other, given)
    for spec, expected in (('self', learner), (given, other)):
      path = directory + '/adr_probe_defence.pt'
      args = build_parser().parse_args(['--adr-probe-defence', spec])
      train_puffer.start_adr_probe_defence(args, learner, path)
      restored = load_frozen_policy(path, _env())
      with torch.no_grad():
        assert torch.allclose(restored.forward_eval(observations, {})[0],
                              expected.forward_eval(observations, {})[0])
      os.remove(path)
    for spec in ('bot', 'random', 'idle'):
      args = build_parser().parse_args(['--adr-probe-defence', spec])
      train_puffer.start_adr_probe_defence(args, learner, path)
      assert not os.path.exists(path)


def test_bot_probes_and_the_bot_check_are_the_default():
  """The built-in bot defends the ADR probes by default; the bot check runs
  every 1000 epochs on 128 episodes, raising the bot after two checks in a
  row with bite below 0.2 (checks with fewer than 0.15 goals against idle
  defenders are skipped)."""
  defaults = build_parser().parse_args([])
  assert defaults.adr_probe_defence == 'bot'
  assert defaults.bot_check_interval == 1000
  assert defaults.bot_check_episodes == 128
  assert defaults.bot_saturation_bite == 0.2
  assert defaults.bot_toothless_checks == 2
  assert defaults.bot_minimum_idle_rate == 0.15


def test_bot_check_interval_is_checked_up_front():
  """The bot check runs on evaluation epochs, so its interval must be a
  multiple of --evaluation-interval; refused before training starts."""
  from gfootball import adr
  from gfootball.examples.train_puffer import BotCheck
  args = build_parser().parse_args(['--evaluation-interval', '100',
                                    '--bot-check-interval', '250'])
  try:
    BotCheck(args, evaluator=None, shared=adr.shared_bot_difficulty())
  except ValueError:
    pass
  else:
    raise AssertionError('interval 250 with evaluations every 100 accepted')
  args = build_parser().parse_args(['--evaluation-interval', '100'])
  check = BotCheck(args, evaluator=None, shared=adr.shared_bot_difficulty())
  assert [check.due(epoch) for epoch in (0, 100, 1000, 2000)] == [
      False, False, True, True]


def test_bot_check_raises_the_bot_once_it_stops_biting():
  """The check scores the attack against the bot and against idle defenders
  on the run's current ADR ranges, the same scenes for both; after two
  checks with bite below 0.2 the bot gets harder, and the probe tallies
  gathered against the old bot are dropped."""
  from gfootball import adr
  from gfootball.examples.train_puffer import BotCheck
  args = build_parser().parse_args([])
  seen = []
  rates = {'bot': 0.45, 'idle': 0.5}

  class FakeEvaluator:
    def score_ranges(self, policy, ranges, defence, episodes):
      seen.append((np.array(ranges), defence, episodes))
      return {'success_rate': rates[defence]}

  shared = adr.shared_bot_difficulty()
  check = BotCheck(args, FakeEvaluator(), shared)
  randomizer = adr.AutomaticDomainRandomization(buffer_size=4)
  randomizer.set_range('blockers', 1.0, 3.0)
  index = adr.PARAMETER_INDEX['blockers']
  for _ in range(3):
    randomizer.record(index, adr.UPPER, 1)
  first = check.run(None, randomizer)
  assert first['bot/raised'] == 0.0 and math.isclose(first['bot/bite'], 0.1)
  second = check.run(None, randomizer)
  assert second['bot/raised'] == 1.0
  assert shared.value == adr.BOT_LADDER[1]
  # The three results against the easy bot were dropped.
  assert not randomizer.record(index, adr.UPPER, 1)
  assert sorted(defence for _, defence, _ in seen) == [
      'bot', 'bot', 'idle', 'idle']
  for ranges, _, episodes in seen:
    np.testing.assert_array_equal(ranges, randomizer.ranges())
    assert episodes == args.bot_check_episodes


def test_evaluation_pools_attack_from_the_left_and_read_the_bot_difficulty():
  """Every evaluation pool fixes the left team as the attack, so each
  defence, the bot included (its engine only lets the left team attack),
  is scored on the same scenes; bot pools read the run's shared bot
  difficulty; score_ranges plays exactly the ranges it is given, with one
  seed whatever the defence."""
  from gfootball import adr
  from gfootball.examples import train_puffer
  args = build_parser().parse_args(['--evaluation-difficulties', '0.5'])
  bot = adr.shared_bot_difficulty()
  pools = _RecordingPools()
  evaluator = Evaluator(args, make_pool=pools, bot_difficulty=bot)
  seen = []

  def fake_evaluate(policy, pool, episodes, seed, device, recurrent_horizon):
    played = adr.AutomaticDomainRandomization(
        shared=pool.kwargs['adr_shared']).ranges()
    seen.append((pool.kwargs['frozen_defence_path'], played, seed, episodes))
    return {'success_rate': 0.5}

  wanted = adr.AutomaticDomainRandomization()
  wanted.set_range('blockers', 2.0, 4.0)
  with patch.object(train_puffer, 'evaluate_episodes', fake_evaluate):
    evaluator.score_ranges(None, wanted.ranges(), 'bot', 128)
    evaluator.score_ranges(None, wanted.ranges(), 'idle', 128)
    metrics = evaluator.versus(None, 'bot', 'vs_bot')
  assert metrics == {'vs_bot/difficulty_0.5/success_rate': 0.5}
  for kwargs in pools.built:
    assert kwargs['attacking_team'] == 'left'
    assert kwargs['bot_difficulty'] is bot
  (bot_name, bot_ranges, bot_seed, bot_episodes), (
      idle_name, idle_ranges, idle_seed, _) = seen[:2]
  assert (bot_name, idle_name) == ('bot', 'idle')
  np.testing.assert_array_equal(bot_ranges, wanted.ranges())
  np.testing.assert_array_equal(idle_ranges, wanted.ranges())
  assert bot_seed == idle_seed and bot_episodes == 128


def test_evaluation_needs_two_workers():
  """Evaluation pools are asynchronous, which needs at least two workers."""
  args = build_parser().parse_args(['--evaluation-workers', '1'])
  try:
    Evaluator(args, make_pool=_RecordingPools())
  except ValueError:
    pass
  else:
    raise AssertionError('one evaluation worker must be refused')


def test_evaluation_difficulties_are_checked_up_front():
  """A difficulty outside [0, 1] is refused when the Evaluator is built,
  before training starts, not at the first evaluation."""
  args = build_parser().parse_args(['--evaluation-difficulties', '0.5,1.5'])
  try:
    Evaluator(args, make_pool=_RecordingPools())
  except ValueError:
    pass
  else:
    raise AssertionError('difficulty 1.5 must be refused')


def test_snapshot_defences_replay_one_memory_window():
  """Training probes and evaluation pools replay a snapshot defence with
  the same window: --frozen-defence-horizon, else this run's own."""
  def window(flags):
    return defence_memory_window(build_parser().parse_args(flags))

  assert window([]) == 32
  assert window(['--carry-lstm-memory']) is None
  assert window(['--frozen-defence-horizon', '64']) == 64
  assert window(['--frozen-defence-horizon', '0']) is None


def test_evaluation_replays_memory_the_way_each_policy_was_trained():
  """Evaluation plays both sides with the memory they were trained with.

  The evaluated policy restarts memory every bptt window unless the run
  carries memory, and a frozen defence follows the run by default; a
  snapshot trained elsewhere (a probe defence from another run) can be
  given its own window, 0 meaning whole episodes.
  """
  cases = (([], 32, 32),
           (['--bptt-horizon', '128'], 128, 128),
           (['--carry-lstm-memory'], None, None),
           (['--carry-lstm-memory', '--frozen-defence-horizon', '32'],
            None, 32),
           (['--frozen-defence-horizon', '0'], 32, None))
  for flags, policy_window, defence_window in cases:
    args = build_parser().parse_args(['--device', 'cpu'] + flags)
    pools = _RecordingPools()
    with patch('gfootball.examples.train_puffer.evaluate_episodes') as evaluate:
      Evaluator(args, make_pool=pools).score(None, 0.5, 'random')
    assert pools.built[0]['frozen_defence_horizon'] == defence_window
    assert evaluate.call_args.kwargs['recurrent_horizon'] == policy_window


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


def test_gpu_filler_is_the_spin_kernel_only():
  """main() builds GpuFiller() with no options.

  Only the spin-kernel filler is offered: a filler that replays a CUDA graph
  advances the global CUDA generator and crashed alongside captured graphs.
  """
  from gfootball.examples.gpu_filler import GpuFiller
  GpuFiller()
  try:
    build_parser().parse_args(['--gpu-filler-kind', 'matmul'])
  except SystemExit:
    pass
  else:
    raise AssertionError('--gpu-filler-kind must not be accepted')


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
  config = dict(clip_coef=0.2, vf_clip_coef=0.2, vf_coef=0.5,
                ent_coef=0.001, max_grad_norm=0.5, device='cpu',
                update_epochs=3)
  rollout = SimpleNamespace(
      observations=torch.randn(segments, horizon, 115),
      terminals=(torch.rand(segments, horizon) < 0.1).float(),
      actions=torch.randint(0, 19, (segments, horizon)),
      logprobs=-3 * torch.rand(segments, horizon))
  policy = FootballPolicy(_env(), hidden_size=16)
  ppo = PpoUpdate(policy, torch.optim.Adam(policy.parameters(), lr=3e-4),
                  rollout, config)
  trainable = torch.rand(segments, horizon) < 0.7
  trainable[4] = False
  segment_index = trainable.any(dim=1).nonzero().flatten()

  totals = ppo.update(segment_index,
                      *(torch.randn(segments, horizon) for _ in range(3)),
                      trainable)

  transitions = totals[PpoUpdate.LOSS_NAMES.index('minibatch_transitions')]
  assert ppo.optimizer_steps == 3
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
        infos.append({'success': float(self.successes[match]),
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


def test_evaluation_takes_a_fixed_quota_from_every_match():
  """Short (successful) episodes must not crowd out long (failed) ones."""
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16)
  # Half the matches score in 2 steps, half fail after 20.
  lengths = [2, 2, 20, 20]
  pool = _AsyncScriptedPool(lengths, successes=[1, 1, 0, 0])
  metrics = evaluate_episodes(policy, pool, 8, 3, 'cpu', 4)
  assert metrics['episodes'] == 8
  assert metrics['success_rate'] == 0.5


def test_evaluation_does_not_depend_on_batching_or_timing():
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16)
  lengths = [3, 5, 7, 11, 13]
  results = []
  for batch in (1, 2, 4):
    pool = _AsyncScriptedPool(lengths, successes=[1, 0, 1, 0, 1], batch=batch)
    results.append(evaluate_episodes(policy, pool, 20, 9, 'cpu', 4))
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
    # With carried memory every rollout also brings new starting memory,
    # which the graph must read from the trainer's buffers in place.
    memories = [torch.randn(2, segments, 16, device=device)
                for _ in layouts]
    finals = {}
    for carry_memory in (False, True):
      results = []
      for graph in (False, True):
        rollout = SimpleNamespace(
            observations=observations.clone(), actions=actions.clone(),
            logprobs=logprobs.clone(),
            terminals=torch.zeros(segments, horizon, device=device))
        policy = copy.deepcopy(base)
        # The real collector's starting memory: one agent per match, so its
        # blocks are the rollout's rows (a third of them in play keeps the
        # buffer at least the two epochs it needs).
        collector = SegmentCollector(
            SimpleNamespace(num_agents=segments // 3,
                            driver_env=SimpleNamespace(num_agents=1)),
            policy, rollout, horizon, device, carry_memory=carry_memory)
        rate = (torch.tensor(schedule[0], device=device) if graph
                else schedule[0])
        optimizer = torch.optim.Adam(policy.parameters(), lr=rate, eps=1e-5,
                                     capturable=graph)
        ppo = PpoUpdate(policy, optimizer, rollout, config, graph=graph,
                        starting_memory=collector.starting_memory)
        totals = []
        for terminals, memory, learning_rate in zip(layouts, memories,
                                                    schedule):
          rollout.terminals.copy_(terminals)
          if carry_memory:
            collector.segment_h.copy_(memory[0])
            collector.segment_c.copy_(memory[1])
          set_learning_rate(optimizer, learning_rate)
          before = torch.cat([p.detach().flatten().clone()
                              for p in policy.parameters()])
          totals.append(ppo.update(segment_index, *epoch, trainable))
        final = torch.cat([p.detach().flatten()
                           for p in policy.parameters()])
        results.append((final, before, torch.stack(totals),
                        ppo.optimizer_steps))
        captures = ppo.captures
      (eager, _, eager_totals, eager_steps), (
          graphed, graphed_before, graph_totals, graph_steps) = results
      assert eager_steps == graph_steps == 9
      # The second layout fits the first's capacity; the third needs more.
      assert captures == 2
      assert torch.allclose(eager, graphed, atol=1e-5), carry_memory
      assert torch.allclose(eager_totals, graph_totals, atol=1e-5)
      assert torch.equal(graphed, graphed_before)
      finals[carry_memory] = graphed
    # The starting memory really reaches the update.
    assert not torch.allclose(finals[False], finals[True], atol=1e-5)
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

  Every checkpoint and policy snapshot before the packed cuDNN forward
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


def test_bptt_forward_continues_each_window_from_its_starting_memory():
  """A window can start from the memory its segment was collected with.

  With carried memory the collector does not clear the LSTM at segment
  starts, so training must begin each window's first episode piece from the
  state the actor had there (state['lstm_h'] / ['lstm_c'], one row per
  window); pieces that start at an episode end still start from zero.  This
  pins the window forward to stepping forward_eval from the same memory,
  for the exact and a padded layout, for outputs and every gradient.
  """
  from gfootball.env.puffer_policy import (
      episode_batch_sizes, episode_pieces)
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16).eval()
  segments, horizon = 6, 7
  observations = torch.randn(segments, horizon, 115)
  done = torch.zeros(segments, horizon)
  done[0, 3] = 1
  done[2, 1] = done[2, 5] = 1
  done[3, 0] = 1
  done[4, 2] = done[4, 3] = 1
  done[4, horizon - 1] = 1
  done[5] = 1
  hidden = torch.randn(segments, 16)
  cell = torch.randn(segments, 16)

  def outputs_and_gradients(logits, values):
    policy.zero_grad()
    (logits.square().mean() + values.square().mean()).backward()
    return [logits.detach(), values.detach()] + [
        parameter.grad.clone() for parameter in policy.parameters()]

  state = {'lstm_h': hidden, 'lstm_c': cell, 'done': None}
  stepwise_logits, stepwise_values = [], []
  for step in range(horizon):
    state['done'] = done[:, step]
    step_logits, step_values = policy.forward_eval(
        observations[:, step], state)
    stepwise_logits.append(step_logits)
    stepwise_values.append(step_values)
  expected = outputs_and_gradients(
      torch.stack(stepwise_logits, dim=1).reshape(segments * horizon, -1),
      torch.stack(stepwise_values, dim=1))

  for pieces in (episode_pieces(done),
                 episode_pieces(done, episode_batch_sizes(done) + 3)):
    actual = outputs_and_gradients(*policy(observations, {
        'episode_pieces': pieces, 'done': done,
        'lstm_h': hidden, 'lstm_c': cell}))
    for wanted, got in zip(expected, actual):
      assert torch.allclose(wanted, got, atol=1e-5)


class _OneMatchPool:
  """One match of two agent rows whose episodes end where `done` says."""

  def __init__(self, observations, done):
    self.observations = observations
    self.done = done
    self.step = 0
    self.num_agents = 2
    self.driver_env = SimpleNamespace(num_agents=2)

  def recv(self):
    terminal = bool(self.done[self.step])
    return (self.observations[self.step].numpy(), np.zeros(2, np.float32),
            np.full(2, terminal), np.zeros(2, dtype=bool), [], np.arange(2),
            np.ones(2, dtype=bool))

  def send(self, actions):
    self.step += 1


def _collector_and_update(pool, policy, horizon, carry_memory):
  """A collector over `pool` with room for three segments, and its update."""
  segments = 6
  rollout = SimpleNamespace(
      observations=torch.zeros(segments, horizon, 115),
      actions=torch.zeros(segments, horizon, dtype=torch.long),
      **{name: torch.zeros(segments, horizon)
         for name in ('logprobs', 'rewards', 'terminals', 'values')})
  collector = SegmentCollector(pool, policy, rollout, horizon, 'cpu',
                               carry_memory=carry_memory)
  config = dict(clip_coef=0.2, vf_clip_coef=0.2, vf_coef=0.5, ent_coef=0.0,
                max_grad_norm=0.5, update_epochs=1)
  ppo = PpoUpdate(policy, torch.optim.Adam(policy.parameters(), lr=0.0),
                  rollout, config, starting_memory=collector.starting_memory)
  return collector, ppo, rollout


def test_collector_memory_is_what_training_replays():
  """The actor's memory at every step is the memory PPO trains through.

  Without carried memory the collector clears the LSTM at every segment
  start; with it, memory runs on across segments and clears only at episode
  ends.  Either way the stored log-probs must follow the actor's memory, and
  the update must reproduce them exactly (importance ratio 1 before any
  step), including for a segment that starts mid-episode.
  """
  horizon = 4
  steps = 2 * horizon
  torch.manual_seed(0)
  observations = torch.randn(steps + 1, 2, 115)
  done = torch.zeros(steps + 1)
  done[5] = 1
  for carry_memory in (False, True):
    torch.manual_seed(1)
    policy = FootballPolicy(_env(), hidden_size=16)
    collector, ppo, rollout = _collector_and_update(
        _OneMatchPool(observations, done), policy, horizon, carry_memory)
    for _ in range(steps):
      collector._collect_batch()

    state = {'lstm_h': None, 'lstm_c': None, 'done': None}
    blocks = list(collector.complete_blocks)
    assert len(blocks) == 2
    for step in range(steps):
      if step % horizon == 0 and not carry_memory:
        state['lstm_h'] = state['lstm_c'] = None
      state['done'] = torch.full((2,), float(done[step]))
      with torch.no_grad():
        logits, _ = policy.forward_eval(observations[step], state)
      rows = blocks[step // horizon] * 2 + torch.arange(2)
      column = step % horizon
      expected = torch.log_softmax(logits, dim=-1).gather(
          1, rollout.actions[rows, column].view(-1, 1)).squeeze(1)
      assert torch.allclose(rollout.logprobs[rows, column], expected,
                            atol=1e-5), (carry_memory, step)

    # The epoch trains the newest segment, which starts mid-episode.
    rows, training_rows = collector.claim_training_rows()
    trainable = training_rows[:, None].expand(-1, horizon).clone()
    totals = ppo.update(rows, rollout.values, torch.randn(6, horizon),
                        torch.randn(6, horizon), trainable)
    totals = dict(zip(PpoUpdate.LOSS_NAMES, totals.tolist()))
    assert totals['approx_kl'] < 1e-10, (carry_memory, totals)
    assert math.isclose(totals['importance'], 1.0, abs_tol=1e-6)


def test_evaluation_can_carry_memory_through_whole_episodes():
  """recurrent_horizon=None clears memory only at episode ends.

  A policy trained with carried memory is evaluated the way it acts, so the
  evaluator must match a window longer than the whole evaluation.
  """
  torch.manual_seed(0)
  policy = FootballPolicy(_env(), hidden_size=16)
  carried, windowed = [evaluate_episodes(
      policy, _AsyncScriptedPool([3, 5, 7], successes=[1, 0, 1]),
      12, 9, 'cpu', window) for window in (None, 1000)]
  assert carried['recurrent_horizon'] == 0.0
  for name, value in windowed.items():
    if name != 'recurrent_horizon':
      assert math.isclose(carried[name], value, rel_tol=1e-6), name


if __name__ == '__main__':
  # At the end of the module so every test above is defined when it runs.
  for name, test in sorted(dict(globals()).items()):
    if name.startswith('test_'):
      test()
      print('ok', name)
