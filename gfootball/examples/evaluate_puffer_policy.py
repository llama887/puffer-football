"""Measure what a trained shared football policy actually does."""

import argparse
from collections import Counter
import json
import math

import numpy as np
import torch

from gfootball.env import football_action_set
from gfootball.env.puffer_env import FootballPufferEnv
from gfootball.examples.train_puffer import FootballPolicy


ACTION_NAMES = tuple(
    str(action) for action in football_action_set.action_set_dict['default'])


def _observation_contract(observations, active, sorted_players=True):
  active_observations = observations[active]
  frames = active_observations.reshape(-1, 4, 115)
  own_positions = frames[:, :, :22].reshape(-1, 4, 11, 2)
  if sorted_players:
    # Training sorts teammates by distance, so the controlled player is
    # always slot 0 rather than the slot named by the active one-hot.
    ego_positions = own_positions[:, :, 0]
  else:
    active_players = frames[:, :, 97:108].argmax(axis=-1)
    rows, history = np.indices(active_players.shape)
    ego_positions = own_positions[rows, history, active_players]
  return (float(np.abs(active_observations).max()),
          float(np.abs(ego_positions).max()),
          int(np.abs(active_observations).argmax() % observations.shape[1]))


def evaluate(checkpoint, episodes, greedy, seed, attacker_only_levels,
             curriculum_level):
  torch.manual_seed(seed)
  env = FootballPufferEnv(
      seed=seed, frame_stack=4, curriculum_window=episodes + 1,
      attacker_only_levels=attacker_only_levels)
  env._curriculum_level = curriculum_level

  action_counts = Counter()
  totals = Counter()
  episode_rows = []
  try:
    policy = FootballPolicy(env)
    policy.load_state_dict(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    policy.eval()
    observations, _ = env.reset()
    for episode in range(episodes):
      attack_sign = 1.0 if env._attacking_left else -1.0
      attacking_slice = slice(0, 11) if attack_sign > 0 else slice(11, 22)
      episode_active = env._active_mask.copy()
      raw_env = env._env.unwrapped._env
      ball_start = float(raw_env.observation()['ball'][0])
      previous_actions = None
      max_ball_progress = 0.0
      first_shot_step = None
      done = False
      step = 0
      while not done:
        max_observation, max_ego_position, max_observation_index = (
            _observation_contract(
                observations, episode_active)
        )
        if max_observation > totals['max_abs_observation']:
          totals['max_abs_observation_index'] = max_observation_index
        totals['max_abs_observation'] = max(
            totals['max_abs_observation'], max_observation)
        totals['max_abs_ego_position'] = max(
            totals['max_abs_ego_position'], max_ego_position)
        ball_x = float(raw_env.observation()['ball'][0])
        max_ball_progress = max(
            max_ball_progress, attack_sign * (ball_x - ball_start))

        with torch.inference_mode():
          logits, _ = policy(torch.as_tensor(observations))
          probabilities = torch.softmax(logits.float(), dim=-1)
          actions = (probabilities.argmax(-1) if greedy else
                     torch.multinomial(probabilities, 1).squeeze(-1))
        action_array = actions.numpy()
        active_actions = action_array[episode_active]
        attack_actions = action_array[attacking_slice][
            episode_active[attacking_slice]]
        active_probabilities = probabilities[episode_active]
        action_counts.update(active_actions.tolist())
        totals['decisions'] += len(active_actions)
        totals['environment_steps'] += 1
        totals['attacking_decisions'] += len(attack_actions)
        totals['attacking_shots'] += int(np.sum(attack_actions == 12))
        totals['attacking_kicks'] += int(np.sum(
            (attack_actions >= 9) & (attack_actions <= 12)))
        totals['movement'] += int(np.sum(
            (active_actions >= 1) & (active_actions <= 8)))
        if previous_actions is not None:
          totals['changed_actions'] += int(np.sum(
              active_actions != previous_actions))
          totals['change_opportunities'] += len(active_actions)
        if first_shot_step is None and np.any(attack_actions == 12):
          first_shot_step = step
        entropy = -(active_probabilities *
                    active_probabilities.clamp_min(1e-12).log()).sum(-1)
        top_two = active_probabilities.topk(2, dim=-1).values
        totals['entropy'] += float(entropy.sum())
        totals['max_probability'] += float(top_two[:, 0].sum())
        totals['probability_margin'] += float(
            (top_two[:, 0] - top_two[:, 1]).sum())

        observations, _, terminals, _, infos = env.step(action_array)
        done = bool(terminals.all())
        previous_actions = active_actions
        step += 1

      info = infos[0]
      score = float(info['score_reward'])
      success = bool(info['curriculum_success'])
      episode_rows.append({
          'episode': episode,
          'success': float(success),
          'score_reward': score,
          'length': step,
          'max_ball_progress': max_ball_progress,
          'first_shot_step': first_shot_step if first_shot_step is not None else step,
      })
  finally:
    env.close()

  decisions = totals['decisions']
  metrics = {
      'episodes': episodes,
      'success_rate': float(np.mean([row['success'] for row in episode_rows])),
      'mean_episode_length': float(np.mean(
          [row['length'] for row in episode_rows])),
      'mean_max_ball_progress': float(np.mean(
          [row['max_ball_progress'] for row in episode_rows])),
      'mean_first_shot_step': float(np.mean(
          [row['first_shot_step'] for row in episode_rows])),
      'policy_entropy': totals['entropy'] / decisions,
      'policy_entropy_fraction': totals['entropy'] / decisions / math.log(19),
      'policy_max_probability': totals['max_probability'] / decisions,
      'policy_probability_margin': totals['probability_margin'] / decisions,
      'active_agents_per_step': decisions / totals['environment_steps'],
      'max_abs_observation': totals['max_abs_observation'],
      'max_abs_observation_frame': (
          totals['max_abs_observation_index'] // 115),
      'max_abs_observation_feature': (
          totals['max_abs_observation_index'] % 115),
      'max_abs_ego_position': totals['max_abs_ego_position'],
      'action_change_rate': totals['changed_actions'] /
                            max(1, totals['change_opportunities']),
      'movement_fraction': totals['movement'] / decisions,
      'attacking_kick_fraction': totals['attacking_kicks'] /
                                 totals['attacking_decisions'],
      'attacking_shot_fraction': totals['attacking_shots'] /
                                 totals['attacking_decisions'],
  }
  action_fractions = {
      name: action_counts[index] / decisions
      for index, name in enumerate(ACTION_NAMES)
  }
  return metrics, action_fractions, episode_rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--episodes', type=int, default=20)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--attacker-only-levels', type=int, default=0)
  parser.add_argument('--curriculum-level', type=int, default=0)
  parser.add_argument('--wandb-project', default='google-football-fast-rl')
  parser.add_argument('--wandb-group', default='policy-postmortem')
  parser.add_argument('--no-wandb', action='store_true')
  args = parser.parse_args()

  results = {}
  rows = []
  for mode, greedy in (('sampled', False), ('greedy', True)):
    metrics, actions, episodes = evaluate(
        args.checkpoint, args.episodes, greedy, args.seed,
        args.attacker_only_levels, args.curriculum_level)
    results[mode] = {'metrics': metrics, 'action_fractions': actions}
    rows.extend([{'mode': mode, **row} for row in episodes])
  print(json.dumps(results, indent=2, sort_keys=True), flush=True)

  if not args.no_wandb:
    import wandb
    run = wandb.init(
        project=args.wandb_project, group=args.wandb_group,
        job_type='evaluation', config=vars(args))
    payload = {}
    for mode, result in results.items():
      payload.update({
          '{}/{}'.format(mode, key): value
          for key, value in result['metrics'].items()
      })
      payload.update({
          '{}/actions/{}'.format(mode, key): value
          for key, value in result['action_fractions'].items()
      })
    payload['episodes'] = wandb.Table(
        columns=list(rows[0]), data=[list(row.values()) for row in rows])
    run.log(payload)
    run.finish()


if __name__ == '__main__':
  main()
