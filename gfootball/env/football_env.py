# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Allows different types of players to play against each other."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import importlib
from absl import logging

from gfootball.env import config as cfg
from gfootball.env import constants
from gfootball.env import football_action_set
from gfootball.env import football_env_core
from gfootball.env import observation_rotation
import gym
import numpy as np


class FootballEnv(gym.Env):
  """Allows multiple players to play in the same environment."""

  def __init__(self, config):
    self._config = config
    player_config = {'index': 0}
    # There can be at most one agent at a time. We need to remember its
    # team and the index on the team to generate observations appropriately.
    self._agent = None
    self._agent_index = -1
    self._agent_left_position = -1
    self._agent_right_position = -1
    self._players = self._construct_players(config['players'], player_config)
    self._env = football_env_core.FootballEnvCore(self._config)
    self._num_actions = len(football_action_set.get_action_set(self._config))
    self._cached_observation = None

  @property
  def action_space(self):
    if self._config.number_of_players_agent_controls() > 1:
      return gym.spaces.MultiDiscrete(
          [self._num_actions] * self._config.number_of_players_agent_controls())
    return gym.spaces.Discrete(self._num_actions)

  def _construct_players(self, definitions, config):
    result = []
    left_position = 0
    right_position = 0
    for definition in definitions:
      (name, d) = cfg.parse_player_definition(definition)
      config_name = 'player_{}'.format(name)
      if config_name in config:
        config[config_name] += 1
      else:
        config[config_name] = 0
      try:
        player_factory = importlib.import_module(
            'gfootball.env.players.{}'.format(name))
      except ImportError as e:
        logging.error('Failed loading player "%s"', name)
        logging.error(e)
        exit(1)
      player_config = copy.deepcopy(config)
      player_config.update(d)
      player = player_factory.Player(player_config, self._config)
      if name == 'agent':
        assert not self._agent, 'Only one \'agent\' player allowed'
        self._agent = player
        self._agent_index = len(result)
        self._agent_left_position = left_position
        self._agent_right_position = right_position
      result.append(player)
      left_position += player.num_controlled_left_players()
      right_position += player.num_controlled_right_players()
      config['index'] += 1
    return result

  def _convert_observations(self, original, player,
                            left_player_position, right_player_position):
    """Converts generic observations returned by the environment to
       the player specific observations.

    Args:
      original: original observations from the environment.
      player: player for which to generate observations.
      left_player_position: index into observation corresponding to the left
          player.
      right_player_position: index into observation corresponding to the right
          player.
    """
    observations = []
    copy_observation = not self._config['fast_mode']
    for is_left in [True, False]:
      count = (player.num_controlled_left_players() if is_left
               else player.num_controlled_right_players())
      if not count:
        continue
      adopted = original if is_left or player.can_play_right(
      ) else observation_rotation.flip_observation(original, self._config)
      prefix = 'left' if is_left or not player.can_play_right() else 'right'
      position = left_player_position if is_left else right_player_position
      controlled = adopted[prefix + '_agent_controlled_player']
      sticky = adopted[prefix + '_agent_sticky_actions']
      assert len(controlled) == len(sticky)
      designated = adopted[prefix + '_team_designated_player']
      # Every player on this side shares the same view of the world and differs
      # only in 'designated'/'active'/'sticky_actions'.  In fast mode the values
      # are shared references anyway, so build the common part once and
      # shallow-copy it -- with 22 controlled players the old per-player loop
      # over EXPOSED_OBSERVATIONS was 440 dict inserts per step.  Outside fast
      # mode each player must keep its own deep copy, so the base is rebuilt.
      base = None
      if not copy_observation:
        base = {v: adopted[v] for v in constants.EXPOSED_OBSERVATIONS}
        base['designated'] = designated
        # There is no frame for players on the right ATM.
        if is_left and 'frame' in original:
          base['frame'] = original['frame']
      for x in range(count):
        if base is not None:
          o = base.copy()
        else:
          o = {v: copy.deepcopy(adopted[v])
               for v in constants.EXPOSED_OBSERVATIONS}
          o['designated'] = designated
          if is_left and 'frame' in original:
            o['frame'] = original['frame']
        if position + x >= len(controlled):
          o['active'] = -1
          o['sticky_actions'] = []
        else:
          o['active'] = controlled[position + x]
          sticky_actions = sticky[position + x]
          o['sticky_actions'] = np.array(
              copy.deepcopy(sticky_actions) if copy_observation else
              sticky_actions, copy=copy_observation)
        observations.append(o)
    return observations

  def _action_to_list(self, a):
    if isinstance(a, np.ndarray):
      return a.tolist()
    if not isinstance(a, list):
      return [a]
    return a

  def _get_actions(self):
    obs = self._env.observation()
    left_actions = []
    right_actions = []
    left_player_position = 0
    right_player_position = 0
    for player in self._players:
      # A player that ignores its observations (the training agent, whose
      # action was already handed to it by set_action) does not need the
      # conversion, which for a 22-player match is the most expensive thing in
      # the step and would otherwise run twice -- once here on the pre-step
      # observation and once in observation() on the post-step one.
      if player.needs_observations():
        adopted_obs = self._convert_observations(obs, player,
                                                 left_player_position,
                                                 right_player_position)
        expected = len(adopted_obs)
      else:
        adopted_obs = None
        expected = player.num_controlled_players()
      left_player_position += player.num_controlled_left_players()
      right_player_position += player.num_controlled_right_players()
      a = self._action_to_list(player.take_action(adopted_obs))
      assert expected == len(
          a), 'Player provided {} actions instead of {}.'.format(len(a),
                                                                 expected)
      if not player.can_play_right():
        for x in range(player.num_controlled_right_players()):
          index = x + player.num_controlled_left_players()
          a[index] = observation_rotation.flip_single_action(
              a[index], self._config)
      left_actions.extend(a[:player.num_controlled_left_players()])
      right_actions.extend(a[player.num_controlled_left_players():])
    actions = left_actions + right_actions
    return actions

  def step(self, action):
    action = self._action_to_list(action)
    if self._agent:
      self._agent.set_action(action)
    else:
      assert len(
          action
      ) == 0, 'step() received {} actions, but no agent is playing.'.format(
          len(action))

    _, reward, done, info = self._env.step(self._get_actions())
    score_reward = reward
    if self._agent:
      reward = ([reward] * self._agent.num_controlled_left_players() +
                [-reward] * self._agent.num_controlled_right_players())
    self._cached_observation = None
    info['score_reward'] = score_reward
    return (self.observation(), np.array(reward, dtype=np.float32), done, info)

  def reset(self):
    self._env.reset()
    for player in self._players:
      player.reset()
    self._cached_observation = None
    return self.observation()

  def observation(self):
    if not self._cached_observation:
      self._cached_observation = self._env.observation()
      if self._agent:
        self._cached_observation = self._convert_observations(
            self._cached_observation, self._agent,
            self._agent_left_position, self._agent_right_position)
    return self._cached_observation

  def write_dump(self, name):
    return self._env.write_dump(name)

  def close(self):
    self._env.close()

  def get_state(self, to_pickle={}):
    return self._env.get_state(to_pickle)

  def set_state(self, state):
    self._cached_observation = None
    return self._env.set_state(state)

  def tracker_setup(self, start, end):
    self._env.tracker_setup(start, end)

  def render(self, mode='human'):
    self._cached_observation = None
    return self._env.render(mode=mode)

  def disable_render(self):
    self._cached_observation = None
    return self._env.disable_render()
