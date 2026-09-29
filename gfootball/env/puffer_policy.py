"""The recurrent football policy, importable without the training loop.

Both the trainer and the environment workers need this network: the trainer
to learn it, the workers to run a frozen copy of it as the defending side
during promotion evaluation.  Keeping it here avoids importing the whole PPO
loop (and PufferLib's trainer) inside every environment process.
"""

import numpy as np
import torch

import pufferlib.pytorch

from gfootball.env import entity_observation as entity


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


class EntityEncoder(torch.nn.Module):
  """Attention over the players, the ball and the context of an entity row.

  The flat MLP sees 22 players as fixed slots and has to learn the same
  relationship (am I closer to the ball than my marker, who is open, who is
  offside) once per slot.  Here every entity is embedded by one shared
  projection per kind, and two pre-norm self-attention layers let every
  token read every other present token, so a relationship learned for one
  player applies to all of them, and absent players are masked out of
  attention.  The result is the same for any order of the other players.

  Output: the context, ball and own-player tokens and the mean of the
  present player tokens, projected to `output_size` for the LSTM.  Its
  input is the normalized row, plus the raw row for the present flags
  (the normalizer would move them off 0/1).
  """

  def __init__(self, output_size, width=64, layers=2, heads=4):
    super().__init__()
    self.context = torch.nn.Linear(entity.CONTEXT_FEATURES, width)
    self.ball = torch.nn.Linear(entity.BALL_FEATURES, width)
    self.player = torch.nn.Linear(entity.PLAYER_FEATURES, width)
    # Layers are called one by one: nn.TransformerEncoder would turn the
    # padding mask into nested tensors, which a CUDA graph cannot capture.
    self.blocks = torch.nn.ModuleList(
        torch.nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width, dropout=0.0,
            batch_first=True, norm_first=True)
        for _ in range(layers))
    self.norm = torch.nn.LayerNorm(width)
    self.output = torch.nn.Sequential(
        pufferlib.pytorch.layer_init(torch.nn.Linear(4 * width, output_size)),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(output_size))

  def forward(self, normalized, observations):
    rows = normalized.shape[0]
    players = normalized[:, entity.PLAYERS_START:].reshape(
        rows, entity.PLAYERS, entity.PLAYER_FEATURES)
    tokens = torch.cat((
        self.context(normalized[:, :entity.CONTEXT_FEATURES])[:, None],
        self.ball(normalized[:, entity.CONTEXT_FEATURES:
                             entity.PLAYERS_START])[:, None],
        self.player(players)), dim=1)
    present = observations[:, entity.PLAYERS_START:].reshape(
        rows, entity.PLAYERS, entity.PLAYER_FEATURES)[
            ..., entity.PLAYER_FEATURE_INDEX['present']] > 0.5
    # The context and ball tokens are always there, so no row attends to
    # nothing.  True marks a key to ignore.
    ignore = torch.cat((present.new_zeros(rows, 2), ~present), dim=1)
    for block in self.blocks:
      tokens = block(tokens, src_key_padding_mask=ignore)
    tokens = self.norm(tokens)
    weights = present.unsqueeze(-1).to(tokens.dtype)
    pooled = ((tokens[:, 2:] * weights).sum(dim=1) /
              weights.sum(dim=1).clamp_min(1.0))
    return self.output(torch.cat(
        (tokens[:, 0], tokens[:, 1], tokens[:, 2], pooled), dim=-1))


def _piece_steps(done):
  """Each grid position's episode piece, its step within it, and the piece's
  rank by length (longest first); positions are segment * horizon + step."""
  segments, horizon = done.shape
  total = segments * horizon
  starts = done.bool().clone()
  starts[:, 0] = True
  starts = starts.flatten()
  positions = torch.arange(total, device=done.device)
  piece = starts.long().cumsum(0) - 1
  piece_start = torch.cummax(
      torch.where(starts, positions, torch.zeros_like(positions)), 0).values
  step_in_piece = positions - piece_start
  lengths = torch.bincount(piece, minlength=total)
  rank = torch.empty_like(lengths)
  rank[torch.argsort(-lengths, stable=True)] = positions
  return step_in_piece, rank[piece]


def episode_batch_sizes(done):
  """How many episode pieces are still running at each step, on the CPU.

  The packed-sequence batch sizes episode_pieces needs for `done`; reading
  them is a host sync.  A caller that keeps a fixed layout checks these fit
  its capacity before calling episode_pieces with that capacity.
  """
  step_in_piece, _ = _piece_steps(done)
  return torch.bincount(step_in_piece, minlength=done.shape[1]).cpu()


def episode_pieces(done, batch_sizes=None):
  """Reorder a (segments, horizon) window into packed episode pieces.

  A training window holds several episodes whenever an episode ends inside
  it, and memory restarts from zero at each end (done[:, t] clears it before
  step t, as in the rollout).  Cutting every segment at its episode ends
  gives variable-length pieces that each start from zero memory, which is
  exactly what cuDNN's packed-sequence LSTM runs in one call.

  `batch_sizes` (CPU, one count per step, non-increasing) is the packed
  layout's number of slots per step; by default exactly
  episode_batch_sizes(done), which costs a host sync.  A larger capacity
  (every step at least what `done` needs) gives the same result: spare slots
  read the zero row FootballPolicy.forward appends and their outputs are
  never read, and packed sequences never mix.  A fixed capacity lets a
  captured CUDA graph serve many rollouts, and needs no sync here.

  Returns `order`, for each packed row the grid position it reads (the
  zero row, index segments * horizon, for spare slots); `inverse`, the
  packed row of each grid position; and `batch_sizes`.
  """
  segments, horizon = done.shape
  total = segments * horizon
  step_in_piece, rank = _piece_steps(done)
  if batch_sizes is None:
    batch_sizes = torch.bincount(step_in_piece, minlength=horizon).cpu()
    batch_sizes = batch_sizes[batch_sizes > 0]
  capacity = batch_sizes.to(done.device)
  first_slot = torch.cumsum(capacity, 0) - capacity
  inverse = first_slot[step_in_piece] + rank
  order = torch.full((int(batch_sizes.sum()),), total, dtype=torch.long,
                     device=done.device)
  order[inverse] = torch.arange(total, device=done.device)
  return order, inverse, batch_sizes


class FootballPolicy(torch.nn.Module):
  """Shared trunk into an LSTM, then an actor head and a value head.

  The trunk is `network`: 'mlp', two dense layers over the whole row (any
  observation), or 'transformer', EntityEncoder's attention over the
  players and ball (entity observations only).  The recurrence is one
  nn.LSTM layer.  Acting advances it one step at a time with the fused LSTM
  cell (forward_eval); training runs whole windows through cuDNN as packed
  episode pieces (forward).  Both use the same weights and give the same
  numbers.
  """

  is_continuous = False

  def __init__(self, env, hidden_size=256, network='mlp'):
    super().__init__()
    observation_size = int(np.prod(env.single_observation_space.shape))
    self.hidden_size = hidden_size
    self.network = network
    self.normalizer = RunningNormalizer(observation_size)
    if network == 'transformer':
      if observation_size != entity.SIZE:
        raise ValueError('the transformer reads entity observations '
                         '({} features), got {}'.format(
                             entity.SIZE, observation_size))
      self.encoder = EntityEncoder(hidden_size)
    elif network == 'mlp':
      self.encoder = torch.nn.Sequential(
          pufferlib.pytorch.layer_init(
              torch.nn.Linear(observation_size, hidden_size)),
          torch.nn.ReLU(),
          pufferlib.pytorch.layer_init(
              torch.nn.Linear(hidden_size, hidden_size)),
          torch.nn.ReLU(),
          torch.nn.LayerNorm(hidden_size),
      )
    else:
      raise ValueError("network must be 'mlp' or 'transformer'")
    self.rnn = torch.nn.LSTM(hidden_size, hidden_size)
    for name, parameter in self.rnn.named_parameters():
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

  def _encode(self, observations):
    """Trunk features of flat (rows, features) observations."""
    normalized = self.normalizer(observations)
    if self.network == 'transformer':
      return self.encoder(normalized, observations)
    return self.encoder(normalized)

  def forward_eval(self, observations, state):
    """Advance one environment step for every agent row.

    Used for acting (the trainer's collector, promotion, frozen defenders),
    where each row carries its own memory in `state` from step to step.
    torch.lstm_cell is the fused kernel LSTMCell runs, so a CUDA graph of
    this step captures cleanly.
    """
    observations = observations.float()
    hidden, cell = self._recurrent_state(
        state, observations.shape[0], observations)
    hidden, cell = self._reset_finished(hidden, cell, state.get('done'))
    encoded = self._encode(observations)
    hidden, cell = torch.lstm_cell(
        encoded, (hidden, cell), self.rnn.weight_ih_l0, self.rnn.weight_hh_l0,
        self.rnn.bias_ih_l0, self.rnn.bias_hh_l0)
    state['lstm_h'] = hidden
    state['lstm_c'] = cell
    return self.actor(hidden), self.critic(hidden).squeeze(-1)

  def forward(self, observations, state=None):
    """Backprop through time over (segments, horizon) training windows.

    Every window starts from zero memory, as every rollout segment does, and
    memory clears wherever state['done'] marks an episode start.  The window
    is cut into episode pieces (episode_pieces) and run as one packed cuDNN
    LSTM call instead of a Python loop of `horizon` small cell kernels; the
    result equals stepping forward_eval through the window.  A caller that
    runs the same windows repeatedly passes state['episode_pieces'] (the
    episode_pieces of its done flags) instead of 'done' to skip recomputing
    the layout and its host sync.
    """
    if observations.dim() == 2:
      return self.forward_eval(observations, dict(state or {}))
    state = dict(state or {})
    segments, horizon = observations.shape[:2]
    encoded = self._encode(
        observations.float().reshape(segments * horizon, -1))
    pieces = state.get('episode_pieces')
    if pieces is None:
      done = state.get('done')
      if done is None:
        done = encoded.new_zeros(segments, horizon)
      pieces = episode_pieces(done)
    order, inverse, batch_sizes = pieces
    # One zero row for the spare slots of a padded layout to read.
    padded = torch.cat((encoded, encoded.new_zeros(1, encoded.shape[1])))
    packed, _ = self.rnn(torch.nn.utils.rnn.PackedSequence(
        padded.index_select(0, order), batch_sizes))
    hidden = packed.data.index_select(0, inverse)
    return self.actor(hidden), self.critic(hidden).view(segments, horizon)


# Checkpoints before the packed cuDNN forward held the recurrence as an
# LSTMCell named `cell`.  nn.LSTM keeps exactly those tensors in the same
# gate order under rnn.*_l0, so those checkpoints load after a key rename.
_LSTM_CELL_KEYS = {
    'cell.weight_ih': 'rnn.weight_ih_l0',
    'cell.weight_hh': 'rnn.weight_hh_l0',
    'cell.bias_ih': 'rnn.bias_ih_l0',
    'cell.bias_hh': 'rnn.bias_hh_l0',
}


def upgrade_state_dict(state_dict):
  """Rename an LSTMCell-era checkpoint onto this policy's nn.LSTM names."""
  if 'cell.weight_hh' not in state_dict:
    return state_dict
  return {_LSTM_CELL_KEYS.get(key, key): value
          for key, value in state_dict.items()}


def hidden_size_from_state_dict(state_dict):
  """Recover the LSTM width so a checkpoint loads without its command line."""
  return int(upgrade_state_dict(state_dict)['rnn.weight_hh_l0'].shape[1])


def network_from_state_dict(state_dict):
  """Recover the trunk ('mlp' or 'transformer') a checkpoint was saved from."""
  if any(key.startswith('encoder.blocks.') for key in state_dict):
    return 'transformer'
  return 'mlp'


def save_policy_snapshot(policy, path):
  """Write a CPU copy of the weights that any process can load."""
  torch.save({key: value.detach().cpu()
              for key, value in policy.state_dict().items()}, path)


def load_frozen_policy(path, env):
  """Load a snapshot for inference only, on the CPU, single threaded."""
  state_dict = torch.load(path, map_location='cpu', weights_only=True)
  policy = FootballPolicy(
      env, hidden_size=hidden_size_from_state_dict(state_dict),
      network=network_from_state_dict(state_dict))
  policy.load_state_dict(upgrade_state_dict(state_dict))
  policy.eval()
  for parameter in policy.parameters():
    parameter.requires_grad_(False)
  return policy
