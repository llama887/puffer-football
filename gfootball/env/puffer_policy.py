"""The recurrent football policy, importable without the training loop.

Both the trainer and the environment workers need this network: the trainer
to learn it, the workers to run a frozen copy of it as the defending side
during promotion evaluation.  Keeping it here avoids importing the whole PPO
loop (and PufferLib's trainer) inside every environment process.
"""

import numpy as np
import torch

import pufferlib.pytorch


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
    keep = (None if done is None else
            (~done.bool()).to(hidden.dtype).unsqueeze(-1))
    # The input half of every LSTM gate depends only on the encoder output,
    # so it is one GEMM over the whole window instead of one per step.  Only
    # the recurrent half stays inside the loop.  Same math as LSTMCell.
    input_gates = torch.nn.functional.linear(encoded, self.cell.weight_ih)
    fused = encoded.is_cuda
    outputs = []
    for step in range(horizon):
      if keep is not None:
        hidden = hidden * keep[:, step]
        cell = cell * keep[:, step]
      hidden_gates = torch.nn.functional.linear(hidden, self.cell.weight_hh)
      if fused:
        # The kernel LSTMCell itself dispatches to on CUDA.
        hidden, cell, _ = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates[:, step], hidden_gates, cell,
            self.cell.bias_ih, self.cell.bias_hh)
      else:
        gates = (input_gates[:, step] + hidden_gates +
                 self.cell.bias_ih + self.cell.bias_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=-1)
        cell = (torch.sigmoid(forget_gate) * cell +
                torch.sigmoid(in_gate) * torch.tanh(cell_gate))
        hidden = torch.sigmoid(out_gate) * torch.tanh(cell)
      outputs.append(hidden)
    hidden = torch.stack(outputs, dim=1).reshape(
        segments * horizon, self.hidden_size)
    state['lstm_h'] = hidden.detach()
    state['lstm_c'] = cell.detach()
    return self.actor(hidden), self.critic(hidden).view(segments, horizon)


def hidden_size_from_state_dict(state_dict):
  """Recover the LSTM width so a checkpoint loads without its command line."""
  return int(state_dict['cell.weight_hh'].shape[1])


def save_policy_snapshot(policy, path):
  """Write a CPU copy of the weights that any process can load."""
  torch.save({key: value.detach().cpu()
              for key, value in policy.state_dict().items()}, path)


def load_frozen_policy(path, env):
  """Load a snapshot for inference only, on the CPU, single threaded."""
  state_dict = torch.load(path, map_location='cpu', weights_only=True)
  policy = FootballPolicy(env, hidden_size=hidden_size_from_state_dict(
      state_dict))
  policy.load_state_dict(state_dict)
  policy.eval()
  for parameter in policy.parameters():
    parameter.requires_grad_(False)
  return policy
