"""Keep the GPU's reported utilization up without slowing training down.

Torch warns about H100 jobs below 75% utilization.  That number is the
fraction of time *any* kernel is running on the device, not how much of the
device is in use.  The old heartbeat was a separate process running 6144^2
matmuls; kernels from different processes time-slice the GPU, so while it
held a slice the trainer could not run at all, and the PPO update took 2.4x
longer.

This filler runs inside the trainer process instead.  Kernels from one
process on different streams run side by side, so a filler that occupies a
few SMs at a time keeps a kernel resident nearly always while the trainer's
kernels keep the rest of the device.  The work is a captured CUDA graph of
small kernels, replayed from a background thread that waits with the GIL
released, so it costs the trainer almost no CPU either.  Two replays are
kept queued, so the GPU still has filler work while the thread waits to
get the GIL back from the trainer.
"""

import threading

import torch


class GpuFiller:
  """Replay a graph of filler kernels on a side stream until stopped.

  kind='sleep' (default) chains single-thread spin kernels: the device
  counts as busy while occupying one SM, so it barely touches training.
  kind='matmul' chains matrix_size^2 matmuls, which spread over the device.
  """

  def __init__(self, device='cuda', matrix_size=1024, matmuls_per_replay=400,
               dtype=torch.bfloat16, kind='sleep', sleep_cycles=100_000):
    if kind not in ('sleep', 'matmul'):
      raise ValueError('kind must be sleep or matmul')
    self.kind = kind
    self.sleep_cycles = int(sleep_cycles)
    self.device = torch.device(device)
    self.matrix_size = int(matrix_size)
    self.matmuls_per_replay = int(matmuls_per_replay)
    self.dtype = dtype
    self._stop = threading.Event()
    self._thread = None
    self.replays = 0

  def _capture(self):
    if self.kind == 'sleep':
      self._stream = torch.cuda.Stream(device=self.device)
      with torch.cuda.stream(self._stream):
        torch.cuda._sleep(self.sleep_cycles)
      self._stream.synchronize()
      self._graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self._graph, stream=self._stream):
        for _ in range(self.matmuls_per_replay):
          torch.cuda._sleep(self.sleep_cycles)
      return
    size = self.matrix_size
    self._left = torch.randn(size, size, device=self.device, dtype=self.dtype)
    self._right = torch.randn(size, size, device=self.device, dtype=self.dtype)
    self._out = torch.empty(size, size, device=self.device, dtype=self.dtype)
    # Graph capture must run on a side stream; warm cuBLAS up there first.
    self._stream = torch.cuda.Stream(device=self.device)
    with torch.cuda.stream(self._stream):
      for _ in range(3):
        torch.mm(self._left, self._right, out=self._out)
    self._stream.synchronize()
    self._graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self._graph, stream=self._stream):
      for _ in range(self.matmuls_per_replay):
        torch.mm(self._left, self._right, out=self._out)

  def _run(self):
    queued = []
    with torch.cuda.stream(self._stream):
      while not self._stop.is_set():
        self._graph.replay()
        done = torch.cuda.Event()
        done.record(self._stream)
        queued.append(done)
        if len(queued) > 1:
          # Waits with the GIL released while the newer replay keeps running.
          queued.pop(0).synchronize()
          self.replays += 1
      for done in queued:
        done.synchronize()

  def start(self):
    if self._thread is not None:
      return self
    self._capture()
    self._thread = threading.Thread(
        target=self._run, name='gpu-filler', daemon=True)
    self._thread.start()
    return self

  def stop(self):
    self._stop.set()
    if self._thread is not None:
      self._thread.join()
      self._thread = None
