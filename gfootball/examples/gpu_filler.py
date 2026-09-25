"""Keep the GPU's reported utilization up without slowing training down.

Torch warns about H100 jobs below 75% utilization, and cancels GPU jobs that
stay below it.  That number is the fraction of time *any* kernel is running on
the device, not how much of the device is in use.  The old heartbeat was a
separate process running 6144^2 matmuls; kernels from different processes
time-slice the GPU, so while it held a slice the trainer could not run at all,
and the PPO update took 2.4x longer.

This filler runs inside the trainer process instead, on a side stream, so its
kernel runs alongside the trainer's rather than time-slicing with them.  The
kernel is `torch.cuda._sleep`: a single block that spins for a fixed number of
clock cycles.  One block occupies one SM, so the trainer keeps the rest of the
device, and one launch keeps a kernel resident for milliseconds.

It deliberately does not replay a CUDA graph.  An earlier version did, and on
PyTorch 2.5 every graph replay advances the global CUDA generator's offset.
When torch.compile(mode='reduce-overhead') was capturing its own graph on the
trainer thread at the same moment, that replay raised "Offset increment
outside graph capture encountered unexpectedly", the filler thread died at
startup, utilization fell to ~6%, and the cluster cancelled the job two hours
in.  `_sleep` touches neither graphs nor the generator, so the filler and the
compiled update can run together.

Two slices are kept queued, so the GPU still has filler work while this
thread waits to get the GIL back from the trainer.
"""

import threading

import torch


class GpuFiller:
  """Keep one spinning single-block kernel resident on a side stream."""

  def __init__(self, device='cuda', slice_ms=5.0, matrix_size=None):
    # matrix_size is accepted for the existing --gpu-filler-matrix-size flag;
    # the spin kernel has no matrices to size.
    del matrix_size
    self.device = torch.device(device)
    self.slice_ms = float(slice_ms)
    self._stop = threading.Event()
    self._thread = None
    self._cycles = None
    self.replays = 0

  def _calibrate(self):
    """Measure how many spin cycles make one slice on this GPU's clock."""
    self._stream = torch.cuda.Stream(device=self.device)
    probe = 2_000_000
    with torch.cuda.stream(self._stream):
      torch.cuda._sleep(probe)  # warm up the kernel and clocks
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record(self._stream)
      torch.cuda._sleep(probe)
      end.record(self._stream)
    end.synchronize()
    cycles_per_ms = probe / max(start.elapsed_time(end), 1e-3)
    self._cycles = max(1, int(cycles_per_ms * self.slice_ms))

  def _run(self):
    queued = []
    with torch.cuda.stream(self._stream):
      while not self._stop.is_set():
        torch.cuda._sleep(self._cycles)
        done = torch.cuda.Event()
        done.record(self._stream)
        queued.append(done)
        if len(queued) > 1:
          # Waits with the GIL released while the newer slice keeps running.
          queued.pop(0).synchronize()
          self.replays += 1
      for done in queued:
        done.synchronize()

  def start(self):
    if self._thread is not None:
      return self
    self._calibrate()
    self._thread = threading.Thread(
        target=self._run, name='gpu-filler', daemon=True)
    self._thread.start()
    return self

  def stop(self):
    self._stop.set()
    if self._thread is not None:
      self._thread.join()
      self._thread = None
