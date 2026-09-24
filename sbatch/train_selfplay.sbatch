#!/bin/bash
#SBATCH --job-name=football-selfplay
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
# Training is CPU-bound: environment throughput scales with cores, and Torch
# allows up to 48 CPUs per L40S-class GPU but only 32 per H100.  Any GPU
# type works; the trainer keeps GPU utilization up itself.
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --mail-type=END,FAIL
# Submit from the repo root; logs land in its sbatch/logs.  Slurm mails the
# submitting user.  Override account and GPU on the command line, e.g.
# sbatch --account=... --partition=l40s_public --gres=gpu:1 ...
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err

set -euo pipefail

# Paths follow whoever submits: the repo is the submit directory and runs go
# under /scratch/$USER/experiments.  Each one can be overridden.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ENV_PREFIX=${ENV_PREFIX:-$REPO/.venv}
TOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX:-/scratch/$USER/.conda/envs/football-fast}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-/scratch/$USER/experiments}
if [[ -z ${PUFFER_SOCCER:-} ]]; then
  for candidate in /scratch/$USER/repos/Puffer-Soccer /scratch/$USER/Puffer-Soccer; do
    if [[ -d $candidate ]]; then
      PUFFER_SOCCER=$candidate
      break
    fi
  done
fi
PUFFER_SOCCER=${PUFFER_SOCCER:-/scratch/$USER/Puffer-Soccer}
NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

# Single-seed results on this task are not reliable: the same configuration
# clears level 0 anywhere between 175 and 2275 epochs depending on the seed.
# Submit as an array to get one run per seed, e.g. `sbatch --array=0-2 ...`;
# each task derives its seed from SEED_BASE and keeps its own run directory.
# Summarize a set of runs with scripts/summarize_runs.py.
SEED_BASE=${SEED_BASE:-20260917}
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  SEED=${SEED:-$((SEED_BASE + SLURM_ARRAY_TASK_ID))}
  RUN_ROOT=$EXPERIMENTS_DIR/football-selfplay-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
else
  SEED=${SEED:-$SEED_BASE}
  RUN_ROOT=$EXPERIMENTS_DIR/football-selfplay-${SLURM_JOB_ID}
fi

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000000}
WANDB_GROUP=${WANDB_GROUP:-self-play-lstm-ppo}
ENV_NAME=${ENV_NAME:-11_vs_11_advantage}
LEARNING_RATE=${LEARNING_RATE:-0.0003}
# 0.01 kept every run within 10% of a uniform policy after 3000+ epochs.
ENT_COEF=${ENT_COEF:-0.001}
VF_COEF=${VF_COEF:-0.5}
CLIP_COEF=${CLIP_COEF:-0.2}
GAMMA=${GAMMA:-0.997}
GAE_LAMBDA=${GAE_LAMBDA:-0.98}
# Potential-based shaping on ball progress toward the attacked goal
# (Ng, Harada & Russell 1999): F = gamma*Phi(s') - Phi(s) with
# Phi = -BALL_POTENTIAL * distance-to-goal-line.  Leaves the optimal policy
# unchanged; 0 disables.  Uses GAMMA above as the discount.
BALL_POTENTIAL=${BALL_POTENTIAL:-0}
# Distance of the nearest teammate to the ball, combined
# with the ball potential before the same discounted potential difference.
PLAYER_POTENTIAL=${PLAYER_POTENTIAL:-0}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
BPTT_HORIZON=${BPTT_HORIZON:-32}
MINIBATCH_SEGMENTS=${MINIBATCH_SEGMENTS:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-256}
FRAME_STACK=${FRAME_STACK:-1}
START_LEVEL=${START_LEVEL:-0}
# 256 episodes every 25 epochs was ~40% of a job and 512+128+128 every 100
# against a frozen defence was ~50%.  256+64+64 every 100 epochs; larger and rarer
# is cheaper and less noisy, and a hopeless evaluation stops after a quarter.
PROMOTION_INTERVAL=${PROMOTION_INTERVAL:-100}
PROMOTION_EPISODES=${PROMOTION_EPISODES:-256}
PROMOTION_WORKERS=${PROMOTION_WORKERS:-$NUM_WORKERS}
PROMOTION_EARLY_ABORT_MARGIN=${PROMOTION_EARLY_ABORT_MARGIN:-0.2}
GREEDY_PROMOTION_EPISODES=${GREEDY_PROMOTION_EPISODES:-64}
SELFPLAY_PROMOTION_EPISODES=${SELFPLAY_PROMOTION_EPISODES:-64}
# Every level is score-gated.  The 200-epoch timer above level 3 pushed two
# seeds from level 4 to level 20 with success at zero once level 3 fell.
# Set SCORED_PROMOTION_LEVELS=4 to restore the old escalator.
SCORED_PROMOTION_LEVELS=${SCORED_PROMOTION_LEVELS:-21}
TIMED_PROMOTION_EPOCHS=${TIMED_PROMOTION_EPOCHS:-200}
# Throughput.  Agent steps trained per second, one L40S, level 0:
#   main as it was (32 CPUs)                          4.8k
#   + game data in node-local memory                 17k
#   + no heartbeat, sync-free update, async envs     23k
#   + CUDA-graph update (torch.compile)              36k
#   + env speedups from wesley-speed-improvement     40k
#   + whole-step update graph + overlap (32 CPUs)    59k
#   + 48 CPUs, 46 workers, graphed actor            143k  (env steps 143k)
# None of these change what PPO computes; MINIBATCH_SEGMENTS does (fewer,
# larger optimizer steps) and is left at 32.
#
# Async collection takes each batch from whichever workers are ready, so an
# engine reset in one match never stalls the others.  Overlap keeps the
# environments stepping while the update runs on the GPU, acting with a copy
# of the policy synced after every update (at most one update stale; the
# stored log-prob keeps PPO's ratio exact).
ASYNC_COLLECTION_ARGS=(--async-collection)
if [[ ${ASYNC_COLLECTION:-1} == 0 ]]; then
  ASYNC_COLLECTION_ARGS=(--no-async-collection)
fi
OVERLAP_ARGS=(--overlap-collection)
if [[ ${OVERLAP_COLLECTION:-1} == 0 ]]; then
  OVERLAP_ARGS=(--no-overlap-collection)
fi
ENVS_PER_WORKER=${ENVS_PER_WORKER:-1}
# A whole PPO minibatch step (forward, loss, backward, clip, Adam) replays as
# one CUDA graph, and so does the acting forward.  GRAPH_UPDATE=0 and
# GRAPH_ACTOR=0 fall back to eager.  COMPILE=1 adds torch.compile, which the
# graphed update makes redundant.
GRAPH_ARGS=(--graph-update --graph-actor)
if [[ ${GRAPH_UPDATE:-1} == 0 ]]; then
  GRAPH_ARGS[0]=--no-graph-update
fi
if [[ ${GRAPH_ACTOR:-1} == 0 ]]; then
  GRAPH_ARGS[1]=--no-graph-actor
fi
COMPILE_ARGS=(--no-compile)
if [[ ${COMPILE:-0} == 1 ]]; then
  COMPILE_ARGS=(--compile --compile-mode "${COMPILE_MODE:-reduce-overhead}")
fi
LOG_INTERVAL_SECONDS=${LOG_INTERVAL_SECONDS:-5}
ASYNC_PROMOTION_ARGS=(--async-promotion)
if [[ ${ASYNC_PROMOTION:-1} == 0 ]]; then
  ASYNC_PROMOTION_ARGS=(--no-async-promotion)
fi
# Greedy and self-play diagnostic evaluations run on every Nth promotion check
# (and on any check that passes the gate); they never decide promotion.
DIAGNOSTIC_PROMOTION_EVERY=${DIAGNOSTIC_PROMOTION_EVERY:-4}
# Torch warns about GPU jobs below 75% utilization, which counts the time any
# kernel is running.  The trainer keeps it up itself (GPU_FILLER=1) with
# single-thread spin kernels on a side stream, which occupy one SM: 91% mean
# utilization at a 3% cost.  Filler matmuls spread over the device and slowed
# the latency-bound update (1024^2: 60k instead of 143k).  The old external
# heartbeat (GPU_HEARTBEAT=1) is a separate process, which time-slices the
# GPU; it cut SPS to less than half.
GPU_FILLER_ARGS=(--gpu-filler --gpu-filler-kind "${GPU_FILLER_KIND:-sleep}"
                 --gpu-filler-matrix-size "${GPU_FILLER_MATRIX_SIZE:-1024}")
if [[ ${GPU_FILLER:-1} == 0 ]]; then
  GPU_FILLER_ARGS=(--no-gpu-filler)
fi
GPU_HEARTBEAT=${GPU_HEARTBEAT:-0}
ANNEAL_LR_ARGS=(--no-anneal-lr)
if [[ ${ANNEAL_LR:-0} == 1 ]]; then
  ANNEAL_LR_ARGS=(--anneal-lr)
fi
# The score gate plays against a frozen snapshot of the policy taken on
# entering the level, so "attackers score 60%" is a fixed target rather than
# the same network's ever-improving defence.  FROZEN_DEFENCE_GATE=0 restores
# the live self-play gate.
FROZEN_GATE_ARGS=(--frozen-defence-gate)
if [[ ${FROZEN_DEFENCE_GATE:-1} == 0 ]]; then
  FROZEN_GATE_ARGS=(--no-frozen-defence-gate)
fi

mkdir -p "$RUN_ROOT/checkpoints" "$REPO/sbatch/logs"

module purge
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Missing $ENV_PREFIX. Run: bash scripts/setup_uv_hpc.sh" >&2
  exit 1
fi
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$REPO/third_party"
export GFOOTBALL_DATA_DIR="$REPO/third_party/gfootball_engine/data"
# Every engine reset re-reads animations, meshes and bitmaps from the data
# dir.  From /scratch that is ~280 ms per reset with multi-second stragglers
# when 30 workers reset at once; from node-local memory it is ~28 ms.
if [[ ${LOCAL_GAME_DATA:-1} == 1 ]]; then
  LOCAL_DATA=/dev/shm/gfootball-data-${SLURM_JOB_ID}
  mkdir -p "$LOCAL_DATA"
  cp -r "$GFOOTBALL_DATA_DIR/." "$LOCAL_DATA/"
  export GFOOTBALL_DATA_DIR="$LOCAL_DATA"
fi
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ -f "$PUFFER_SOCCER/.env" ]; then
  set -a
  source "$PUFFER_SOCCER/.env"
  set +a
fi

# PufferLib looks for its bundled resources relative to the working directory.
if [[ ! -e "$REPO/resources" ]]; then
  ln -s "$TOOLCHAIN_PREFIX/lib/python3.10/site-packages/pufferlib/resources" \
    "$REPO/resources"
fi

export GPU_HEARTBEAT_TARGET_UTILIZATION="${GPU_HEARTBEAT_TARGET_UTILIZATION:-85}"
export GPU_HEARTBEAT_CHECK_INTERVAL=0.2
export GPU_HEARTBEAT_MATRIX_SIZE=6144
export GPU_HEARTBEAT_DTYPE=bfloat16

cleanup() {
  if [[ -n ${LOCAL_DATA:-} ]]; then
    rm -rf "$LOCAL_DATA"
  fi
  if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
    kill "$HEARTBEAT_PID"
    wait "$HEARTBEAT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nvidia-smi
if [[ $GPU_HEARTBEAT == 1 ]]; then
  nice -n 19 "$PUFFER_SOCCER/.venv/bin/python" -u \
    "$PUFFER_SOCCER/sbatch/gpu_heartbeat.py" &
  HEARTBEAT_PID=$!
  echo "Started GPU heartbeat with PID: $HEARTBEAT_PID"
fi

cd "$RUN_ROOT"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.train_puffer \
  "${ANNEAL_LR_ARGS[@]}" \
  --device cuda \
  --env-name "$ENV_NAME" \
  --num-workers "$NUM_WORKERS" \
  --envs-per-worker "$ENVS_PER_WORKER" \
  "${ASYNC_COLLECTION_ARGS[@]}" \
  "${OVERLAP_ARGS[@]}" \
  "${GRAPH_ARGS[@]}" \
  --diagnostic-promotion-every "$DIAGNOSTIC_PROMOTION_EVERY" \
  "${ASYNC_PROMOTION_ARGS[@]}" \
  "${GPU_FILLER_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  --log-interval-seconds "$LOG_INTERVAL_SECONDS" \
  --frame-stack "$FRAME_STACK" \
  --hidden-size "$HIDDEN_SIZE" \
  --curriculum-window 20 \
  --curriculum-success-threshold 0.6 \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef "$ENT_COEF" \
  --vf-coef "$VF_COEF" \
  --clip-coef "$CLIP_COEF" \
  --gamma "$GAMMA" \
  --ball-potential "$BALL_POTENTIAL" \
  --player-potential "$PLAYER_POTENTIAL" \
  --gae-lambda "$GAE_LAMBDA" \
  --update-epochs "$UPDATE_EPOCHS" \
  --bptt-horizon "$BPTT_HORIZON" \
  --minibatch-segments "$MINIBATCH_SEGMENTS" \
  --seed "$SEED" \
  --start-level "$START_LEVEL" \
  --promotion-interval "$PROMOTION_INTERVAL" \
  --promotion-episodes "$PROMOTION_EPISODES" \
  --promotion-workers "$PROMOTION_WORKERS" \
  --promotion-early-abort-margin "$PROMOTION_EARLY_ABORT_MARGIN" \
  --greedy-promotion-episodes "$GREEDY_PROMOTION_EPISODES" \
  --selfplay-promotion-episodes "$SELFPLAY_PROMOTION_EPISODES" \
  "${FROZEN_GATE_ARGS[@]}" \
  --scored-promotion-levels "$SCORED_PROMOTION_LEVELS" \
  --timed-promotion-epochs "$TIMED_PROMOTION_EPOCHS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --wandb \
  --wandb-project google-football-fast-rl \
  --wandb-group "$WANDB_GROUP" \
  ${WANDB_TAG:+--wandb-tag "$WANDB_TAG"} \
  --data-dir "$RUN_ROOT/checkpoints"
