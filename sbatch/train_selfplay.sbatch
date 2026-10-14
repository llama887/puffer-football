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

# Single-seed results on this task are not reliable: seeds of the same
# configuration differ as much as different configurations do.  Submit as an
# array to get one run per seed, e.g. `sbatch --array=0-2 ...`; each task
# derives its seed from SEED_BASE and keeps its own run directory.
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
# PPO defaults are the entities-transformer settings of the September 29
# tuning study (trial 3), which every run since October 2 has used.
LEARNING_RATE=${LEARNING_RATE:-0.000543}
# 0.01 kept every run within 10% of a uniform policy after 3000+ epochs.
ENT_COEF=${ENT_COEF:-0.00159}
VF_COEF=${VF_COEF:-0.428}
CLIP_COEF=${CLIP_COEF:-0.132}
GAMMA=${GAMMA:-0.995}
GAE_LAMBDA=${GAE_LAMBDA:-0.927}
# Potential-based shaping on ball progress toward the attacked goal
# (Ng, Harada & Russell 1999): F = gamma*Phi(s') - Phi(s) with
# Phi = BALL_POTENTIAL * ball x toward the attacked goal (in [-1, 1]); the
# other team gets exactly the negative, so shaping is zero-sum.  Leaves the
# optimal policy unchanged; 0 disables.  Uses GAMMA above as the discount,
# which it must: a different discount leaves a real reward proportional
# to Phi.
BALL_POTENTIAL=${BALL_POTENTIAL:-0.5}
# Distance of the nearest teammate to the ball, combined
# with the ball potential before the same discounted potential difference.
PLAYER_POTENTIAL=${PLAYER_POTENTIAL:-0}
# Fade the player potential to 0 as the observed ball speed reaches this
# (ball-direction units: dribble <~0.27, struck ball ~1), so kicking costs no
# potential; 0 keeps the plain distance potential.
PLAYER_POTENTIAL_FADE_SPEED=${PLAYER_POTENTIAL_FADE_SPEED:-0}
# ADR (gfootball/adr.py): every episode draws its settings from ranges that
# grow and shrink with probe episodes.  ADR_PROBE_DEFENCE defends the probes:
# 'self' (the learner's starting weights) or a policy file, which the learner
# replaces once it scores ADR_PROBE_REFRESH_MARGIN more against it than
# against its own defence (checked every ADR_PROBE_REFRESH_INTERVAL epochs),
# or 'random' (uniform random defenders, never replaced).
ADR_ARGS=(--adr-probe-fraction "${ADR_PROBE_FRACTION:-0.2}"
          --adr-buffer-size "${ADR_BUFFER_SIZE:-64}"
          --adr-low-threshold "${ADR_LOW_THRESHOLD:-0.3}"
          --adr-high-threshold "${ADR_HIGH_THRESHOLD:-0.6}"
          --adr-probe-defence "${ADR_PROBE_DEFENCE:-self}"
          --adr-probe-refresh-interval "${ADR_PROBE_REFRESH_INTERVAL:-1000}"
          --adr-probe-refresh-margin "${ADR_PROBE_REFRESH_MARGIN:-0.1}")
# Every EVALUATION_INTERVAL epochs the policy is scored against uniform random
# defenders on fixed ADR scenes at each test difficulty (every setting from
# its start value out to that share of the way to its limits); empty
# disables.
EVALUATION_INTERVAL=${EVALUATION_INTERVAL:-100}
EVALUATION_DIFFICULTIES=${EVALUATION_DIFFICULTIES-0.25,0.5,1.0}
EVALUATION_EPISODES=${EVALUATION_EPISODES:-64}
EVALUATION_WORKERS=${EVALUATION_WORKERS:-$NUM_WORKERS}
# Optimizer steps per rollout.  Each step trains on the whole rollout (one
# segment per agent, ~0.6 GB on the GPU at 46 matches), so there is no
# minibatch size to set.
UPDATE_EPOCHS=${UPDATE_EPOCHS:-3}
BPTT_HORIZON=${BPTT_HORIZON:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-256}
# OBSERVATION=entities: egocentric offsets to players, ball, goals and
# touchlines, ball owner, keepers, offside, sticky buttons and steps left
# (gfootball/env/entity_observation.py);
# NETWORK=transformer runs attention over its players and ball (needs it).
OBSERVATION=${OBSERVATION:-entities}
NETWORK=${NETWORK:-transformer}
FRAME_STACK=${FRAME_STACK:-1}
# Throughput.  Agent steps trained per second, one L40S, level 0:
#   main as it was (32 CPUs)                          4.8k
#   + game data in node-local memory                 17k
#   + no heartbeat, sync-free update, async envs     23k
#   + CUDA-graph update (torch.compile)              36k
#   + env speedups from wesley-speed-improvement     40k
#   + whole-step update graph + overlap (32 CPUs)    59k
#   + 48 CPUs, 46 workers, graphed actor            143k  (env steps 143k)
#   + whole-rollout steps, packed cuDNN LSTM       122k  (at 7-32 epochs)
# None of these change what PPO computes except the last, which trains on
# the whole rollout per optimizer step instead of 32-segment minibatches.
#
# Async collection takes each batch from whichever workers are ready, so an
# engine reset in one match never stalls the others.  Overlap keeps the
# environments stepping while the update runs on the GPU, acting with a copy
# of the policy synced after every update (at most one update stale; the
# stored log-prob keeps PPO's ratio exact).
OVERLAP_ARGS=(--overlap-collection)
if [[ ${OVERLAP_COLLECTION:-1} == 0 ]]; then
  OVERLAP_ARGS=(--no-overlap-collection)
fi
ENVS_PER_WORKER=${ENVS_PER_WORKER:-1}
# The acting forward replays as one CUDA graph, and so does the update: one
# whole-rollout step is captured per rollout and replayed UPDATE_EPOCHS
# times, so the main thread keeps serving the environments.  GRAPH_UPDATE=0
# and GRAPH_ACTOR=0 fall back to eager.  COMPILE=1 adds torch.compile.
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
# Torch warns about GPU jobs below 75% utilization, which counts the time any
# kernel is running.  The trainer keeps it up itself (GPU_FILLER=1) with
# single-thread spin kernels on a side stream, which occupy one SM: 91% mean
# utilization at a 3% cost.  Filler matmuls spread over the device and slowed
# the latency-bound update (1024^2: 60k instead of 143k), and an external
# heartbeat process time-sliced the GPU and cut SPS to less than half.
GPU_FILLER_ARGS=(--gpu-filler)
if [[ ${GPU_FILLER:-1} == 0 ]]; then
  GPU_FILLER_ARGS=(--no-gpu-filler)
fi
ANNEAL_LR_ARGS=(--no-anneal-lr)
if [[ ${ANNEAL_LR:-0} == 1 ]]; then
  ANNEAL_LR_ARGS=(--anneal-lr)
fi
# FROZEN_DEFENCE_HORIZON: the memory window a snapshot probe defence trained
# by another run used (0: whole episodes); by default the same as this run's.
FROZEN_DEFENCE_ARGS=()
if [[ -n ${FROZEN_DEFENCE_HORIZON:-} ]]; then
  FROZEN_DEFENCE_ARGS=(--frozen-defence-horizon "$FROZEN_DEFENCE_HORIZON")
fi
# CARRY_LSTM_MEMORY=1 keeps recurrent memory across rollout segments (cleared
# only at episode ends) and trains each segment from the memory it started
# with, instead of restarting memory every BPTT_HORIZON steps.
MEMORY_ARGS=(--no-carry-lstm-memory)
if [[ ${CARRY_LSTM_MEMORY:-0} == 1 ]]; then
  MEMORY_ARGS=(--carry-lstm-memory)
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

cleanup() {
  if [[ -n ${LOCAL_DATA:-} ]]; then
    rm -rf "$LOCAL_DATA"
  fi
}
trap cleanup EXIT

nvidia-smi

cd "$RUN_ROOT"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.train_puffer \
  "${ANNEAL_LR_ARGS[@]}" \
  --device cuda \
  --num-workers "$NUM_WORKERS" \
  --envs-per-worker "$ENVS_PER_WORKER" \
  "${OVERLAP_ARGS[@]}" \
  "${GRAPH_ARGS[@]}" \
  "${GPU_FILLER_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  --log-interval-seconds "$LOG_INTERVAL_SECONDS" \
  --frame-stack "$FRAME_STACK" \
  --hidden-size "$HIDDEN_SIZE" \
  --observation "$OBSERVATION" \
  --network "$NETWORK" \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef "$ENT_COEF" \
  --vf-coef "$VF_COEF" \
  --clip-coef "$CLIP_COEF" \
  --gamma "$GAMMA" \
  --ball-potential "$BALL_POTENTIAL" \
  --player-potential "$PLAYER_POTENTIAL" \
  --player-potential-fade-speed "$PLAYER_POTENTIAL_FADE_SPEED" \
  "${ADR_ARGS[@]}" \
  --evaluation-interval "$EVALUATION_INTERVAL" \
  --evaluation-difficulties "$EVALUATION_DIFFICULTIES" \
  --evaluation-episodes "$EVALUATION_EPISODES" \
  --evaluation-workers "$EVALUATION_WORKERS" \
  --gae-lambda "$GAE_LAMBDA" \
  --update-epochs "$UPDATE_EPOCHS" \
  --bptt-horizon "$BPTT_HORIZON" \
  "${MEMORY_ARGS[@]}" \
  --seed "$SEED" \
  "${FROZEN_DEFENCE_ARGS[@]}" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --wandb \
  --wandb-project google-football-fast-rl \
  --wandb-group "$WANDB_GROUP" \
  ${WANDB_TAG:+--wandb-tag "$WANDB_TAG"} \
  --data-dir "$RUN_ROOT/checkpoints"
