#!/bin/bash
#SBATCH --job-name=football-selfplay-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_355_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2445@nyu.edu
#SBATCH --chdir=/scratch/wz2445/puffer-football
#SBATCH --output=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/wz2445/puffer-football
ENV_PREFIX="$REPO/.venv"
TOOLCHAIN_PREFIX=/scratch/wz2445/.conda/envs/football-fast
RUN_ROOT=/scratch/wz2445/experiments/football-selfplay-${SLURM_JOB_ID}

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000000}
WANDB_GROUP=${WANDB_GROUP:-self-play-lstm-ppo}
ENV_NAME=${ENV_NAME:-11_vs_11_curriculum}
LEARNING_RATE=${LEARNING_RATE:-0.0003}
# Left unset, these fall through to the tuned defaults in train_puffer.py
# rather than pinning the older, more expensive values.
ENT_COEF=${ENT_COEF:-0.001}
VF_COEF=${VF_COEF:-0.5}
CLIP_COEF=${CLIP_COEF:-0.2}
GAMMA=${GAMMA:-0.99}
GAE_LAMBDA=${GAE_LAMBDA:-0.95}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
BPTT_HORIZON=${BPTT_HORIZON:-32}
MINIBATCH_SEGMENTS=${MINIBATCH_SEGMENTS:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-256}
FRAME_STACK=${FRAME_STACK:-1}
SEED=${SEED:-0}
START_LEVEL=${START_LEVEL:-0}
PROMOTION_INTERVAL=${PROMOTION_INTERVAL:-100}
PROMOTION_EPISODES=${PROMOTION_EPISODES:-256}
# Measured on a 64-core cs node (5 min per config, cumulative steps/sec):
#   32w x 4env = 33.8K   48w x 4env = 30.6K   24w x 4env = 29.1K
#   40w x 2env = 25.8K   30w x 2env = 23.3K   30w x 1env = 16.1K
#   48w x 2env = 14.9K   62w x 2env =  7.7K   62w x 1env =  3.7K
# Two lessons: more envs per worker is the real lever (2.1x from 1 to 4,
# because a worker with spare matches keeps stepping while one resets), and
# MORE WORKERS IS WORSE past ~40 -- workers plus TORCH_THREADS must fit the
# core count or everything contends.  batch_size (= NUM_WORKERS) must also be
# divisible by ENVS_PER_WORKER, or make_vector_env refuses at startup.
NUM_WORKERS=${NUM_WORKERS:-32}
ENVS_PER_WORKER=${ENVS_PER_WORKER:-4}
PROMOTION_WORKERS=${PROMOTION_WORKERS:-$NUM_WORKERS}
# Unset means every level is score-gated, which is the tuned default.  Set it
# to a number to let levels at or above it advance on the timer instead.
SCORED_PROMOTION_LEVELS=${SCORED_PROMOTION_LEVELS:-}
TIMED_PROMOTION_EPOCHS=${TIMED_PROMOTION_EPOCHS:-200}
# The update and the evaluation forward run in this process, not the workers,
# and OMP_NUM_THREADS=1 below would otherwise pin them to a single core.
TORCH_THREADS=${TORCH_THREADS:-16}
ANNEAL_LR_ARGS=(--no-anneal-lr)
if [[ ${ANNEAL_LR:-0} == 1 ]]; then
  ANNEAL_LR_ARGS=(--anneal-lr)
fi
SCORED_ARGS=()
if [[ -n "$SCORED_PROMOTION_LEVELS" ]]; then
  SCORED_ARGS=(--scored-promotion-levels "$SCORED_PROMOTION_LEVELS")
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
# GRF re-reads its data directory inside the C++ engine on EVERY episode
# reset.  $REPO is on /scratch, which is NFS, and that read is the single
# largest cost in this job: measured 189ms for one worker and 3875ms once 30
# workers reset at the same time.  Staged on node-local storage the same
# reset is 27ms and does not degrade with concurrency at all (27.5ms at 30
# workers).  Stage it once per node and point the engine at the copy.
GF_DATA_SRC="$REPO/third_party/gfootball_engine/data"
GF_DATA_LOCAL="${SLURM_TMPDIR:-/dev/shm}/gfdata-${SLURM_JOB_ID:-$$}"
if cp -r "$GF_DATA_SRC" "$GF_DATA_LOCAL" 2>/dev/null; then
  export GFOOTBALL_DATA_DIR="$GF_DATA_LOCAL"
  trap 'rm -rf "$GF_DATA_LOCAL"' EXIT
  echo "staged GRF data on node-local storage: $GF_DATA_LOCAL" >&2
else
  export GFOOTBALL_DATA_DIR="$GF_DATA_SRC"
  echo "WARNING: could not stage GRF data locally; resets will be slow" >&2
fi
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ -f /scratch/wz2445/Puffer-Soccer/.env ]; then
  set -a
  source /scratch/wz2445/Puffer-Soccer/.env
  set +a
fi

# PufferLib looks for its bundled resources relative to the working directory.
if [[ ! -e "$REPO/resources" ]]; then
  ln -s "$TOOLCHAIN_PREFIX/lib/python3.10/site-packages/pufferlib/resources" \
    "$REPO/resources"
fi

cd "$RUN_ROOT"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.train_puffer \
  "${ANNEAL_LR_ARGS[@]}" \
  --device cpu \
  --torch-threads "$TORCH_THREADS" \
  --env-name "$ENV_NAME" \
  --num-workers "$NUM_WORKERS" \
  --envs-per-worker "$ENVS_PER_WORKER" \
  --frame-stack "$FRAME_STACK" \
  --hidden-size "$HIDDEN_SIZE" \
  --curriculum-window 20 \
  --curriculum-success-threshold 0.6 \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef "$ENT_COEF" \
  --vf-coef "$VF_COEF" \
  --clip-coef "$CLIP_COEF" \
  --gamma "$GAMMA" \
  --gae-lambda "$GAE_LAMBDA" \
  --update-epochs "$UPDATE_EPOCHS" \
  --bptt-horizon "$BPTT_HORIZON" \
  --minibatch-segments "$MINIBATCH_SEGMENTS" \
  --seed "$SEED" \
  --start-level "$START_LEVEL" \
  --promotion-interval "$PROMOTION_INTERVAL" \
  --promotion-episodes "$PROMOTION_EPISODES" \
  --promotion-workers "$PROMOTION_WORKERS" \
  ${SCORED_ARGS[@]+"${SCORED_ARGS[@]}"} \
  --timed-promotion-epochs "$TIMED_PROMOTION_EPOCHS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --wandb \
  --wandb-project google-football-fast-rl \
  --wandb-group "$WANDB_GROUP" \
  --data-dir "$RUN_ROOT/checkpoints"
