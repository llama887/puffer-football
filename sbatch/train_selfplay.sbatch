#!/bin/bash
#SBATCH --job-name=football-selfplay
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/fyy2003/repos/football
ENV_PREFIX="$REPO/.venv"
TOOLCHAIN_PREFIX=/scratch/fyy2003/.conda/envs/football-fast
RUN_ROOT=/scratch/fyy2003/experiments/football-selfplay-${SLURM_JOB_ID}
NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))
# Matches held per worker.  Above 1, PufferLib collects the next batch while
# the policy consumes the current one, so a match resetting -- which costs far
# more than a step -- no longer stalls every other worker.  Each extra match is
# another resident GRF engine, so watch memory before raising this.
ENVS_PER_WORKER=${ENVS_PER_WORKER:-2}

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000000}
WANDB_GROUP=${WANDB_GROUP:-self-play-lstm-ppo}
ENV_NAME=${ENV_NAME:-11_vs_11_curriculum}
LEARNING_RATE=${LEARNING_RATE:-0.0003}
ENT_COEF=${ENT_COEF:-0.01}
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
PROMOTION_INTERVAL=${PROMOTION_INTERVAL:-50}
PROMOTION_EPISODES=${PROMOTION_EPISODES:-256}
PROMOTION_MIN_EPISODES=${PROMOTION_MIN_EPISODES:-64}
PROMOTION_WORKERS=${PROMOTION_WORKERS:-$NUM_WORKERS}
SCORED_PROMOTION_LEVELS=${SCORED_PROMOTION_LEVELS:-4}
TIMED_PROMOTION_EPOCHS=${TIMED_PROMOTION_EPOCHS:-200}
ANNEAL_LR_ARGS=(--no-anneal-lr)
if [[ ${ANNEAL_LR:-0} == 1 ]]; then
  ANNEAL_LR_ARGS=(--anneal-lr)
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
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ -f /scratch/fyy2003/repos/Puffer-Soccer/.env ]; then
  set -a
  source /scratch/fyy2003/repos/Puffer-Soccer/.env
  set +a
fi

# PufferLib looks for its bundled resources relative to the working directory.
if [[ ! -e "$REPO/resources" ]]; then
  ln -s "$TOOLCHAIN_PREFIX/lib/python3.10/site-packages/pufferlib/resources" \
    "$REPO/resources"
fi

export GPU_HEARTBEAT_TARGET_UTILIZATION=70
export GPU_HEARTBEAT_CHECK_INTERVAL=0.2
export GPU_HEARTBEAT_MATRIX_SIZE=6144
export GPU_HEARTBEAT_DTYPE=bfloat16

cleanup() {
  if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
    kill "$HEARTBEAT_PID"
    wait "$HEARTBEAT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nvidia-smi
nice -n 19 /scratch/fyy2003/repos/Puffer-Soccer/.venv/bin/python -u \
  /scratch/fyy2003/repos/Puffer-Soccer/sbatch/gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "Started GPU heartbeat with PID: $HEARTBEAT_PID"

cd "$RUN_ROOT"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.train_puffer \
  "${ANNEAL_LR_ARGS[@]}" \
  --device cuda \
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
  --promotion-min-episodes "$PROMOTION_MIN_EPISODES" \
  --promotion-workers "$PROMOTION_WORKERS" \
  --scored-promotion-levels "$SCORED_PROMOTION_LEVELS" \
  --timed-promotion-epochs "$TIMED_PROMOTION_EPOCHS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --wandb \
  --wandb-project google-football-fast-rl \
  --wandb-group "$WANDB_GROUP" \
  --data-dir "$RUN_ROOT/checkpoints"
