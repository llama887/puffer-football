#!/bin/bash
#SBATCH --job-name=football-ppo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100_tandon
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2445@nyu.edu
#SBATCH --chdir=/scratch/wz2445/puffer-football
#SBATCH --output=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/wz2445/puffer-football
ENV_PREFIX="$REPO/.venv"
TOOLCHAIN_PREFIX=/scratch/wz2445/.conda/envs/football-fast
RUN_ROOT=/scratch/wz2445/experiments/football-ppo-${SLURM_JOB_ID}
NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000000}
WANDB_GROUP=${WANDB_GROUP:-simple-marl-ppo}
ATTACKER_ONLY_LEVELS=${ATTACKER_ONLY_LEVELS:-13}
CRITIC_LEARNING_RATE=${CRITIC_LEARNING_RATE:-0.00001}
LEARNING_RATE=${LEARNING_RATE:-0.00008}
ENT_COEF=${ENT_COEF:-0.01}
SEED=${SEED:-0}
PROMOTION_INTERVAL=${PROMOTION_INTERVAL:-50}
PROMOTION_EPISODES=${PROMOTION_EPISODES:-256}
PROMOTION_WORKERS=${PROMOTION_WORKERS:-$NUM_WORKERS}
GRADIENT_AUDIT_ARGS=()
if [[ ${GRADIENT_AUDIT:-0} == 1 ]]; then
  GRADIENT_AUDIT_ARGS+=(--gradient-audit)
fi
ANNEAL_LR_ARGS=(--anneal-lr)
if [[ ${ANNEAL_LR:-1} == 0 ]]; then
  ANNEAL_LR_ARGS=(--no-anneal-lr)
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

if [ -f /scratch/wz2445/Puffer-Soccer/.env ]; then
  set -a
  source /scratch/wz2445/Puffer-Soccer/.env
  set +a
fi

export GPU_HEARTBEAT_TARGET_UTILIZATION="${GPU_HEARTBEAT_TARGET_UTILIZATION:-85}"
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
nice -n 19 /scratch/wz2445/Puffer-Soccer/.venv/bin/python -u \
  /scratch/wz2445/Puffer-Soccer/sbatch/gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "Started GPU heartbeat with PID: $HEARTBEAT_PID"

cd "$RUN_ROOT"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.train_puffer \
  "${GRADIENT_AUDIT_ARGS[@]}" \
  "${ANNEAL_LR_ARGS[@]}" \
  --device cuda \
  --num-workers "$NUM_WORKERS" \
  --frame-stack 4 \
  --curriculum-levels 33 \
  --curriculum-window 20 \
  --curriculum-success-threshold 0.6 \
  --attacker-only-levels "$ATTACKER_ONLY_LEVELS" \
  --critic-learning-rate "$CRITIC_LEARNING_RATE" \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef "$ENT_COEF" \
  --seed "$SEED" \
  --promotion-interval "$PROMOTION_INTERVAL" \
  --promotion-episodes "$PROMOTION_EPISODES" \
  --promotion-workers "$PROMOTION_WORKERS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --wandb \
  --wandb-project google-football-fast-rl \
  --wandb-group "$WANDB_GROUP" \
  --data-dir "$RUN_ROOT/checkpoints"
