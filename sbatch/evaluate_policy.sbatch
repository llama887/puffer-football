#!/bin/bash
#SBATCH --job-name=football-policy-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_355_general
#SBATCH --chdir=/scratch/wz2445/puffer-football
#SBATCH --output=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.err

set -euo pipefail

CHECKPOINT=${1:?checkpoint path is required}
ATTACKER_ONLY_LEVELS=${ATTACKER_ONLY_LEVELS:-0}
CURRICULUM_LEVEL=${CURRICULUM_LEVEL:-0}
REPO=/scratch/wz2445/puffer-football
ENV_PREFIX="$REPO/.venv"
TOOLCHAIN_PREFIX=/scratch/wz2445/.conda/envs/football-fast
RUN_DIR=/scratch/wz2445/experiments/football-policy-eval-${SLURM_JOB_ID}

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
export PYTHONUNBUFFERED=1

if [ -f /scratch/wz2445/Puffer-Soccer/.env ]; then
  set -a
  source /scratch/wz2445/Puffer-Soccer/.env
  set +a
fi

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.evaluate_puffer_policy \
  --checkpoint "$CHECKPOINT" \
  --episodes 20 \
  --attacker-only-levels "$ATTACKER_ONLY_LEVELS" \
  --curriculum-level "$CURRICULUM_LEVEL" \
  --wandb-project google-football-fast-rl \
  --wandb-group policy-postmortem
