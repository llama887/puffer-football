#!/bin/bash
#SBATCH --job-name=football-gate-smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.err
set -euo pipefail
REPO=/scratch/fyy2003/repos/football
ENV_PREFIX="$REPO/.venv"
TOOLCHAIN_PREFIX=/scratch/fyy2003/.conda/envs/football-fast
RUN_ROOT=/scratch/fyy2003/experiments/football-gate-smoke-${SLURM_JOB_ID}
mkdir -p "$RUN_ROOT/checkpoints"
module purge
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$REPO/third_party"
export GFOOTBALL_DATA_DIR="$REPO/third_party/gfootball_engine/data"
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$RUN_ROOT"
# 8 workers x 22 agents x 32 horizon = 5632 steps per epoch; 7 epochs total,
# promotion every 3 epochs -> evaluations at epochs 0, 3, 6.
python -u -m gfootball.examples.train_puffer \
  --device cpu --env-name 11_vs_11_advantage --num-workers 8 \
  --total-timesteps $((7 * 8 * 22 * 32)) \
  --promotion-interval 3 --promotion-episodes 32 --promotion-workers 8 \
  --greedy-promotion-episodes 16 --selfplay-promotion-episodes 16 \
  --promotion-early-abort-margin 0.2 \
  --curriculum-success-threshold 0.01 --promotion-worst-template-threshold 0.0 \
  --seed 3 --no-wandb --data-dir "$RUN_ROOT/checkpoints" 2>&1 | grep -E "PROMOTION|Traceback|Error|error|Saved model|^\{" | cut -c1-2500
echo "--- snapshots"; ls -la "$RUN_ROOT/checkpoints" | grep -i frozen
