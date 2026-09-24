#!/bin/bash
#SBATCH --job-name=football-shaping-tune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%A_%a.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%A_%a.err
set -euo pipefail
: "${SHAPING_STUDY:?Set SHAPING_STUDY to the immutable study directory}"
module purge
export PYTHONPATH="$SHAPING_STUDY/repo:$SHAPING_STUDY/repo/third_party"
export LD_LIBRARY_PATH="/scratch/fyy2003/.conda/envs/football-fast/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GFOOTBALL_DATA_DIR="$SHAPING_STUDY/repo/third_party/gfootball_engine/data"
export GFOOTBALL_FONT="$SHAPING_STUDY/repo/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
cd "$SHAPING_STUDY/repo"
/scratch/fyy2003/repos/football/.venv/bin/python -u scripts/tune_shaping.py run \
  --study "$SHAPING_STUDY" --index "$SLURM_ARRAY_TASK_ID"
