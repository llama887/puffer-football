#!/bin/bash
#SBATCH --job-name=football-shaping-tune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# Submit from the repo root.  48 CPUs: environment throughput scales with
# cores (Torch allows 48 per L40S-class GPU).  Any GPU type works: training
# is CPU-bound and the trainer keeps GPU utilization up itself.
#SBATCH --output=sbatch/logs/%x-%A_%a.out
#SBATCH --error=sbatch/logs/%x-%A_%a.err
set -euo pipefail
: "${SHAPING_STUDY:?Set SHAPING_STUDY to the immutable study directory}"
module purge
export ENV_PREFIX=${ENV_PREFIX:-${SLURM_SUBMIT_DIR}/.venv}
export TOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX:-/scratch/$USER/.conda/envs/football-fast}
export PYTHONPATH="$SHAPING_STUDY/repo:$SHAPING_STUDY/repo/third_party"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GFOOTBALL_FONT="$SHAPING_STUDY/repo/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
# The held-out benchmark after training resets engines constantly; read the
# game data from node-local memory, not /scratch.  The trainer makes its own
# copy and removes it when it exits, so this one has a different name.
BENCH_DATA=/dev/shm/gfootball-bench-${SLURM_JOB_ID}
mkdir -p "$BENCH_DATA"
cp -r "$SHAPING_STUDY/repo/third_party/gfootball_engine/data/." "$BENCH_DATA/"
trap 'rm -rf "$BENCH_DATA"' EXIT
export GFOOTBALL_DATA_DIR="$BENCH_DATA"
cd "$SHAPING_STUDY/repo"
# SHAPING_STEPS / SHAPING_BENCHMARK_EPISODES shrink a trial for smoke tests;
# summarize only ranks trials run at the full budget.
"$ENV_PREFIX/bin/python" -u scripts/tune_shaping.py run \
  --study "$SHAPING_STUDY" --index "$SLURM_ARRAY_TASK_ID" \
  ${SHAPING_STEPS:+--steps "$SHAPING_STEPS"} \
  ${SHAPING_BENCHMARK_EPISODES:+--benchmark-episodes "$SHAPING_BENCHMARK_EPISODES"}
