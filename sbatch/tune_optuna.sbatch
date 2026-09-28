#!/bin/bash
#SBATCH --job-name=football-optuna-level0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
# One Optuna trial per array task. Submit from the football repo root with
# OPTUNA_STUDY set to the study directory, e.g. --array=0-39%10.  Optuna runs
# from its own small venv (OPTUNA_PYTHON) so it never touches the trainer's
# environment, whose numpy comes from the conda toolchain.
#SBATCH --output=sbatch/logs/%x-%A_%a.out
#SBATCH --error=sbatch/logs/%x-%A_%a.err
set -euo pipefail
: "${OPTUNA_STUDY:?Set OPTUNA_STUDY to the study directory}"
# The trainer copies game data to node-local memory under this name; remove
# it even if the trainer is stopped early by the pruner.
trap 'rm -rf "/dev/shm/gfootball-data-${SLURM_JOB_ID}"' EXIT
# OPTUNA_STEPS shrinks a trial for smoke tests.
"${OPTUNA_PYTHON:-/scratch/$USER/envs/optuna/bin/python}" -u "$OPTUNA_STUDY/repo/scripts/tune_optuna.py" run \
  --study "$OPTUNA_STUDY" ${OPTUNA_STEPS:+--steps "$OPTUNA_STEPS"}
