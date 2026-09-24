#!/bin/bash
#SBATCH --job-name=build-pufferlib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_355_general
#SBATCH --partition=cpu_short
#SBATCH --chdir=/scratch/wz2445/puffer-football
#SBATCH --output=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.err

set -eo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/wz2445/.conda/envs/football-fast

export TMPDIR=/scratch/wz2445/.cache/tmp
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

python -m pip install --no-input --cache-dir /scratch/wz2445/.cache/pip 'pufferlib>=3.0,<3.1'

echo "DONE: pufferlib installed"
