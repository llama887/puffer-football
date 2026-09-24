#!/bin/bash
#SBATCH --job-name=install-gl-deps
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
export CONDA_PKGS_DIRS=/scratch/wz2445/.cache/conda/pkgs

conda install -y -p /scratch/wz2445/.conda/envs/football-fast \
  -c conda-forge \
  libglvnd libglvnd-devel libgl-devel libegl libegl-devel libglu xorg-libx11 xorg-libxext xorg-libxrender xorg-libxrandr xorg-libxi xorg-libxfixes xorg-libxcursor xorg-libxinerama

echo "DONE: GL deps installed"
