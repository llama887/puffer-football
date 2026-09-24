#!/bin/bash
#SBATCH --job-name=build-football-fast
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_355_general
#SBATCH --partition=cpu_short
#SBATCH --chdir=/scratch/wz2445/puffer-football
#SBATCH --output=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.err

set -euo pipefail

export CONDA_PKGS_DIRS=/scratch/wz2445/.cache/conda/pkgs
module purge
module load anaconda3/2025.06

conda create -y -p /scratch/wz2445/.conda/envs/football-fast \
  -c pytorch -c nvidia -c conda-forge \
  python=3.10 \
  pytorch pytorch-cuda=12.4 \
  compilers cmake pkg-config make \
  mesalib sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-cpp

echo "DONE: football-fast env built"
