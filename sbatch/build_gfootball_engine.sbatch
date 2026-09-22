#!/bin/bash
#SBATCH --job-name=build-gfootball-engine
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_355_general
#SBATCH --partition=cpu_short
#SBATCH --chdir=/scratch/wz2445/puffer-football
#SBATCH --output=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/wz2445/puffer-football/sbatch/logs/%x-%j.err

set -eo pipefail

REPO=/scratch/wz2445/puffer-football
TOOLCHAIN_PREFIX=/scratch/wz2445/.conda/envs/football-fast
VENV="$REPO/.venv"

module purge
export PATH="$TOOLCHAIN_PREFIX/bin:$PATH"
export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPython_ROOT_DIR=$VENV -DPython_FIND_VIRTUALENV=ONLY -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON"
export CC="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CMAKE_PREFIX_PATH="$TOOLCHAIN_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$TOOLCHAIN_PREFIX/include${CPATH:+:$CPATH}"
export PKG_CONFIG_PATH="$TOOLCHAIN_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR=/scratch/wz2445/.cache/tmp

cd "$REPO/third_party/gfootball_engine"
cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
make -j "$SLURM_CPUS_PER_TASK"
ln -sf libgame.so _gameplayfootball.so

echo "DONE: gfootball_engine built"
