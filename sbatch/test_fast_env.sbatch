#!/bin/bash
#SBATCH --job-name=football-fast-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
REPO=/scratch/fyy2003/repos/football
ENV_PREFIX="$REPO/.venv"
TOOLCHAIN_PREFIX=/scratch/fyy2003/.conda/envs/football-fast
mkdir -p sbatch/logs

bash scripts/setup_uv_hpc.sh

module purge
export PATH="$ENV_PREFIX/bin:$PATH"
export CMAKE_ARGS="-DPython_ROOT_DIR=$ENV_PREFIX -DPython_FIND_VIRTUALENV=ONLY -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON"
export CC="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CMAKE_PREFIX_PATH="$TOOLCHAIN_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$TOOLCHAIN_PREFIX/include${CPATH:+:$CPATH}"
export PKG_CONFIG_PATH="$TOOLCHAIN_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ ! -e gfootball_engine ] && [ ! -L gfootball_engine ]; then
  ln -s third_party/gfootball_engine gfootball_engine
fi
if [ ! -e third_party/gfootball_engine/fonts ] && \
   [ ! -L third_party/gfootball_engine/fonts ]; then
  ln -s ../fonts third_party/gfootball_engine/fonts
fi
if [ -f third_party/gfootball_engine/Makefile ]; then
  make -C third_party/gfootball_engine clean
fi
gfootball/build_game_engine.sh
"$ENV_PREFIX/bin/python" -m gfootball.env.football_env_test \
  FootballEnvTest.test_fast_mode_matches_baseline_dynamics0 \
  FootballEnvTest.test_fast_mode_matches_baseline_dynamics1 \
  FootballEnvTest.test_fast_mode_matches_baseline_dynamics2
"$ENV_PREFIX/bin/python" -m gfootball.env.puffer_env_test
"$ENV_PREFIX/bin/python" -m gfootball.env.observation_equivalence_test
"$ENV_PREFIX/bin/python" -m gfootball.examples.benchmark_fast_mode \
  --steps=500 --repeats=3 --players=11
# Where the remaining per-step time goes, at the training shape.
"$ENV_PREFIX/bin/python" -m gfootball.examples.benchmark_fast_mode \
  --profile --steps=500 --players=11
