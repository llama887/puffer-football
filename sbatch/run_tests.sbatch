#!/bin/bash
#SBATCH --job-name=football-unit-tests
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.err
set -uo pipefail
REPO=/scratch/fyy2003/repos/football
ENV_PREFIX="$REPO/.venv"
TOOLCHAIN_PREFIX=/scratch/fyy2003/.conda/envs/football-fast
module purge
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GFOOTBALL_DATA_DIR="$REPO/third_party/gfootball_engine/data"
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
status=0
export PYTHONPATH="$REPO:$REPO/third_party"
echo "=== train_puffer_test"; python -m gfootball.examples.train_puffer_test || status=1
echo "=== puffer_env_test";  python -m gfootball.env.puffer_env_test 2>&1 | grep -E "^\[|OK|FAIL|Error|Ran " || status=1
echo "=== evaluate_puffer_policy_test (this branch)"; python -m gfootball.examples.evaluate_puffer_policy_test 2>&1 | grep -E "^\[|OK|FAIL|Ran "
BASE=/scratch/fyy2003/experiments/football-level0-tuner
echo "=== evaluate_puffer_policy_test (base commit worktree, for comparison)"
(cd $BASE && PYTHONPATH="$BASE:$BASE/third_party" python -m gfootball.examples.evaluate_puffer_policy_test 2>&1 | grep -E "^\[|OK|FAIL|Ran ")
echo "EXIT $status"
exit $status
