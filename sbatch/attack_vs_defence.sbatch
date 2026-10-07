#!/bin/bash
#SBATCH --job-name=football-attack-vs-defence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --time=03:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out
#SBATCH --error=sbatch/logs/%x-%A_%a.err
# scripts/attack_vs_defence.py over the rows of $CONFIGS, one row per array
# task: "LABEL CHECKPOINT DEFENCE SCENE".  Results go to $OUT/LABEL.json.
# Submit from the repo root, e.g.
#   sbatch --array=0-8 --account=<account> \
#     --export=ALL,CONFIGS=/abs/configs.txt,OUT=/abs/results \
#     sbatch/attack_vs_defence.sbatch
# VIDEO=1 also renders every episode to $OUT/LABEL/ (webm).  Rendering needs
# EGL, which the cs6xx CPU nodes lack: add --gres=gpu:1 and a GPU partition
# on the command line, e.g. --partition=h100_tandon,a100_tandon.
set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ENV_PREFIX=${ENV_PREFIX:-$REPO/.venv}
TOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX:-/scratch/$USER/.conda/envs/football-fast}
CONFIGS=${CONFIGS:?set CONFIGS to a file of "LABEL CHECKPOINT DEFENCE SCENE" rows}
OUT=${OUT:?set OUT to a results directory}
module purge
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$REPO/third_party:$REPO/third_party/gfootball_engine"
export GFOOTBALL_DATA_DIR="$REPO/third_party/gfootball_engine/data"
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export SDL_VIDEODRIVER=offscreen EGL_PLATFORM=surfaceless
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIGS")
read -r LABEL CHECKPOINT DEFENCE SCENE <<< "$LINE"
echo "$LABEL checkpoint=$CHECKPOINT defence=$DEFENCE scene=$SCENE"
VIDEO_ARGS=()
if [[ ${VIDEO:-0} == 1 ]]; then
  mkdir -p "$OUT/$LABEL"
  VIDEO_ARGS=(--video-dir "$OUT/$LABEL")
fi
cd "$REPO"
"$ENV_PREFIX/bin/python" -u scripts/attack_vs_defence.py \
  --checkpoint "$CHECKPOINT" --defence "$DEFENCE" --scene "$SCENE" \
  ${ADR_LOG:+--adr-log "$ADR_LOG"} --episodes "${EPISODES:-128}" \
  --seed "${SEED:-4242}" "${VIDEO_ARGS[@]}" --output "$OUT/$LABEL.json"
