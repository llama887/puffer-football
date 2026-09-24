#!/bin/bash
# Launch a seed sweep as a Slurm array against an immutable study directory.
#
#   bash sbatch/launch_train_seeds.sh              # 6 seeds, 3 at a time
#   bash sbatch/launch_train_seeds.sh 0-11%4       # 12 seeds, 4 at a time
#
# The repo is COPIED into the study, not symlinked, so editing the working
# tree while trials are queued cannot change what they run.  That matters
# here: a mid-flight checkout is exactly how this project last shipped four
# crashing refactors into a job that had already been submitted.
#
# Any knob the array sbatch reads can be overridden at launch:
#   TOTAL_TIMESTEPS=50000000 bash sbatch/launch_train_seeds.sh 0-2
# but pass it through --export below so the array tasks actually see it.

set -euo pipefail

ARRAY=${1:-0-5%3}
SRC=${SRC:-/scratch/wz2445/puffer-football}
STUDY_ROOT=${STUDY_ROOT:-/scratch/wz2445/experiments}

if [ ! -d "$SRC/gfootball" ]; then
  echo "SRC=$SRC does not look like the football repo" >&2
  exit 1
fi

mkdir -p "$STUDY_ROOT"
STUDY=$(mktemp -d "$STUDY_ROOT/football-seeds-$(date +%Y%m%d)-XXXXXX")
mkdir -p "$STUDY/results" "$STUDY/logs"

echo "study:  $STUDY"
echo "source: $SRC"
echo -n "snapshotting repo (~514M, excluding .git) ... "
# --link-dest would share inodes with a tree that is still being edited, so
# take a real copy.  .git is the only large thing a trial never reads.
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '.git' "$SRC/" "$STUDY/repo/"
else
  mkdir -p "$STUDY/repo"
  tar -C "$SRC" --exclude=.git -cf - . | tar -C "$STUDY/repo" -xf -
fi
echo "done"

# Pin the exact source state so results stay attributable.
git -C "$SRC" rev-parse HEAD > "$STUDY/repo/GIT_HEAD" 2>/dev/null || true
git -C "$SRC" status --short > "$STUDY/repo/GIT_DIRTY" 2>/dev/null || true
if [ -s "$STUDY/repo/GIT_DIRTY" ]; then
  echo "note: working tree had uncommitted changes; recorded in repo/GIT_DIRTY"
fi

chmod -R a-w "$STUDY/repo" 2>/dev/null || true

JOB=$(sbatch --parsable \
  --array="$ARRAY" \
  --output="$STUDY/logs/%x-%A_%a.out" \
  --error="$STUDY/logs/%x-%A_%a.err" \
  --export=ALL,TRAIN_STUDY="$STUDY" \
  "$STUDY/repo/sbatch/train_seeds.sbatch")

echo "submitted array $ARRAY as job $JOB"
cat <<EOF

  watch:    squeue -u \$USER -j $JOB
  logs:     $STUDY/logs/
  results:  $STUDY/results/seed<NN>/
  pace:     $SRC/.venv/bin/python $SRC/scripts/pace.py $STUDY/logs/football-seeds-${JOB}_0.out
EOF
