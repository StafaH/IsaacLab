#!/bin/bash
# Run a batch of train_interfaces.py, one run per GPU:
#
#     scripts/reinforcement_learning/mdp2/launch_batch.sh <batch> "<GPUs, such as 0 1 2 3>" [checkout]
#
# The batch is the file runs/research/<batch>.txt of this checkout, with one run per line: a name, then the
# arguments of the learner. The runs execute in [checkout], a worktree with its own environment that pins their
# code (this checkout by default), and write runs/research/<batch>/<name>.{json,log,pt} here. Runs whose log has
# a TRAIN line are skipped, so that a batch can be launched again after it was interrupted.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BATCH=$1
WORK=${3:-$ROOT}
export BATCH WORK GPU_LIST="$2" OUT=$ROOT/runs/research/$1
# Assets are downloaded to the temporary folder, which must belong to this user.
export TMPDIR=$ROOT/runs/tmp
mkdir -p "$OUT" "$TMPDIR"
cd "$WORK" || exit 1
grep -v '^#' "$ROOT/runs/research/$BATCH.txt" | xargs -P "$(wc -w <<< "$2")" --process-slot-var=SLOT -L 1 bash -c '
    gpus=($GPU_LIST)
    grep -qs "^TRAIN" $OUT/$0.log && exit 0
    echo "COMMIT $(git rev-parse --short HEAD) GPU ${gpus[$SLOT]} START $(date +%FT%T)" > $OUT/$0.log
    CUDA_VISIBLE_DEVICES=${gpus[$SLOT]} .venv/bin/python scripts/reinforcement_learning/mdp2/train_interfaces.py \
        --out $OUT/$0.json --save $OUT/$0.pt "$@" >> $OUT/$0.log 2>&1
    echo "EXIT $? $(date +%FT%T)" >> $OUT/$0.log'
