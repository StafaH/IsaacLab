#!/bin/bash
# Run the queue of train_interfaces.py runs on one GPU, until the queue is empty:
#
#     setsid nohup scripts/reinforcement_learning/mdp2/run_queue.sh <GPU> > runs/research/worker<GPU>.log 2>&1 &
#
# The queue is the file runs/research/queue.txt of this checkout. Start one worker per GPU: each takes the first
# line of the file when it is free, so the file can be edited while the workers run. A line is
#
#     <checkout> <batch>/<name> <arguments of the learner>
#
# The run executes in <checkout>, a worktree with its own environment that pins its code, and writes
# runs/research/<batch>/<name>.{json,log,pt} here. A run whose log has a TRAIN line is skipped.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
QUEUE=$ROOT/runs/research/queue.txt
# Assets are downloaded to the temporary folder, which must belong to this user.
export TMPDIR=$ROOT/runs/tmp
mkdir -p "$TMPDIR"
while true; do
    line=$(flock "$QUEUE.lock" sh -c "sed -i '/^#/d;/^ *\$/d' '$QUEUE'; head -n 1 '$QUEUE'; sed -i 1d '$QUEUE'")
    [ -z "$line" ] && break
    read -r work run arguments <<< "$line"
    out=$ROOT/runs/research/$run
    grep -qs "^TRAIN" "$out.log" && continue
    mkdir -p "$(dirname "$out")"
    cd "$ROOT" && cd "$work" || continue
    echo "COMMIT $(git rev-parse --short HEAD) GPU $1 START $(date +%FT%T)" > "$out.log"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$1 .venv/bin/python scripts/reinforcement_learning/mdp2/train_interfaces.py \
        --out "$out.json" --save "$out.pt" $arguments >> "$out.log" 2>&1
    echo "EXIT $? $(date +%FT%T)" >> "$out.log"
done
