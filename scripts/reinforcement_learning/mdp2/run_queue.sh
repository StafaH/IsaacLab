#!/bin/bash
# Run the queue of the MDP2 interface experiments on one GPU, until the queue is empty:
#
#     setsid nohup scripts/reinforcement_learning/mdp2/run_queue.sh <GPU> > runs/research/worker<GPU>.log 2>&1 &
#
# The queue is the file runs/research/queue.txt of this checkout. Start one worker per GPU: each takes the first
# line of the file when it is free, so the file can be edited while the workers run. A line is
#
#     <checkout> <script> <batch>/<name> <arguments of the script>
#
# with train_interfaces.py or evaluate_interfaces.py as the script. The run executes in <checkout>, a worktree with
# its own environment that pins its code, and writes runs/research/<batch>/<name>.{json,log} here, and the model
# <name>.pt when it trains. A run whose log ends with "EXIT 0" is skipped.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
QUEUE=$ROOT/runs/research/queue.txt
# Assets are downloaded to the temporary folder, which must belong to this user.
export TMPDIR=$ROOT/runs/tmp
mkdir -p "$TMPDIR"
while true; do
    line=$(flock "$QUEUE.lock" sh -c "sed -i '/^#/d;/^ *\$/d' '$QUEUE'; head -n 1 '$QUEUE'; sed -i 1d '$QUEUE'")
    [ -z "$line" ] && break
    read -r work script run arguments <<< "$line"
    out=$ROOT/runs/research/$run
    grep -qs "^EXIT 0" "$out.log" && continue
    mkdir -p "$(dirname "$out")"
    cd "$ROOT" && cd "$work" || continue
    [ "$script" = train_interfaces.py ] && arguments="--save $out.pt $arguments"
    echo "COMMIT $(git rev-parse --short HEAD) GPU $1 START $(date +%FT%T)" > "$out.log"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$1 .venv/bin/python scripts/reinforcement_learning/mdp2/$script \
        --out "$out.json" $arguments >> "$out.log" 2>&1
    echo "EXIT $? $(date +%FT%T)" >> "$out.log"
done
