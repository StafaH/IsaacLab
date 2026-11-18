#!/bin/bash
# Evaluate the saved policy and the tracker of the development clip from starts displaced by growing amounts.
cd "$(dirname "$0")"
mkdir -p runs/displaced
COMMON="--clips 26 --iterations 200 --log_interval 200 --frozen --num_envs 64 --eval_envs 1280 --ladder 0,4,4,4"
for amount in 0.25 0.5 0.75 1.0; do
    ./run.sh train_parkour.py $COMMON --offset $amount --turn $amount --load runs/demo/student26.pt \
        --out runs/displaced/student_$amount.json > runs/displaced/student_$amount.log 2>&1
    ./run.sh train_parkour.py $COMMON --offset $amount --turn $amount --tracker --load runs/dev/tracker.pt \
        --out runs/displaced/tracker_$amount.json > runs/displaced/tracker_$amount.log 2>&1
done
echo DISPLACED DONE
