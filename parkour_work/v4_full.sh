#!/bin/bash
# The experiments of all clips on the reworked task: the tracker, the policy, the even baseline, a second seed.
cd "$(dirname "$0")"
D=runs/v4/full
S="--gains stiff --residual_rad 0.5 --ema 0.5 --damping 0.1 --action_rate 0.1 --power 0.3"
BIG="$S --clips all --num_envs 8192 --eval_envs 1680 --iterations 12000 --log_interval 500 --hidden 1024 512 256"
RULE="--teacher $D/tracker.pt --teacher_log $D/tracker.json --schedule deficit --gate --allocation deficit"
./run.sh train_parkour.py $BIG --share 1.0 --tracker --allocation deficit --gate --out $D/tracker.json --save $D/tracker.pt > $D/tracker.log 2>&1
./run.sh train_parkour.py $BIG $RULE --out $D/credit_s0.json --save $D/credit_s0.pt > $D/credit_s0.log 2>&1
./run.sh train_parkour.py $BIG --teacher $D/tracker.pt --share 0.5 --out $D/even_s0.json > $D/even_s0.log 2>&1
./run.sh train_parkour.py $BIG $RULE --seed 1 --out $D/credit_s1.json --save $D/credit_s1.pt > $D/credit_s1.log 2>&1
./run.sh train_parkour.py $BIG $RULE --holdout_levels 0.9,1.1 --out $D/holdout_levels_s0.json > $D/holdout_levels_s0.log 2>&1
./run.sh train_parkour.py $BIG $RULE --holdout_clips 3,10,20,23,26 --out $D/holdout_clips_s0.json > $D/holdout_clips_s0.log 2>&1
echo V4 FULL DONE
for n in tracker credit_s0 even_s0 credit_s1 holdout_levels_s0 holdout_clips_s0; do echo "== $n $(grep LOG $D/$n.log | tail -1 | cut -c1-175)"; done
