#!/bin/bash
# The rest of the one-clip experiments: seeds 1 and 2 of the schedules, their traces, and one longer run of the rule.
cd "$(dirname "$0")"
D=runs/v4/dev
S="--clips 26 --gains stiff --residual_rad 0.5 --ema 0.5 --damping 0.1 --action_rate 0.1 --power 0.3"
T="--levels 1.0 --iterations 0 --num_envs 2 --eval_envs 32 --ladder 0,4 --video_steps 550"
C="$S --teacher $D/tracker.pt --iterations 2400 --log_interval 100"
{
for seed in 1 2; do
    echo "--schedule deficit --gate --seed $seed --save $D/gate_s$seed.pt --out $D/gate_s$seed.json"
    echo "--share 0.5 --seed $seed --out $D/fixed0.50_s$seed.json"
    echo "--share 0.25 --seed $seed --out $D/fixed0.25_s$seed.json"
    echo "--schedule two_stage --seed $seed --out $D/two_stage_s$seed.json"
    echo "--schedule progress --seed $seed --out $D/progress_s$seed.json"
    echo "--share 1.0 --seed $seed --out $D/fixed1.00_s$seed.json"
done
echo "--schedule deficit --gate --seed 0 --iterations 4800 --save $D/gate_long_s0.pt --out $D/gate_long_s0.json"
echo "--share 0.5 --seed 0 --iterations 4800 --out $D/fixed0.50_long_s0.json"
} | xargs -P 4 -I {} bash -c "./run.sh train_parkour.py $C {} > \$(echo '{}' | sed 's/.*--out \(.*\)\.json/\1.log/') 2>&1"
for n in gate_s1 gate_s2 gate_long_s0; do
    ./run.sh train_parkour.py $S --load $D/$n.pt $T --trace $D/$n.npz > $D/${n}_trace.log 2>&1
done
echo V5 DEV DONE
for f in $D/*_s*.log; do case $f in *trace*|*tracker*) continue;; esac; echo "== $f $(grep LOG $f | tail -1 | cut -c1-175)"; done
./run.sh analyse_trace.py $D/gate_s0.npz $D/gate_s1.npz $D/gate_s2.npz $D/gate_long_s0.npz | grep -E "npz|clip|rung"
