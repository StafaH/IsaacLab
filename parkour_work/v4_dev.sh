#!/bin/bash
# The experiments of one clip on the reworked task: its tracker, then the schedules at three seeds.
cd "$(dirname "$0")"
D=runs/v4/dev
S="--clips 26 --gains stiff --residual_rad 0.5 --ema 0.5 --damping 0.1 --action_rate 0.1 --power 0.3"
T="--levels 1.0 --iterations 0 --num_envs 2 --eval_envs 32 --ladder 0,4 --video_steps 550"
# A tracker learns the climb in some seeds only: three are trained, and the one that climbs most teaches.
for seed in 0 1 2; do
    ./run.sh train_parkour.py $S --share 1.0 --tracker --iterations 2400 --log_interval 100 --seed $seed --out $D/tracker_s$seed.json --save $D/tracker_s$seed.pt > $D/tracker_s$seed.log 2>&1 &
done
wait
best=$(for seed in 0 1 2; do echo "$(grep LOG $D/tracker_s$seed.log | tail -1 | sed 's/.*by rung \([0-9.]*\).*/\1/') $seed"; done | sort -rn | head -1 | cut -d' ' -f2)
echo "TRACKER SEED $best"; for seed in 0 1 2; do grep LOG $D/tracker_s$seed.log | tail -1 | cut -c1-175; done
cp $D/tracker_s$best.pt $D/tracker.pt; cp $D/tracker_s$best.json $D/tracker.json; cp $D/tracker_s$best.log $D/tracker.log
./run.sh train_parkour.py $S --tracker --load $D/tracker.pt $T --trace $D/tracker.npz > $D/tracker_trace.log 2>&1
C="$S --teacher $D/tracker.pt --iterations 2400 --log_interval 100"
{
for seed in 0 1 2; do
    echo "--schedule deficit --gate --seed $seed --save $D/gate_s$seed.pt --out $D/gate_s$seed.json"
    echo "--share 0.5 --seed $seed --out $D/fixed0.50_s$seed.json"
    echo "--share 0.25 --seed $seed --out $D/fixed0.25_s$seed.json"
    echo "--schedule two_stage --seed $seed --out $D/two_stage_s$seed.json"
    echo "--schedule progress --seed $seed --out $D/progress_s$seed.json"
    echo "--share 1.0 --seed $seed --out $D/fixed1.00_s$seed.json"
done
} | xargs -P 3 -I {} bash -c "./run.sh train_parkour.py $C {} > \$(echo '{}' | sed 's/.*--out \(.*\)\.json/\1.log/') 2>&1"
for seed in 0 1 2; do
    ./run.sh train_parkour.py $S --load $D/gate_s$seed.pt $T --trace $D/gate_s$seed.npz > $D/gate_s${seed}_trace.log 2>&1
done
echo V4 DEV DONE
for n in tracker gate_s0 gate_s1 gate_s2 fixed0.50_s0 fixed0.25_s0 two_stage_s0 progress_s0 fixed1.00_s0; do echo "== $n $(grep LOG $D/$n.log | tail -1 | cut -c1-175)"; done
./run.sh analyse_trace.py $D/tracker.npz $D/gate_s0.npz $D/gate_s1.npz $D/gate_s2.npz | grep -E "npz|clip|rung"
