#!/bin/bash
# Render the four panels of the one-clip video: the clip, the tracker, the policy, the policy from a displaced start.
cd "$(dirname "$0")"
P=$PWD; D=$P/runs/v4/dev; O=$P/${OUT:-out5}
export DECK_FIRST=2 DECK_PICK=0 DECK_SHOW=1 DECK_W=960 DECK_H=540 DECK_GROUND=0.10,0.105,0.11 DECK_DOME=700 DECK_FOCAL=24
export DECK_BG=0.20,0.23,0.28 DECK_EYE=${EYE:--4.3,-2.5,0.9} DECK_CENTER=${CENTER:-0,0,0.25}
S="--clips 26 --gains stiff --residual_rad 0.5 --ema 0.5 --damping 0.1 --action_rate 0.1 --power 0.3"
T="--levels 1.0 --iterations 0 --num_envs 2 --eval_envs 1 --video_steps 352 --seed ${SEED:-0}"
for panel in ${PANELS:-clip tracker policy away}; do
    case $panel in
        clip) A="--tracker --load $D/tracker.pt --playback --ladder 0"; L="Retargeted clip";;
        tracker) A="--tracker --load $D/tracker.pt --ladder 0"; L="Motion tracker";;
        policy) A="--load $D/gate_long_s0.pt --ladder 0"; L="Policy";;
        away) A="--load $D/gate_long_s0.pt --ladder 4"; L="Policy from a displaced start";;
    esac
    DECK_LABELS="$L" ./render.sh train_parkour.py $S $A $T --video $O/$panel.mp4 > $O/$panel.log 2>&1
    ffmpeg -y -loglevel error -ss 3.5 -i $O/$panel.mp4 -frames:v 1 $O/$panel.jpg
done
