#!/bin/bash
cd "$(dirname "$0")"
{
echo "--ema 0.5 --damping 0.1 --action_rate 0.1 --power 0.3 --out runs/v5/proxy/base.json"
echo "--ema 1.0 --damping 0.1 --action_rate 0.1 --power 0.3 --out runs/v5/proxy/ema1.json"
echo "--ema 0.5 --damping 0.04 --action_rate 0.1 --power 0.3 --out runs/v5/proxy/damp004.json"
echo "--ema 0.5 --damping 0.1 --action_rate 0 --power 0 --out runs/v5/proxy/nocost.json"
} | xargs -P 2 -I {} bash -c "./run.sh train_parkour.py --clips 10,11,24 --gains stiff --residual_rad 0.5 --share 1.0 --tracker --iterations 2400 --log_interval 200 --eval_envs 360 {} > \$(echo '{}' | sed 's/.*--out \(.*\)\.json/\1.log/') 2>&1"
for n in base ema1 damp004 nocost; do echo "== $n $(grep LOG runs/v5/proxy/$n.log | tail -1 | cut -c1-175)"; done
