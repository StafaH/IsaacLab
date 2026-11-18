#!/bin/bash
# Run a script of this folder, or of scripts/reinforcement_learning/mdp2, against this checkout. Works from this folder.
P=$(cd "$(dirname "$0")" && pwd)
W=$(dirname "$P")
S=$W/scripts/reinforcement_learning/mdp2
export PYTHONPATH=$W/source/isaaclab:$W/source/isaaclab_experimental:$W/source/isaaclab_tasks:$W/source/isaaclab_newton:$W/source/isaaclab_assets:$W/source/isaaclab_rl:$S:$P
T=$1
[ -f "$P/$T" ] || T=$S/$1
cd "$P" && exec "${PYTHON:-$W/.venv/bin/python}" "$T" "${@:2}"
