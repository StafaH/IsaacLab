#!/bin/bash
# As run.sh, with the OVRTX renderer for the video. Needs the ovrtx extra of the checkout.
P=$(cd "$(dirname "$0")" && pwd)
W=$(dirname "$P")
S=$W/scripts/reinforcement_learning/mdp2
export DECK_REPO=$W
export PYTHONPATH=$P/render:$W/source/isaaclab:$W/source/isaaclab_experimental:$W/source/isaaclab_tasks:$W/source/isaaclab_newton:$W/source/isaaclab_assets:$W/source/isaaclab_rl:$W/source/isaaclab_ov:$S:$P
T=$P/$1
[ -f "$T" ] || T=$S/$1
cd "$P" && exec uv run --project "$W" --extra ovrtx python "$T" "${@:2}"
