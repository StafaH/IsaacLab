# Clips of the Unitree G1 climbing boxes

The clips are not part of the repository. They are the terrain clips of the OmniRetarget dataset
(https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset, MIT license): motion capture of a person
climbing boxes, retargeted to the 29-DoF G1 on boxes of five heights. Download the dataset, unpack
`robot-terrain.zip` in it, and convert it into this folder:

```bash
uv run python scripts/reinforcement_learning/mdp2/convert_omniretarget.py <dataset> \
    source/isaaclab_tasks/isaaclab_tasks/mdp2/parkour/config/g1/motions
```
