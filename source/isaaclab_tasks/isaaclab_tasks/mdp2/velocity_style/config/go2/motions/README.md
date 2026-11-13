# Motions of the Unitree Go2

The clips are not part of the repository. They are clips of a dog, retargeted to the Unitree A1, from the
`datasets/mocap_motions` folder of https://github.com/escontra/AMP_for_hardware. The A1 has the joints of the Go2
and legs that are 6% shorter, so the clips fit the Go2 with their root positions scaled. Each clip is one cycle of
a gait; the converter repeats it and writes one row per frame at 47.6 fps, with the root position, the root
quaternion (x, y, z, w), and the positions of the twelve joints. Download and convert the ones a task names into
this folder, for example:

```bash
curl -L -o trot0.txt https://raw.githubusercontent.com/escontra/AMP_for_hardware/main/datasets/mocap_motions/trot0.txt
uv run python scripts/reinforcement_learning/mdp2/convert_quadruped_motion.py trot0.txt trot0.csv
```
