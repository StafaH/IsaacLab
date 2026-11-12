# Motions of the Unitree G1

The clips are not part of the repository. They are the G1 files of the LAFAN1 retargeting dataset of Unitree
(https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset): one row per frame at 30 fps, with the root
position, the root quaternion (x, y, z, w), and the positions of the 29 joints. Download the ones a task names into
this folder, for example:

```bash
curl -L -o walk1_subject1.csv \
    https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/walk1_subject1.csv
```
