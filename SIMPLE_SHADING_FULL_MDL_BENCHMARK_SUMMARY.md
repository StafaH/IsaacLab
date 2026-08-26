# Simple Shading Full MDL benchmark summary

These results reproduce the `SSFM` filter from the database's **Isaac Lab v3 — All Tasks FPS Overview** using the complete August 26, 2026 run for Isaac Lab commit `079ec904662750a9ef44dbcec7ef80e32dcdd25a`.

- Metric: Mean Environment Step FPS
- Workload: Warm non-RL camera collection
- Environments: 8,192
- Output: Simple Shading Full MDL, 64×64
- Hardware: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition

## Environment step FPS

| Camera task | PhysX + Isaac Sim RTX | Newton MJWarp + Isaac Sim RTX | Newton MJWarp + OV RTX | OV PhysX + OV RTX |
|---|---:|---:|---:|---:|
| Isaac-Cartpole-Camera | 12.46 | 13.97 | **21.09** | 19.21 |
| Isaac-Cartpole-Camera-Direct | 15.35 | 17.56 | **30.87** | 26.98 |
| Isaac-Lift-KukaAllegro-Camera | 1.73 | 2.05 | **12.84** | 3.31 |
| Isaac-Reorient-Cube-Shadow-Camera | 3.54 | 3.57 | **5.03** | 4.24 |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 5.33 | 5.15 | **9.45** | 7.08 |
| Isaac-Reorient-KukaAllegro-Camera | 1.74 | 2.09 | **12.90** | 3.39 |

## Newton renderer comparison

This comparison holds Newton MJWarp physics constant and measures OV RTX relative to Isaac Sim RTX:

\[
\text{OV RTX uplift} = \left(\frac{\text{Newton + OV RTX}}{\text{Newton + Isaac Sim RTX}} - 1\right) \times 100
\]

| Camera task | Newton + Isaac Sim RTX | Newton + OV RTX | OV RTX uplift |
|---|---:|---:|---:|
| Isaac-Cartpole-Camera | 13.97 | 21.09 | **+50.99%** |
| Isaac-Cartpole-Camera-Direct | 17.56 | 30.87 | **+75.79%** |
| Isaac-Lift-KukaAllegro-Camera | 2.05 | 12.84 | **+526.10%** |
| Isaac-Reorient-Cube-Shadow-Camera | 3.57 | 5.03 | **+41.03%** |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 5.15 | 9.45 | **+83.62%** |
| Isaac-Reorient-KukaAllegro-Camera | 2.09 | 12.90 | **+518.57%** |

## Presentation assets

- [Presentation figure](camera_benchmark_figures/simple-shading-full-mdl-overview.png)
- [Editable SVG](camera_benchmark_figures/simple-shading-full-mdl-overview.svg)
- [Exact database export](camera_benchmark_figures/simple-shading-full-mdl-data.csv)

The CSV also includes total FPS, database record IDs, CI job IDs, timestamps, and the full source commit for every plotted value.

## Benchmark system

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GB
- CPU: Intel Xeon Gold 5512U, 56 physical cores
- RAM: 251.36 GB
- NVIDIA driver: 595.58.03
- CUDA: 12.8
- Isaac Sim: `6.1.0-alpha.66+develop.48049.25ea2292.gl.manylinux_2_35_x86_64.release`
- Machine profile: `XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`
