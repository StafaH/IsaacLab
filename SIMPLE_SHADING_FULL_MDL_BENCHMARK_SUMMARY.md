# Simple Shading Full MDL benchmark summary

These results reproduce the `SSFM` view from the database's **Isaac Lab v3 — All Tasks FPS Overview** using the complete August 26, 2026 run for Isaac Lab commit `079ec904662750a9ef44dbcec7ef80e32dcdd25a`. The Newton Renderer columns use the overview's plain `PHYSX_WARP` and `NEWTON_WARP` rows because Newton Renderer supports RGB as its only camera output and therefore has no separate SSFM group.

- Metric: Mean Environment Step FPS
- Workload: Warm non-RL camera collection
- Environments: 8,192
- Output: Simple Shading Full MDL for Isaac Sim RTX and OV RTX; RGB for Newton Renderer, 64×64
- Hardware: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition

## Environment step FPS and backend coverage

| Camera task | PhysX + Isaac Sim RTX | PhysX + Newton Renderer | Newton MJWarp + Isaac Sim RTX | Newton MJWarp + Newton Renderer | Newton MJWarp + OV RTX | OV PhysX + OV RTX |
|---|---:|---:|---:|---:|---:|---:|
| Isaac-Cartpole-Camera | 12.46 | 44.33 | 13.97 | **59.97** | **21.09** | 19.21 |
| Isaac-Cartpole-Camera-Direct | 15.35 | 86.49 | 17.56 | **164.60** | **30.87** | 26.98 |
| Isaac-Lift-KukaAllegro-Camera | 1.73 | 8.20 | 2.05 | **20.46** | **12.84** | 3.31 |
| Isaac-Reorient-Cube-Shadow-Camera | 3.54 | 4.91 | 3.57 | **6.16** | **5.03** | 4.24 |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 5.33 | 10.90 | 5.15 | **16.07** | **9.45** | 7.08 |
| Isaac-Reorient-KukaAllegro-Camera | 1.74 | 8.02 | 2.09 | **19.43** | **12.90** | 3.39 |

## Primary Newton Renderer comparison

This comparison holds Newton MJWarp physics constant and measures Newton Renderer relative to OV RTX. Per the overview convention, the Newton Renderer value comes from the plain `NEWTON_WARP` RGB row while the OV RTX value comes from its SSFM row:

\[
\text{Newton Renderer uplift} = \left(\frac{\text{Newton + Newton Renderer}}{\text{Newton + OV RTX}} - 1\right) \times 100
\]

| Camera task | Newton + Newton Renderer | Newton + OV RTX | Newton Renderer uplift |
|---|---:|---:|---:|
| Isaac-Cartpole-Camera | 59.97 | 21.09 | **+184.33%** |
| Isaac-Cartpole-Camera-Direct | 164.60 | 30.87 | **+433.21%** |
| Isaac-Lift-KukaAllegro-Camera | 20.46 | 12.84 | **+59.40%** |
| Isaac-Reorient-Cube-Shadow-Camera | 6.16 | 5.03 | **+22.42%** |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 16.07 | 9.45 | **+70.00%** |
| Isaac-Reorient-KukaAllegro-Camera | 19.43 | 12.90 | **+50.58%** |

## Available same-physics renderer comparison

The available comparison holds Newton MJWarp physics constant and measures OV RTX relative to Isaac Sim RTX:

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

- [Combined presentation figure](camera_benchmark_figures/simple-shading-full-mdl-overview.png)
- [Editable SVG](camera_benchmark_figures/simple-shading-full-mdl-overview.svg)
- [Exact database export](camera_benchmark_figures/simple-shading-full-mdl-data.csv)

Separate task figures:

- [Isaac-Cartpole-Camera](camera_benchmark_figures/simple-shading-full-mdl-isaac-cartpole-camera.png) ([SVG](camera_benchmark_figures/simple-shading-full-mdl-isaac-cartpole-camera.svg))
- [Isaac-Cartpole-Camera-Direct](camera_benchmark_figures/simple-shading-full-mdl-isaac-cartpole-camera-direct.png) ([SVG](camera_benchmark_figures/simple-shading-full-mdl-isaac-cartpole-camera-direct.svg))
- [Isaac-Lift-KukaAllegro-Camera](camera_benchmark_figures/simple-shading-full-mdl-isaac-lift-kukaallegro-camera.png) ([SVG](camera_benchmark_figures/simple-shading-full-mdl-isaac-lift-kukaallegro-camera.svg))
- [Isaac-Reorient-Cube-Shadow-Camera](camera_benchmark_figures/simple-shading-full-mdl-isaac-reorient-cube-shadow-camera.png) ([SVG](camera_benchmark_figures/simple-shading-full-mdl-isaac-reorient-cube-shadow-camera.svg))
- [Isaac-Reorient-Cube-Shadow-Camera-Direct](camera_benchmark_figures/simple-shading-full-mdl-isaac-reorient-cube-shadow-camera-direct.png) ([SVG](camera_benchmark_figures/simple-shading-full-mdl-isaac-reorient-cube-shadow-camera-direct.svg))
- [Isaac-Reorient-KukaAllegro-Camera](camera_benchmark_figures/simple-shading-full-mdl-isaac-reorient-kukaallegro-camera.png) ([SVG](camera_benchmark_figures/simple-shading-full-mdl-isaac-reorient-kukaallegro-camera.svg))

The CSV also includes total FPS, database record IDs, CI job IDs, timestamps, and the full source commit for every plotted value.

## Benchmark system

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GB
- CPU: Intel Xeon Gold 5512U, 56 physical cores
- RAM: 251.36 GB
- NVIDIA driver: 595.58.03
- CUDA: 12.8
- Isaac Sim: `6.1.0-alpha.66+develop.48049.25ea2292.gl.manylinux_2_35_x86_64.release`
- Machine profile: `XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`
