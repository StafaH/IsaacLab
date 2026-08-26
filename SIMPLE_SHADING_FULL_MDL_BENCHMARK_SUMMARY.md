# Simple Shading Full MDL benchmark summary

These results reproduce the `SSFM` filter from the database's **Isaac Lab v3 — All Tasks FPS Overview** using the complete August 26, 2026 run for Isaac Lab commit `079ec904662750a9ef44dbcec7ef80e32dcdd25a`.

- Metric: Mean Environment Step FPS
- Workload: Warm non-RL camera collection
- Environments: 8,192
- Output: Simple Shading Full MDL, 64×64
- Hardware: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition

## Environment step FPS and backend coverage

| Camera task | PhysX + Isaac Sim RTX | PhysX + Newton Renderer | Newton MJWarp + Isaac Sim RTX | Newton MJWarp + Newton Renderer | Newton MJWarp + OV RTX | OV PhysX + OV RTX |
|---|---:|---:|---:|---:|---:|---:|
| Isaac-Cartpole-Camera | 12.46 | N/A | 13.97 | **N/A** | **21.09** | 19.21 |
| Isaac-Cartpole-Camera-Direct | 15.35 | N/A | 17.56 | **N/A** | **30.87** | 26.98 |
| Isaac-Lift-KukaAllegro-Camera | 1.73 | N/A | 2.05 | **N/A** | **12.84** | 3.31 |
| Isaac-Reorient-Cube-Shadow-Camera | 3.54 | N/A | 3.57 | **N/A** | **5.03** | 4.24 |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 5.33 | N/A | 5.15 | **N/A** | **9.45** | 7.08 |
| Isaac-Reorient-KukaAllegro-Camera | 1.74 | N/A | 2.09 | **N/A** | **12.90** | 3.39 |

## Primary Newton Renderer comparison

The database contains no comparable SSFM measurements for either PhysX + Newton Renderer or Newton MJWarp + Newton Renderer on this hardware. The expected `PHYSX_WARP_SSFM` and `NEWTON_WARP_SSFM` comparison groups are absent, so a Newton MJWarp + Newton Renderer versus Newton MJWarp + OV RTX percentage cannot be calculated honestly.

The figures retain both missing combinations as explicit `N/A` rows. Mixing the existing RGB Newton Renderer values with SSFM OV RTX values would compare different camera outputs and is not valid.

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
