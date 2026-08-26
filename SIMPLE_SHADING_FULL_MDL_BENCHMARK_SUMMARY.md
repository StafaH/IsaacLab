# Simple Shading Full MDL benchmark summary

These results reproduce the `SSFM` view from the database's **Isaac Lab v3 — All Tasks FPS Overview** using the complete August 26, 2026 run for Isaac Lab commit `079ec904662750a9ef44dbcec7ef80e32dcdd25a`. The Newton Renderer columns use the overview's plain `PHYSX_WARP` and `NEWTON_WARP` rows because Newton Renderer supports RGB as its only camera output and therefore has no separate SSFM group.

- Metric: Total Mean FPS
- Workload: Warm non-RL camera collection
- Environments: 8,192
- Output: Simple Shading Full MDL for Isaac Sim RTX and OV RTX; RGB for Newton Renderer, 64×64
- Hardware: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition

## Total mean FPS and backend coverage

| Camera task | PhysX + Isaac Sim RTX | PhysX + Newton Renderer | Newton MJWarp + Isaac Sim RTX | Newton MJWarp + Newton Renderer | Newton MJWarp + OV RTX | OV PhysX + OV RTX |
|---|---:|---:|---:|---:|---:|---:|
| Isaac-Cartpole-Camera | 102,094.47 | 363,126.04 | 114,430.50 | **491,267.77** | **172,781.78** | 157,376.84 |
| Isaac-Cartpole-Camera-Direct | 125,742.84 | 708,485.18 | 143,856.05 | **1,348,363.78** | **252,878.31** | 221,056.87 |
| Isaac-Lift-KukaAllegro-Camera | 14,138.08 | 67,166.45 | 16,793.50 | **167,596.04** | **105,144.37** | 27,141.03 |
| Isaac-Reorient-Cube-Shadow-Camera | 29,008.07 | 40,263.00 | 29,245.16 | **50,490.71** | **41,244.13** | 34,732.17 |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 43,639.25 | 89,267.72 | 42,163.83 | **131,617.82** | **77,421.83** | 57,972.04 |
| Isaac-Reorient-KukaAllegro-Camera | 14,244.23 | 65,736.49 | 17,089.60 | **159,176.68** | **105,710.71** | 27,797.46 |

## Primary Newton Renderer comparison

This comparison holds Newton MJWarp physics constant and measures Newton Renderer relative to OV RTX. Per the overview convention, the Newton Renderer value comes from the plain `NEWTON_WARP` RGB row while the OV RTX value comes from its SSFM row:

\[
\text{Newton Renderer uplift} = \left(\frac{\text{Newton + Newton Renderer}}{\text{Newton + OV RTX}} - 1\right) \times 100
\]

| Camera task | Newton + Newton Renderer | Newton + OV RTX | Newton Renderer uplift |
|---|---:|---:|---:|
| Isaac-Cartpole-Camera | 491,267.77 | 172,781.78 | **+184.33%** |
| Isaac-Cartpole-Camera-Direct | 1,348,363.78 | 252,878.31 | **+433.21%** |
| Isaac-Lift-KukaAllegro-Camera | 167,596.04 | 105,144.37 | **+59.40%** |
| Isaac-Reorient-Cube-Shadow-Camera | 50,490.71 | 41,244.13 | **+22.42%** |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 131,617.82 | 77,421.83 | **+70.00%** |
| Isaac-Reorient-KukaAllegro-Camera | 159,176.68 | 105,710.71 | **+50.58%** |

## Available same-physics renderer comparison

The available comparison holds Newton MJWarp physics constant and measures OV RTX relative to Isaac Sim RTX:

\[
\text{OV RTX uplift} = \left(\frac{\text{Newton + OV RTX}}{\text{Newton + Isaac Sim RTX}} - 1\right) \times 100
\]

| Camera task | Newton + Isaac Sim RTX | Newton + OV RTX | OV RTX uplift |
|---|---:|---:|---:|
| Isaac-Cartpole-Camera | 114,430.50 | 172,781.78 | **+50.99%** |
| Isaac-Cartpole-Camera-Direct | 143,856.05 | 252,878.31 | **+75.79%** |
| Isaac-Lift-KukaAllegro-Camera | 16,793.50 | 105,144.37 | **+526.10%** |
| Isaac-Reorient-Cube-Shadow-Camera | 29,245.16 | 41,244.13 | **+41.03%** |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 42,163.83 | 77,421.83 | **+83.62%** |
| Isaac-Reorient-KukaAllegro-Camera | 17,089.60 | 105,710.71 | **+518.57%** |

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
