# Camera rendering benchmark summary

These results use the camera **collection** workload measured on August 24, 2026 with 8,192 environments. Higher FPS is better.

## Key Newton renderer comparison

The primary percentage uses **Newton MJWarp + Newton Renderer as the baseline**:

\[
\text{OV RTX change} = \frac{\text{OV RTX FPS} - \text{Newton Renderer FPS}}{\text{Newton Renderer FPS}} \times 100
\]

Because percentage comparisons are directional, the final column also reports the inverse comparison using Newton MJWarp + OV RTX as the baseline.

| Camera task | Newton MJWarp + Newton Renderer | Newton MJWarp + OV RTX | OV RTX relative to Newton Renderer | Newton Renderer relative to OV RTX |
|---|---:|---:|---:|---:|
| Isaac-Cartpole-Camera | 500,100.07 FPS | 82,483.79 FPS | **83.51% lower** | 506.30% higher |
| Isaac-Cartpole-Camera-Direct | 1,388,058.92 FPS | 137,434.56 FPS | **90.10% lower** | 909.98% higher |
| Isaac-Lift-KukaAllegro-Camera | 162,372.83 FPS | 75,597.52 FPS | **53.44% lower** | 114.79% higher |
| Isaac-Reorient-Cube-Shadow-Camera | 50,514.43 FPS | 29,434.95 FPS | **41.73% lower** | 71.61% higher |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 131,706.89 FPS | 62,080.29 FPS | **52.86% lower** | 112.16% higher |
| Isaac-Reorient-KukaAllegro-Camera | 160,231.39 FPS | 76,390.97 FPS | **52.32% lower** | 109.75% higher |

## Complete backend results

| Camera task | Isaac Sim PhysX + Isaac Sim RTX | Isaac Sim PhysX + Newton Renderer | Newton MJWarp + Isaac Sim RTX | Newton MJWarp + Newton Renderer | Newton MJWarp + OV RTX | OV PhysX + OV RTX |
|---|---:|---:|---:|---:|---:|---:|
| Isaac-Cartpole-Camera | 49,034.67 | 357,241.72 | 51,321.24 | **500,100.07** | **82,483.79** | 80,450.33 |
| Isaac-Cartpole-Camera-Direct | 74,621.20 | 723,674.16 | 80,137.44 | **1,388,058.92** | **137,434.56** | 132,050.32 |
| Isaac-Lift-KukaAllegro-Camera | 12,890.02 | 66,810.80 | 15,000.74 | **162,372.83** | **75,597.52** | 25,248.82 |
| Isaac-Reorient-Cube-Shadow-Camera | 18,377.72 | 40,260.16 | 18,765.25 | **50,514.43** | **29,434.95** | 25,747.01 |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 33,279.39 | 89,752.33 | 33,688.72 | **131,706.89** | **62,080.29** | 48,579.13 |
| Isaac-Reorient-KukaAllegro-Camera | 12,881.41 | 63,412.39 | 14,848.44 | **160,231.39** | **76,390.97** | 24,883.49 |

All values are total mean FPS. The charts use a logarithmic horizontal scale so every backend combination remains legible despite the large throughput range.

## Presentation figures

- [Isaac-Cartpole-Camera](camera_benchmark_figures/isaac-cartpole-camera.png) ([editable SVG](camera_benchmark_figures/isaac-cartpole-camera.svg))
- [Isaac-Cartpole-Camera-Direct](camera_benchmark_figures/isaac-cartpole-camera-direct.png) ([editable SVG](camera_benchmark_figures/isaac-cartpole-camera-direct.svg))
- [Isaac-Lift-KukaAllegro-Camera](camera_benchmark_figures/isaac-lift-kukaallegro-camera.png) ([editable SVG](camera_benchmark_figures/isaac-lift-kukaallegro-camera.svg))
- [Isaac-Reorient-Cube-Shadow-Camera](camera_benchmark_figures/isaac-reorient-cube-shadow-camera.png) ([editable SVG](camera_benchmark_figures/isaac-reorient-cube-shadow-camera.svg))
- [Isaac-Reorient-Cube-Shadow-Camera-Direct](camera_benchmark_figures/isaac-reorient-cube-shadow-camera-direct.png) ([editable SVG](camera_benchmark_figures/isaac-reorient-cube-shadow-camera-direct.svg))
- [Isaac-Reorient-KukaAllegro-Camera](camera_benchmark_figures/isaac-reorient-kukaallegro-camera.png) ([editable SVG](camera_benchmark_figures/isaac-reorient-kukaallegro-camera.svg))

### Benchmark system

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GB
- CPU: Intel Xeon Gold 5512U, 56 physical cores
- RAM: 251.36 GB
- NVIDIA driver: 595.58.03
- CUDA bindings: 12.9.7
- Source commit: `f86b4e1ad66e637e7770a727e181ffe150a2077b`
