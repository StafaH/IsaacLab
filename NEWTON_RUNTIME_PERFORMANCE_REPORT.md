# Newton Runtime Performance Investigation

Date: 2026-08-15  
Baseline revision: `0caae64dc7c08b3fec9e748b165e4ecb211194cc` (`develop`, clean)  
Final state: uncommitted working-tree changes

## Executive summary

Eight root-cause changes were retained:

1. Newton reconciles MJWarp solver reset state once per simulation step instead of twice, and clean CUDA-graph steps now skip reset/FK reconciliation entirely through a device-authored predicate.
2. Newton renderer scene state is refreshed once through the render-context boundary instead of again inside `render()`.
3. Always-updating cameras avoid reading a known-full environment mask back from the GPU.
4. Built-in velocity commands update heading/standing masks without device `nonzero()` synchronization.
5. Newton actuator setup initializes velocity-limit metadata and resolves remote neural checkpoints through the canonical asset cache.
6. The G1 Newton MJWarp preset uses the validated all-implicit Newton actuator/full-decimation path without changing its PhysX presets.
7. Termination bookkeeping reuses the environment's required reset selection instead of compacting the same device mask twice.
8. Resolved scene selectors, lift progress state, and common joint reductions remain device-resident and fixed-shape instead of repeatedly staging Python indices or producing dynamically shaped tensors.

The exact final tree delivers significant gains across every benchmark: G1 locomotion improves **16.28%**, AnymalD
locomotion **6.28%**, non-camera Kuka manipulation **12.55%**, RGB camera manipulation **17.62%**, and depth camera
manipulation **17.39%**. G1 moves from 70,316.46 to 81,762.51 FPS at 1,024 environments while preserving the validated
absolute-action trajectory. RGB camera Kuka moves from 1,656.45 to 1,948.30 FPS at 16 environments. All final points are
three-run means from the same post-compatibility tree; their coefficients of variation range from 0.076% to 2.299%.

Profiling showed that synchronization and launch overhead, rather than copy volume or GPU computation, dominated the
remaining cost. The synchronization campaign removed all 19 selector uploads from G1 and reduced Kuka from eight stream
synchronizations per step to two. On Kuka, termination and reward managers now issue no steady synchronization or
device-to-host copies; the only long wait is the single compact reset selection required by the current manager API.
The retained synchronization campaign improves G1 by 8.4% across its two staged comparisons, AnymalD by 6.3%, Kuka by
12.6%, and RGB camera Kuka by 16.6% relative to their phase-A controls.

Several tempting shortcuts were rejected: native forward depth removed post-passes but was 0.28% slower; MJWarp
`update_data_interval=0` was 12-18% slower; replacing termination compaction with a dense write merely relocated the
wait; and globally enabling full decimation failed Kuka relative-action and Anymal neural-actuator trajectory gates.
The retained termination design instead reuses the already-required reset IDs. The G1 actuator default remains
deliberately backend- and task-scoped.

## Scope and workflow

The investigation used three parallel workstreams:

- A benchmark owner serialized all GPU measurements to prevent cross-run contention.
- An MJWarp workstream traced solver, reset, force, state-sync, and asset-update hot paths.
- A renderer workstream traced camera scheduling, scene refresh, render graphs, depth paths, and host/device transfers.

Each retained change was benchmarked as a separate phase. Camera phase B was temporarily removed and restored to obtain
an A-only camera measurement. The depth experiment was compared against an A+B+C-only detached source tree with module
provenance recorded before every run. The detached tree was removed after measurement.

The second pass used correctness gates before every long actuator benchmark. Reset-heavy and fixed-action smokes logged
resolved config flags, decimation ownership, soft velocity limits, terminations, rewards, targets, and state finiteness.
This caught three independent problems before they could be mistaken for performance wins: missing fast-path velocity-
limit metadata, relative-action decimation semantics, and unresolved remote neural checkpoints. Long runs proceeded only
for G1 after its absolute-action trajectory matched the default path.

## Benchmark configuration

The requested `uv run benchmark runtime` executable is not installed in this repository. The equivalent supported CLI is
`uv run isaaclab benchmark runtime`, which was used for every result.

Hardware and software:

| Component | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB, 600 W |
| Driver | 580.159.03 |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads |
| RAM | 62.30 GiB |
| Newton | 1.5.0 |
| MuJoCo Warp | 3.11.0 |
| Warp | 1.16.0 |
| PyTorch | 2.11.0+cu128 |

Common command template:

```bash
uv run isaaclab benchmark runtime \
  --task TASK \
  --num_envs NUM_ENVS \
  --warmup_steps 100 \
  --num_steps NUM_STEPS \
  --seed 42 \
  --visualizer none \
  --benchmark_formatter schema,summary \
  --output_path OUTPUT_PATH \
  physics=newton_mjwarp [renderer=newton_renderer] [presets=...]
```

Short 10-20-step warmups exposed 193-303 ms transient steps. Final results use 100 warmup steps, three independent
runs, and longer measurement windows:

| Workload | Role | Environments | Measured steps | Presets |
|---|---|---:|---:|---|
| `Isaac-Velocity-Flat-AnymalD` | Locomotion, non-camera | 1,024 | 1,000 | `physics=newton_mjwarp` |
| `Isaac-Lift-KukaAllegro` | Manipulation, non-camera | 256 | 1,000 | `physics=newton_mjwarp` |
| `Isaac-Lift-KukaAllegro-Camera` | Manipulation, RGB camera | 16 | 1,000 | `rgb64,single_camera` |
| `Isaac-Lift-KukaAllegro-Camera` | Manipulation, depth camera | 16 | 500 | `depth64,single_camera` |
| `Isaac-Velocity-Flat-G1` | Locomotion, non-camera | 1,024 | 1,000 | `physics=newton_mjwarp` |

FPS is the benchmark's aggregate `Mean Total FPS`, not per-environment control frequency.
The runtime benchmark's default measurement mode is `host_return`: it measures `env.step()` with a host monotonic clock
and does not insert a CUDA/Warp synchronization around the step. This is the appropriate primary measure for an
asynchronous pipeline. `--measure_sync_step` is useful diagnostically, but it synchronizes before and after both the
environment step and every simulation step, serializing the overlap being optimized. All retained comparisons therefore
use the default mode and pair long repeated runs with Nsight traces that locate the eventual queue drain.

## End-to-end results

| Workload | Clean mean FPS | Retained mean FPS | Clean CV | Retained CV | Change |
|---|---:|---:|---:|---:|---:|
| AnymalD, 1,024 environments | 117,262.95 | 124,622.56 | 2.027% | 2.299% | **+6.276%** |
| KukaAllegro, 256 environments | 21,474.98 | 24,170.84 | 0.526% | 0.782% | **+12.553%** |
| KukaAllegro RGB64, 16 environments | 1,656.45 | 1,948.30 | 0.344% | 0.251% | **+17.619%** |
| KukaAllegro depth64, 16 environments | 1,662.54 | 1,951.62 | 0.023% | 0.076% | **+17.388%** |
| G1, 1,024 environments | 70,316.46 | 81,762.51 | 0.559% | 0.869% | **+16.278%** |

The final G1 runs were 82,336.57, 80,967.62, and 81,983.34 FPS. Anymal's isolated velocity-mask comparison remains
approximately +0.5% after excluding a 311 ms control outlier; its larger final gain comes from eliminating repeated
contact-selector staging. The post-compatibility selector guards changed every pre-compatibility mean by less than 1%,
confirming that preserving runtime selector mutation has no material throughput cost.

### Isolated camera phases

| Phase | Change | Mean FPS | CV | Incremental change |
|---|---|---:|---:|---:|
| Clean | Baseline | 1,656.45 | 0.344% | - |
| A | Remove duplicate solver reconciliation | 1,664.17 | 0.286% | +0.466% |
| A+B | Refresh renderer scene state once | 1,678.87 | 0.088% | **+0.883%** |
| A+B+C | Avoid known-full camera mask readback | 1,689.38 | 0.432% | **+0.626%** |

Depth64 was also measured with A+B+C in isolation: 1,681.68 FPS, 0.107% CV, or **+1.151%** over clean.

## Retained changes

### A. Consume MJWarp reset masks once per step

`NewtonManager.step()` called `_reset_solver_internals_delegate()` and then always reached `forward()`, which called the
same delegate again before FK and mask clearing. On GPU, MJWarp's reset launches across
`(world_count, max(nv, nu, na, nbody))` even when every world mask entry is false. At 4,096 worlds and a persistent-state
width of 29, each duplicate dispatch scans roughly 119,000 threads.

The eager call in `step()` was removed. The single `forward()` boundary now owns reset/FK reconciliation and preserves
both graphable and non-graphable execution ordering. A regression test verifies one reset/FK sequence per step; the
existing selected-world MJWarp solver-reset test verifies that reset semantics are preserved.

Measured effect:

- KukaAllegro non-camera: **+0.990%**.
- KukaAllegro RGB camera: **+0.466%**.
- AnymalD locomotion: no statistically distinguishable change.

### B. Refresh Newton renderer state once

`NewtonWarpRenderer.update_transforms()` already calls the physics-manager forward/state refresh through the render
context. `NewtonWarpRenderer.render()` immediately called `NewtonManager.get_state()` again, repeating reconciliation and
visualization-state work. Under Newton/MJWarp this repeated reset/FK-related launches; under the PhysX shadow path it
repeated scene-data synchronization.

The renderer-side `get_state()` call was removed so the render-context update boundary is the sole scene refresh.
Newton/Newton-renderer and PhysX/Newton-renderer golden rendering tests confirm the state is current for RGB and all depth
conventions.

Measured camera effect: **+0.883% incremental** over phase A.

### C. Avoid the known-full camera mask device readback

Camera buffer refresh previously called `env_mask.numpy()` before every render to determine whether any view was due.
On CUDA this creates a device-to-host synchronization. The renderer always renders the complete tiled camera batch, so
the readback cannot reduce render work when the complete batch is already known to be stale.

`SensorBase` now tracks a conservative host-known `_all_envs_outdated` hint for initialization, full resets, and sensors
with `update_period <= 0`. `Camera` skips the device mask check only when that hint is true. Partial resets and periodic
sensor semantics retain the existing device-mask path, and a regression test covers full update, partial reset, and full
reset transitions. This is safe under captured device-side invalidation: an unknown state falls back to the existing
readback instead of incorrectly skipping an update.

Measured camera effect: **+0.626% incremental** over phases A+B. A post-change trace confirms no stream synchronization
inside the camera/render range.

### D. Remove dynamic-index synchronization from velocity commands

`UniformVelocityCommand` used CUDA `nonzero()` first to gather heading environments and again to gather standing
environments on every control step. `NormalVelocityCommand` did the same for standing and three component masks. These
operations determine dynamic output sizes on the host and synchronize even though the result is used only for masked
device assignment.

Full-shape `torch.where` and `masked_fill_` operations now preserve the exact heading/standing precedence without host
indices. Seeded reference tests cover heading-enabled/disabled, overlapping standing/heading masks, Normal component
masks, and mask immutability. Alternating AnymalD A/B runs measured +0.472% and +0.527% in the two outlier-free pairs.

### E. Make the Newton actuator data contract independent of the hot loop

The Newton actuator path bypasses `_apply_actuator_model()`. That method had been the only writer of
`soft_joint_vel_limits`, leaving the public buffer zero on the fast path. Kuka's standard velocity-limit termination
therefore ended every environment on its first nonzero velocity. Actuator construction now initializes the metadata once,
which is both the correct lifecycle and removes dependence on per-step computation. Implicit-only Lab/Newton trajectory,
torque, metadata, decimation=4, two-substep, interval=2, and CUDA-graph tests pass (10 focused tests).

Neural-actuator schema authoring also passed remote URLs directly to PyTorch, unlike the normal actuator loader. It now
uses the canonical `read_file()` local/HTTP/Omniverse resolver and cache, rewinding the stream for legacy dict fallback.
Six format/path tests and the real cached ANYdrive checkpoint passed. This unlocked Anymal investigation, although its
Newton neural-actuator dynamics did not pass the trajectory gate and were not enabled.

### F. Scope full-decimation acceleration to validated G1 Newton physics

G1 uses only implicit actuators and an absolute `JointPositionAction`; its target is constant throughout the four physics
ticks in one control step. Enabling the Newton actuator path lets `NewtonManager` capture the entire decimation loop and
replace four environment-owned write/step/update cycles with one graph-owned step. A new
`NewtonCfg.use_newton_actuators` setting allows the Newton preset to opt in without affecting sibling PhysX presets or
the existing simulation-wide flag.

Correctness smoke at 16 environments produced identical targets and positions, a maximum peak-velocity difference of
1.10e-5, finite state, equal 20 rad/s soft limits, and no terminations. At 1,024 environments:

| G1 variant | Mean FPS | CV | Change |
|---|---:|---:|---:|
| Previous default | 70,316.46 | 0.559% | - |
| Explicit fast-path experiment | 75,638.54 | 2.478% | +7.569% |
| Retained backend-scoped default | 76,444.58 | 0.177% | **+8.715%** |
| Pre-section-H tree with device predicate | 77,756.93 | 0.625% | **+10.581%** |

### G. Skip clean MJWarp reconciliation with a device-authored predicate

After removing the duplicate reset call, every clean CUDA-graph step still launched a solver reset, masked FK, and two
mask clears. A host boolean cannot safely guard these launches because captured asset-write graphs can invalidate state
without re-entering Python.

`NewtonManager` now owns a one-element device dirty flag written by the same dense and sparse invalidation kernels that
set its world/FK masks. Standard and relaxed simulation graphs contain a conditional reconciliation node at their head;
the public `forward()` boundary uses a separately captured conditional graph. A dirty replay resets only selected solver
state, evaluates masked FK, and clears all masks and the predicate. A clean replay performs none of those kernels. Eager,
non-MJWarp, and conditional-graph-unavailable paths preserve the previous unconditional behavior.

Alternating Kuka A/B runs measured 21,749.36 → 21,974.86 FPS, **+1.037%**, with every pair positive
(+0.784%, +1.654%, +0.678%). CPU captured-writer replay, CUDA selected-world warm-start preservation, standard capture,
actual relaxed capture, and a live Newton-renderer camera smoke all passed.

Pre-section-H exact-tree repeats also moved G1 from 76,444.58 to 77,756.93 FPS (+1.717% over its pre-predicate retained point)
and RGB camera Kuka from 1,689.38 to 1,713.93 FPS (+1.453%). Those RGB runs were 1,716.26, 1,712.36, and 1,713.16
FPS. These cross-session comparisons agree with the isolated alternating Kuka result but are reported separately rather
than treated as additional isolated A/B measurements.

### H. Remove redundant manager and task synchronization

The final synchronization pass followed every steady CUDA stream wait through consecutive environment steps. The large
wait was not caused by meaningful transfer volume: a 4-byte dynamic-index result was forcing the host to wait for the
previously queued physics graph. Four related fixes keep the work fixed-shape and device-resident until the one selection
the environment genuinely needs:

1. `TerminationManager.compute()` no longer compacts per-term dones for private episode bookkeeping. The RL loop records
   them with the exact `reset_buf.nonzero()` IDs it must already obtain for auto-reset. Initial, explicit, and manual reset
   semantics remain unchanged.
2. `SceneEntityCfg` materializes resolved list-based body selectors once on the scene device. Public list/slice fields and
   serialization remain unchanged, while full slices keep view semantics.
3. Lift progress terms seed and update their persistent best-error state with full-shape `torch.where` operations instead
   of boolean advanced indexing.
4. Four common joint-reduction rewards use a cached boolean selection mask. Duplicate selectors, including negative-index
   aliases, fall back to the original advanced-index path so multiplicity is preserved; unselected NaNs remain excluded.

Compatibility guards preserve the existing mutable configuration and manager lifecycles. If public `body_ids` or
`joint_ids` fields change after resolution, the properties detect the mismatch and fall back to the live selector rather
than using a stale cache. Stable and experimental environment loops record termination causes before reset-time
consumers, while direct `TerminationManager.compute()/reset()` callers retain automatic recording through the public
reset path. The managed hot loop disables only that duplicate fallback compaction.

The sequence was benchmarked incrementally rather than attributed to a single edit:

| Workload and phase | Control mean FPS | Candidate mean FPS | Control CV | Candidate CV | Change |
|---|---:|---:|---:|---:|---:|
| AnymalD, cached body selectors | 116,387.14 | 123,669.42 | 2.060% | 2.572% | **+6.257%** |
| G1, termination/body/progress phase | 75,818.59 | 78,146.84 | 0.498% | 0.288% | **+3.071%** |
| G1, fixed-shape joint reductions | 77,658.27 | 82,216.76 | 0.193% | 0.059% | **+5.870%** |
| Kuka, termination/body/progress phase | 21,595.74 | 24,307.47 | 0.375% | 0.534% | **+12.557%** |
| Kuka RGB64, termination/body/progress phase | 1,658.11 | 1,933.10 | 1.281% | 0.166% | **+16.585%** |

Every paired repeat was positive for all five comparisons. A G1 trace after the joint-mask phase contains zero of the 19
original per-step selector host-to-device copies. Its only remaining selector-like upload is an unrelated 4-byte command
manager flag. A Kuka confirmation trace fell from eight to two stream synchronizations per step: the required reset-row
compaction averaged 2.406 ms and the command scheduler averaged 0.0038 ms. `TerminationManager` and `RewardManager`
issued zero steady synchronizations and zero device-to-host copies. The first long drain has therefore reached the
current environment reset boundary as intended.

## Profiling findings

Clean Nsight Systems traces used one representative steady environment step:

| Metric | Kuka non-camera, 256 envs | Kuka RGB64, 16 envs |
|---|---:|---:|
| Environment-step host range | 12.951 ms | 11.629 ms |
| CUDA API time | 6.632 ms | 4.658 ms |
| Stream synchronizations | 8 / 4.640 ms | 8 / 2.500 ms |
| Synchronization share of API time | **70.0%** | **53.7%** |
| GPU kernels | 527 / 0.506 ms | 545 / 0.467 ms |
| GPU memory operations | 0.074 ms | 0.059 ms |
| Device-to-host copies | 7 / <0.001 MB | 8 / <0.001 MB |

The key signal is synchronization latency, not transfer volume. The camera trace broke its buffer refresh down to
0.411 ms, rendering into the camera to 0.279 ms, Newton render to 0.145 ms, and transform refresh to 0.128 ms.

In a pre-section-H depth trace, the camera path itself issued no stream synchronization. One 2.266 ms synchronization was
attributed to `TerminationManager.compute`; because it drains prior asynchronous work, the issuer is not necessarily the
producer. This should be traced across consecutive steps before moving synchronization based only on the attributed call
site.

The full synchronization campaign subsequently traced the last 15 steady steps of each exact-tree workload:

| Workload | Stream syncs / step | Time in stream syncs | Environment step | First long-wait issuer before the final fixes |
|---|---:|---:|---:|---|
| G1, 1,024 environments | 23 | 8.215 ms | 12.018 ms | `illegal_contact` Python-list selector |
| Kuka, 256 environments | 8 | 4.836 ms | 12.800 ms | termination bookkeeping `nonzero()` |
| Kuka RGB64, 16 environments | 8 | 2.494 ms | 11.410 ms | termination bookkeeping `nonzero()` |
| AnymalD, 1,024 environments | 8 | 2.040 ms | 8.693 ms | `illegal_contact` Python-list selector |

There was no steady `cudaDeviceSynchronize` or `cuCtxSynchronize`. On G1, one body selector and 18 reward selectors
created 19 tiny host-to-device transfers; on Kuka, a single 4-byte device-to-host copy attributed to termination
bookkeeping drained 4.811 ms of queued work. These facts are why optimizing transfer bytes or blindly moving the
attributed wait would miss the root problem. The final Kuka trace confirms that the retained chain removes both manager
issuers and leaves the reset selection as the first and only material barrier.

The retained G1 full-decimation path reduced a representative steady environment step from 14.302 ms to 12.830 ms
(-10.29%). It collapsed four simulation steps, scene writes/updates, action applications, and graph launches to one.
Kernel time fell from 659.7 to 304.3 us (-53.9%), kernel instances from 559 to 284 (-49.2%), and `cudaLaunchKernel` calls
from 482 to 258. Transfer counts were unchanged. Synchronization attribution crossed asynchronous queue boundaries, so
the higher sync time attributed to the faster trace is not interpreted as additional work.

## Rejected experiment: native forward depth

Newton 1.5 can write planar depth directly from its ray-tracing megakernel. An implementation bound Isaac Lab's `depth`
output to that native slot and used render clear values for max-depth clipping, removing the separate projection and
background-replacement kernels. Golden output tests passed, and Nsight confirmed both post-passes disappeared.

The end-to-end result did not improve:

| Depth64 variant | Mean FPS | CV | Difference |
|---|---:|---:|---:|
| Retained A+B+C | 1,681.68 | 0.107% | - |
| Native forward depth | 1,676.89 | 0.670% | -0.285% |

The experiment was reverted. The likely mechanism is visible in Newton's render kernel: every hit pixel performs a
camera-forward transform, normalization, and dot product inside an already register-heavy ray-tracing megakernel (85
registers per thread in the sampled launch). The old bandwidth-oriented post-pass did not add those live values to the
ray tracer.

The root fix belongs upstream: for normalized camera-space rays and a rigid camera transform, forward projection can be
computed from the ray's camera-space forward component, for example
`hit_distance * (-camera_ray_direction.z)`. Precomputing that scalar per ray would remove transform/normalize/dot work
without restoring the image post-pass. Native forward depth should be benchmarked again after that change.

## Other investigated experiments

### MJWarp data-update interval

`update_data_interval=0` passed finite/reset smoke but materially reduced throughput: AnymalD fell 18.46% and Kuka fell
12.12%. Kuka interval 1 was 0.61% slower than its maintained interval 2. More importantly, MJWarp selects periodic updates
with a Python step counter; CUDA graph capture freezes that branch. A one-step graph can therefore replay “always” or
“never” instead of the configured cadence. The root fix is an explicit device-authored state-dirty/sync contract, not an
interval tweak.

### Dense termination bookkeeping update

An early prototype replaced `TerminationManager`'s `nonzero()` with a full-shape `where`. It removed that dynamic output,
but added a dense `(environment, term)` write and merely moved the queue drain into rewards. Its noisy end-to-end result
did not justify the work. That implementation was reverted. The retained design in section H is materially different:
it performs no replacement kernel or selection and reuses the compact reset IDs already required by the RL loop.

### Actuator fast-path correctness gates

- Kuka reset behavior matched after the soft-limit fix, but its relative action is recomputed from current position on
  every ordinary physics tick and only once when decimation is manager-owned. After 20 constant-action steps, mean joint
  position differed by 0.0219 rad and peak velocity by 0.584 rad/s. It was not enabled.
- Anymal's neural checkpoint resolved after the asset-reader fix, but Newton neural-actuator dynamics differed by
  0.1235 rad mean position and 7.787 rad/s peak velocity after 20 steps; the adapter was also not fully graphable. It was
  not enabled.
- G1 uses absolute actions and all-implicit drives, passed parity, and was retained.

### Low-return launch/copy candidates

- Eliding `torch.cat` for one-term observation groups was prototyped with lifetime, history, contiguity, and alias tests.
  Nsight bounded the saving to about 0.037% per 16x64 RGB group, so the extra storage/autograd special case was rejected.
- Native camera sensor graph caching would save a measured 8-10 us flag upload and 17-24 us graph launch per query, but
  less than 0.1% of the sampled single-camera step. It remains useful for many-camera scaling, not this matrix.
- Compatible multi-camera batching would save launches but not rays; its estimated current Kuka effect is below 0.3%
  and it requires a cross-camera backing-store coordinator.
- Acceleration kernels cost 16 launches per Kuka control step before full decimation, but G1's full-decimation path already
  reduces analogous asset updates fourfold. A model-wide redesign remains primarily a correctness/lifecycle cleanup.

## Recommended next work

### 1. Make reset lifecycles mask-native

The final traces have moved the only long steady wait to `reset_buf.nonzero()`. The current manager contracts require
compact host-known environment IDs for scene, action, observation, reward, curriculum, command, event, termination,
recorder, and Same-Step final-observation handling. Removing that barrier correctly requires capability-based mask-native
reset methods and a device-predicated reset graph, while retaining compact-ID fallbacks for host-only recorders and user
extensions. Random sampling must use per-environment counter/stateless streams before replacing `rand(len(env_ids))`, or
the optimization will silently change seeded trajectories and RNG consumption. Preserve partial/manual resets, reset
ordering, final observations, recorder hooks, and captured asset writers.

An asynchronous pinned-host eligibility prototype removed 50/50 command and interval-event `nonzero()` calls in a
microprofile and passed 17 parity tests. It was not retained: after the required reset drain each saved wait is only about
3-4 us, while the design adds host/device event state and becomes fragile if users mutate public timer tensors directly.
It should be reconsidered only as part of the mask-native reset/RNG contract, not as a standalone scheduler cache.

### 2. Native masked camera/world scheduling

Phase C removes the readback for always-updating/full-reset cases but deliberately preserves per-environment partial and
periodic semantics. Extending Newton's tiled camera with a `(camera, world)` enable mask would let the render megakernel
skip non-due tiles without redefining timestamps or frame counts. A device predicate can then skip the whole graph when
no view is due. Test staggered partial resets and update periods before removing the fallback mask path.

### 3. Remove per-query sensor graph flag uploads

`NewtonManager._update_sensor_tasks()` builds a NumPy flag array, uploads it to the GPU, and launches a conditional graph
for each query. Cache direct graphs by stable requested-task tuple and refit state, with a bounded cache for arbitrary
subsets. This removes the pageable host-to-device flag upload and predicate dispatch without introducing a host readback.

### 4. Consolidate eager acceleration updates

Articulations, rigid objects, and collections independently finite-difference joint/body accelerations on every scene
update, even when no observation or reward reads them. These full-state scans sit outside the solver graph and scale with
asset count. Prefer a manager-owned model-wide captured post-step update with mapped asset views, or an explicit interest
registration lifecycle that still advances history correctly before the first later read.

### 5. Consolidate external-force assembly

Newton allocates `body_f` for rigid-body states, so MJWarp applies body forces and Isaac Lab clears the full force buffer
every substep even when no external-wrench composer is active. Upstream MJWarp currently launches body-force and free-
joint-force kernels over the same `(world, body)` domain. Fuse them into one xfrc assembly kernel, then add a graph-safe
force-activity scalar so Lab clears force staging only after an actual writer. Do not set `body_f=None`; that would break
bindings and first use after graph capture.

### 6. Replace periodic Newton-to-MJWarp synchronization

`MJWarpSolverCfg.update_data_interval` controls a full `(world_count, joints_per_world)` coordinate conversion, but its
Python modulo branch is frozen into CUDA graphs. Add an explicit upstream `sync_state(state, world_mask)` operation and
drive it from the same device-authored reconciliation flag as resets/FK. Only after resets, randomization, captured direct
writes, contacts, sensors, and trajectories pass should periodic conversion be removed.

### 7. Fix moving-camera ownership, then batch compatible instances

The Kuka wrist camera is attached to `palm_link`, but explicit-pose renderers only refresh it when public pose metadata is
requested; under Newton/OVRTX it can therefore remain stale. Split renderer pose freshness from the
`update_latest_camera_pose` metadata option and pass native OpenGL poses directly, eliminating the OpenGL→world→OpenGL
kernel chain. Add an observable attached-camera image-motion test first.

Compatible camera sensors then can group render data and launch fewer megakernels. Group cameras with
matching resolution, outputs, clipping, render configuration, and cadence into shared `(world, camera, H, W, C)` backing
stores, then expose zero-copy per-camera views. This is especially relevant to Kuka's compatible base and wrist cameras.

### 8. Lower-priority measured-path candidates

- Fuse or alias repeated articulation target copies after profiling identity-order and reorder paths.
- Move global actuator adapter reset out of per-articulation reset and retain scratch masks.
- Fuse moving-camera convention/transform kernels and eliminate orientation round trips.
- Fuse camera observation conversion from BHWC uint8 to normalized BCHW float where task preprocessing is material.
- Profile camera reset-buffer prefill separately; it grows noticeably with environment count but affects startup, not
  steady-state runtime.

## Validation

The final retained tree passed:

- 199 sensor-base, renderer-contract, render-context, and Newton-manager abstraction tests.
- 163 focused Newton manager and ordering-kernel tests after integrating device-predicated reconciliation.
- The selected-world MJWarp solver-internal reset integration test on CUDA.
- The captured clean/selected-world reconciliation integration test on CUDA, plus a live Newton-renderer camera benchmark
  smoke exercising relaxed graph capture.
- Eight Kuka golden rendering combinations covering Newton and PhysX physics, Newton renderer, RGB, planar depth, ray
  distance, and image-plane distance.
- 10 implicit-actuator equivalence and full-decimation tests, including velocity metadata, torque telemetry,
  decimation=4, two substeps, interval=2, and CUDA graphs.
- Eleven velocity-command and local/HTTP/Omniverse checkpoint-resolution unit tests.
- Termination-recording lifecycle, cached-selector mutation/serialization, fixed-shape joint-reduction, and lift-progress
  parity tests.
- All 15 post-compatibility benchmark schemas completed with the expected Newton MJWarp/Newton renderer presets, and
  source hashes matched at the start and end of the serial matrix.
- The manager-based Newton actuator authoring integration test and the retained G1 behavior/parity smoke.
- Ruff, Ruff format, whitespace, symlink, YAML, TOML, conflict, executable, private-key, debug-statement, codespell, SPDX,
  RST, and Git LFS checks through `uv run isaaclab -f`.
- Direct changelog fragment validation for all three changed packages.

The aggregate formatter command reports only the changelog-fragment hook as failed because pre-commit ignores untracked
files. The four fragments remain untracked to satisfy the request to leave all work uncommitted; the repository's direct
`uv run python tools/changelog/cli.py check develop` gate passed with the worktree fragments included.

## Artifacts

Raw schemas, summaries, command logs, and traces are outside the repository under:

- `/tmp/isaaclab_newton_bench/final_*`
- `/tmp/isaaclab_newton_bench/phase_a_*`
- `/tmp/isaaclab_newton_bench/phase_b_*`
- `/tmp/isaaclab_newton_bench/phase_c_*`
- `/tmp/isaaclab_newton_bench/phase_d_*`
- `/tmp/isaaclab_newton_bench/nsys/`
- `/tmp/isaaclab_newton_bench/device_dirty_ab/`
- `/tmp/isaaclab_newton_bench/device_dirty_live_camera_smoke/`
- `/tmp/isaaclab_newton_bench/newton_actuators/`
- `/tmp/isaaclab_newton_bench/final_exact_tree/`
- `/tmp/isaaclab_newton_bench/final_integrated_sync/`
- `/tmp/isaaclab_newton_bench/final_post_compatibility/`
- `/tmp/isaaclab_newton_bench/nsys/sync_campaign_exact/`
- `/tmp/isaaclab_newton_bench/sync_candidate_a/`
- `/tmp/isaaclab_newton_bench/sync_candidate_b/`
- `/tmp/isaaclab_newton_bench/sync_candidate_b2/`
- `/tmp/isaaclab_newton_bench/sync_candidate_d/`

The isolated depth source provenance is recorded at
`/tmp/isaaclab_newton_bench/phase_c_camera_depth64_16/import_provenance.log`.
