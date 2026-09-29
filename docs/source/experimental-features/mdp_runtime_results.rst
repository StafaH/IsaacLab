MDP Runtime Results
===================

Measured results for :doc:`mdp_runtime`: capture boundaries, speed against the stable MDP and the experimental
Warp frontend, and training throughput and learning against ``warp-rl``.

Setup
-----

.. list-table::
   :widths: 30 70

   * - GPU
     - NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB), driver 595.91.07
   * - OS
     - Linux 7.0.0-31-generic x86_64, glibc 2.39
   * - Isaac Lab
     - ``ef2cd3272dec48fb510ac6c089fbce8340d63ba1`` (branch ``feature/experimental-mdp-runtime``)
   * - Dependencies
     - Python 3.12.13, Warp 1.17.0, Torch 2.12.0+cu130, Newton 1.6.0, mujoco-warp 3.12.0
   * - warp-rl
     - sibling checkout ``../warp-rl``, version 0.1.0, branch ``feat/warp-ppo`` with no commits (untracked files).
       SHA-256 prefix of the sorted ``src/**/*.py`` contents: ``de9442a5129ad63b``

All measurements use 4,096 environments on Newton/MJWarp. Every task simulates the scene and physics of its
stable task (``stable_physics_cfgs``), unless stated otherwise.


Speed against the existing MDP systems
--------------------------------------

Environment step time with uniform random actions in [-1, 1], median of 5 samples of 100 steps after 50 warm-up
steps, each system in its own process (``benchmark_mdp_runtime_compare.py``). ``stable`` is
:class:`~isaaclab.envs.ManagerBasedRLEnv` (``gym.make``); ``warp_frontend`` is ``--frontend warp``
(``ManagerBasedRLEnvWarp``, per-manager graphs). The runtime rows replay one CUDA graph per step, or run eagerly.
Final observations are off, as in the stable environment.

.. list-table:: Step time [ms] (speed-up over ``stable``)
   :header-rows: 1

   * - System
     - Cartpole
     - Go2 flat velocity
     - Franka reach
   * - ``stable`` (Torch managers)
     - 2.008 (1.00x)
     - 6.801 (1.00x)
     - 5.039 (1.00x)
   * - ``warp_frontend``
     - 0.816 (2.46x)
     - 6.018 (1.13x)
     - 4.928 (1.02x)
   * - **runtime, Warp, graph**
     - **0.451 (4.46x)**
     - **4.541 (1.50x)**
     - **3.927 (1.28x)**
   * - runtime, Torch, graph
     - 0.648 (3.10x)
     - 6.357 (1.07x)
     - 4.939 (1.02x)
   * - runtime, Warp, eager
     - 5.253
     - 22.789
     - 39.465
   * - runtime, Torch, eager
     - 6.223
     - 31.755
     - 44.982
   * - MDP only (Warp graph, no physics)
     - 0.018
     - 0.053
     - 0.035
   * - physics only, robots at rest
     - 0.430
     - 3.585
     - 2.785

The runtime (Warp graph) is 1.25x to 1.81x faster than the Warp frontend and 1.28x to 4.46x faster than the stable
MDP. The Torch backend, also captured into a graph, runs at the speed of the Warp frontend.

**Speed of light.** The GPU time of the runtime's three generated MDP kernels per step, measured with Warp's CUDA
activity timer over steps of the same run, is 0.012 ms (Cartpole), 0.048 ms (Go2), and 0.026 ms (Reach): 2.7%,
1.1%, and 0.7% of the step. The rest is physics, so no MDP implementation can make these steps more than that
share faster. "Physics only" replays physics with the default joint targets (robots standing still), a lighter
workload than the random-action steps, so it is a reference rather than a ceiling. Eager execution is 5x to 12x
slower than graph replay: Python launch overhead dominates without a graph.

**Faithfulness.** ``validate_mdp_runtime_parity.py`` evaluates the runtime's terms on the stable environment's state
after every step (50 steps, 256 environments, noise off):

* termination flags are identical for all terms;
* observations agree within 3.3e-6 (0 for Cartpole and Reach);
* reward terms agree to float precision, e.g. ``dof_acc_l2`` within 2 of values up to 4.2e7, and all others within
  1e-3 absolute (``dof_torques_l2`` values reach 7e3).

The Go2 runtime task omits the stable task's startup randomization of friction, base mass, and base COM. The Reach
task omits the reward-weight curriculum.


Against warp-rl
---------------

warp-rl PPO: horizon 32, 4 epochs, 4 minibatches, (128, 128) networks, fixed learning rate 3e-4,
``reward_scale = 1 / step_dt``. Probes follow warp-rl's ``speed_of_light.py``: CUDA-event GPU time of graph
replays, FPS = environment transitions per second (``benchmark_mdp_runtime_training.py``).

``runtime_go2_warprl_mdp`` is warp-rl's own Go2 MDP written as a runtime configuration, running on warp-rl's
Isaac-Lab-hosted Go2 scene and physics (1 solver substep, MJWarp contacts). This is the same workload as warp-rl's
hand-written direct environment (``warprl_isaac_go2``); only the implementation differs.

.. list-table:: Throughput [M transitions/s]
   :header-rows: 1

   * - Environment
     - Full training
     - Collection
     - Learning (update only)
     - Env, action tape
     - Env, zero actions
   * - **runtime, warp-rl Go2 MDP**
     - **1.531**
     - **1.709**
     - 15.87
     - **1.733**
     - **2.683**
   * - warp-rl hand-written Isaac-hosted Go2
     - 1.538
     - 1.716
     - 15.62
     - 1.738
     - 2.691
   * - runtime, stable Go2 MDP and scene
     - 0.842
     - 0.892
     - 15.62
     - 0.903
     - 1.098
   * - runtime, Franka reach
     - 0.940
     - 1.062
     - 16.05
     - 0.989
     - 1.374
   * - runtime, Cartpole
     - 5.442
     - 7.980
     - 17.28
     - 9.151
     - 9.213
   * - warp-rl standalone Newton Go2 (``--solver fast``)
     - 2.020
     - 2.298
     - 15.72
     - 2.393
     - 2.672

On the identical workload, the runtime reaches 99.5% of warp-rl's hand-written environment for full training and
at least 99.6% in every other probe. The remaining differences between rows are workload, not implementation:

* The stable Go2 task runs 2 solver substeps per physics step with Newton's collision pipeline, a contact sensor,
  noise, pushes, and a heavier reset distribution, so it is 1.8x slower than warp-rl's Go2 workload.
* warp-rl's standalone Newton Go2 steps MuJoCo Warp directly with its fast preset (one solver iteration, Euler,
  Menagerie primitives, explicit PD), which is a different simulation.

For the stable Go2 task, learning is 5.4% of a training iteration; the runtime spends 1.1% of the environment
step in the MDP, and the rest is physics.

**Learning.** Fresh learners, 1,500 iterations (196.6 M transitions), two seeds each, same harness:

.. list-table::
   :header-rows: 1

   * - Environment
     - Seed
     - Episode return at 200 / 500 / 800 / 1,500 iterations
     - Episode length at 1,500
     - Wall-clock transitions/s
   * - runtime, warp-rl Go2 MDP
     - 42
     - 11.6 / 17.4 / 26.8 / 38.1
     - 994
     - 1.47 M
   * - runtime, warp-rl Go2 MDP
     - 1
     - 11.9 / 17.2 / 25.9 / 35.9
     - 997
     - 1.47 M
   * - warp-rl hand-written Go2
     - 42
     - 12.4 / 13.7 / 18.5 / 25.0
     - 991
     - 1.53 M
   * - warp-rl hand-written Go2
     - 1
     - 12.0 / 16.5 / 19.6 / 25.9
     - 973
     - 1.51 M

Both learn to walk. With two seeds each, the runtime's higher returns may be seed variance or the small
remaining differences listed in ``warprl_go2_mdp`` (reset joint positions clamped to soft limits, no non-finite
height termination); they are not evidence of a better implementation. Wall-clock throughput includes periodic
metric reads, so it is lower than the GPU-time probes.


Capture boundaries
------------------

Each boundary is reported as one of three levels:

* **MDP capture demonstrated:** one graph containing actions, terminations, rewards, resets, events, commands,
  and observations; physics is bound but not stepped.
* **Physics–MDP capture demonstrated:** one graph also containing the Newton physics step.
* **Whole training loop capture demonstrated:** one graph containing ``horizon`` (32) physics+MDP steps and every
  PPO epoch and minibatch update of warp-rl.

Cartpole, 1,024 environments (``benchmark_mdp_runtime_capture.py``). Step boundaries replay their graph for 100
steps from a fresh reset and compare observations and rewards with 100 eager steps from an identical state.

.. list-table::
   :header-rows: 1

   * - Boundary
     - Backend
     - Replay vs eager (max abs diff)
     - Graph [ms/step]
     - Eager [ms/step]
     - Status
   * - MDP
     - Warp
     - 0.0
     - 0.014
     - 0.096
     - MDP capture demonstrated
   * - MDP
     - Torch
     - 0.0
     - 0.207
     - 1.216
     - MDP capture demonstrated
   * - Physics + MDP
     - Warp
     - 0.0
     - 0.421
     - 5.228
     - Physics–MDP capture demonstrated
   * - Physics + MDP
     - Torch
     - 0.0
     - 0.614
     - 6.299
     - Physics–MDP capture demonstrated

Whole training loop (Cartpole, warp-rl PPO with (64, 64) networks, 300 iterations):

.. list-table::
   :header-rows: 1

   * - MDP backend
     - Graph owner
     - Transitions/s
     - Result
     - Status
   * - Warp
     - warp-rl ``OnPolicyRunner`` (``wp.ScopedCapture``)
     - 1.92 M
     - evaluation return -21.48 -> 4.85
     - Whole training loop capture demonstrated
   * - Torch
     - Torch CUDA graph (``--graph_owner torch``)
     - 1.42 M
     - final training episode return 4.91, survival 99.9%
     - Whole training loop capture demonstrated
   * - Torch
     - warp-rl ``OnPolicyRunner``
     - --
     - --
     - **Blocked**

The Go2 and Reach training probes above are whole-training-loop captures as well.

Blocked boundary
^^^^^^^^^^^^^^^^

**Torch MDP backend inside a Warp-owned training graph.** warp-rl's ``OnPolicyRunner`` captures with
``wp.ScopedCapture`` on a Warp stream. The Torch program enqueues its tensor operations, and redirects the physics
launches, to the current Torch stream, which by default is the legacy default stream. The legacy stream cannot
join a capture, and CUDA rejects the dependency at the first program step inside the capture:

.. code-block:: text

   AcceleratorError: CUDA error: operation would make the legacy stream depend on a capturing blocking stream

Reproduce it with ``--boundary training --backend torch``. Running the Torch operations on the Warp capture stream
instead would record tensor temporaries from Torch's caching allocator into a graph Torch does not own, and Torch
could hand those blocks to other tensors between replays. The supported path captures the same PPO iteration in a
Torch-owned graph (``--graph_owner torch``).


Reproducibility notes
---------------------

* Graph replay equals eager execution bitwise for the MDP and for Cartpole physics. With contacts (Go2), Newton's
  collision pipeline is not bitwise reproducible between runs, eager or replayed.
* An earlier version of the runtime named its generated kernel modules ``"unique"``. Warp's module hash does not
  cover ``wp.static`` values in nested functions, so programs differing only in constants reused each other's
  cached kernels, silently (for example, an observation scale was dropped and Go2 failed to learn). Kernels are now
  keyed by a program signature, and ``test_programs_differing_only_in_constants_do_not_share_kernels`` guards it.
  All numbers on this page were measured after the fix.


Commands
--------

.. code-block:: bash

   # Speed against the stable MDP and the Warp frontend (one task per call)
   uv run python scripts/benchmarks/benchmark_mdp_runtime_compare.py --task go2 --num_envs 4096 --steps 100 \
       --samples 5 --profile

   # Term parity with the stable task
   uv run python scripts/benchmarks/validate_mdp_runtime_parity.py --task go2

   # Training probes and learning against warp-rl
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_training.py \
       --env runtime_go2_warprl_mdp
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_training.py \
       --env warprl_isaac_go2
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_training.py \
       --env runtime_go2_warprl_mdp --seed 42 --skip_probes --train_iterations 1500
   cd ../warp-rl && uv run --project ../mustafa_isaaclab3 --with-editable . python examples/speed_of_light.py \
       --num_envs 4096 --solver fast

   # Capture boundaries
   uv run python scripts/benchmarks/benchmark_mdp_runtime_capture.py --boundary mdp physics_mdp --backend warp torch
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_capture.py --boundary training
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_capture.py --boundary training \
       --backend torch --graph_owner torch

Pass ``--output results.json`` to any benchmark to write the full results, including dependency revisions.
