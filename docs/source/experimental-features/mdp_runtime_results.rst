MDP Runtime Capture Results
===========================

Measured results for :doc:`mdp_runtime` on the runtime Cartpole task
(``isaaclab_tasks_experimental.mdp_runtime.cartpole``) with Newton/MJWarp physics, 1024 environments, physics
at 120 Hz, and decimation 2.

Each capture boundary is reported as one of three levels:

* **MDP capture demonstrated:** one graph containing actions, terminations, rewards, resets, events, and
  observations; physics is bound but not stepped.
* **Physics–MDP capture demonstrated:** one graph also containing the Newton physics step.
* **Whole training loop capture demonstrated:** one graph containing ``horizon`` (32) physics+MDP steps and every
  PPO epoch and minibatch update of warp-rl.

Setup
-----

.. list-table::
   :widths: 30 70

   * - GPU
     - NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB), driver 595.91.07
   * - OS
     - Linux 7.0.0-31-generic x86_64, glibc 2.39
   * - Isaac Lab
     - ``f2aa0e1bf8a4b96cd932f35d1a491564bc949891`` (branch ``feature/experimental-mdp-runtime`` on develop
       ``eeb6e0458``)
   * - Dependencies
     - Python 3.12.13, Warp 1.17.0, Torch 2.12.0+cu130, Newton 1.6.0, mujoco-warp 3.12.0
   * - warp-rl
     - sibling checkout ``../warp-rl``, version 0.1.0, branch ``feat/warp-ppo`` with no commits (untracked files).
       SHA-256 prefix of the sorted ``src/**/*.py`` contents: ``de9442a5129ad63b``

Commands
--------

.. code-block:: bash

   # MDP and physics-MDP boundaries, both backends
   uv run python scripts/benchmarks/benchmark_mdp_runtime_capture.py \
       --boundary mdp physics_mdp --backend warp torch

   # Whole training loop, Warp backend, graph captured by warp-rl's OnPolicyRunner (and eager baseline)
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_capture.py \
       --boundary training --backend warp [--eager]

   # Whole training loop, Torch backend: warp-rl-owned graph (blocked), Torch-owned graph (demonstrated)
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_capture.py \
       --boundary training --backend torch
   uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_capture.py \
       --boundary training --backend torch --graph_owner torch

Pass ``--output results.json`` to write the full results, including dependency revisions and training
histories.

Step boundaries
---------------

Before timing, each step boundary replays its graph for 100 steps from a fresh reset and compares the
observations and rewards with 100 eager steps from an identical state. These steps include many random
resets. Timings are the median of 5 samples of 200 replays (CUDA events) or 200 eager steps (host wall clock,
synchronized).

.. list-table::
   :header-rows: 1

   * - Boundary
     - Backend
     - Replay vs eager (max abs diff)
     - Graph [ms/step]
     - Eager [ms/step]
     - Graph [env-steps/s]
     - Status
   * - MDP
     - Warp
     - 0.0
     - 0.033
     - 0.141
     - 31.2 M
     - MDP capture demonstrated
   * - MDP
     - Torch
     - 0.0
     - 0.203
     - 1.057
     - 5.05 M
     - MDP capture demonstrated
   * - Physics + MDP
     - Warp
     - 0.0
     - 0.437
     - 5.187
     - 2.34 M
     - Physics–MDP capture demonstrated
   * - Physics + MDP
     - Torch
     - 0.0
     - 0.609
     - 7.378
     - 1.68 M
     - Physics–MDP capture demonstrated

The Torch backend is captured with :class:`torch.cuda.CUDAGraph`, with Warp registered as an external capture
(see "Graph lifetime" in :doc:`mdp_runtime`).

Whole training loop
-------------------

warp-rl PPO: horizon 32, 4 epochs, 4 minibatches, (64, 64) networks, ``reward_scale = 1 / step_dt``, seed 42,
300 iterations (9.8 M transitions). Actions are clipped to ``[-1, 1]`` before the ×100 effort scale.
Evaluations are warp-rl's deterministic 600-step evaluations; the maximum return is about 5.

.. list-table::
   :header-rows: 1

   * - MDP backend
     - Graph owner
     - Transitions/s
     - Setup [s]
     - Evaluation return, before → after
     - Final training episode return / survival
     - Status
   * - Warp
     - warp-rl ``OnPolicyRunner`` (``wp.ScopedCapture``)
     - 1.86 M
     - 1.4
     - −21.48 → 4.95
     - 4.90 / 98.9 %
     - Whole training loop capture demonstrated
   * - Warp
     - none (``--eager``)
     - 0.185 M
     - --
     - −21.51 → 4.94
     - 4.90 / 99.4 %
     - Eager baseline
   * - Torch
     - Torch CUDA graph (``--graph_owner torch``)
     - 1.43 M
     - 1.9
     - not run (see below)
     - 4.92 / 99.7 %
     - Whole training loop capture demonstrated
   * - Torch
     - warp-rl ``OnPolicyRunner``
     - --
     - --
     - --
     - --
     - **Blocked**

Captured training runs 10× faster than eager training, and reaches the same return. Setup is the warm-up iteration plus capture
(warp-rl discards its warm-up update; the Torch-owned run keeps it), with warm kernel caches.

Blocked boundary
----------------

**Torch MDP backend inside a Warp-owned training graph.** warp-rl's ``OnPolicyRunner`` captures with
``wp.ScopedCapture`` on a Warp stream. The Torch program enqueues its tensor operations, and redirects the
physics launches, to the current Torch stream, which by default is the legacy default stream. The legacy stream
cannot join a capture, and CUDA rejects the dependency at the first program step inside the capture:

.. code-block:: text

   AcceleratorError: CUDA error: operation would make the legacy stream depend on a capturing blocking stream

Reproduce it with the third command above (``--boundary training --backend torch``).

Running the Torch operations on the Warp capture stream instead would record tensor temporaries from Torch's
caching allocator into a graph Torch does not own. Torch could hand those blocks to other tensors between
replays. The runtime therefore does not attempt it. The supported path captures the same PPO iteration in a
Torch-owned graph (``--graph_owner torch``), as shown above. warp-rl's evaluation also captures with
``wp.ScopedCapture``, so the Torch-owned run reports training metrics instead of before/after evaluations.

What is not measured
--------------------

* Heterogeneous populations run on analytic point-mass physics only. Their single-graph replay is covered by
  ``test_heterogeneous_population``, not timed here.
* No comparison against the stable manager-based Cartpole or the ``--frontend warp`` path is included.
  warp-rl's own documentation reports 1.68 M transitions/s for its adapter around the direct Warp Cartpole
  task, measured on its own setup, which this page does not reproduce.
