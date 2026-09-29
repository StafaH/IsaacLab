Experimental MDP Runtime
========================

.. currentmodule:: isaaclab_experimental.mdp_runtime

:mod:`isaaclab_experimental.mdp_runtime` executes an MDP specification (actions, observations, rewards,
terminations, and events) as a fixed list of operations over preallocated buffers. The same configuration
runs on a **Warp** or a **Torch** backend, and a whole step, with or without physics, can be captured into one
CUDA graph and replayed.

The runtime is separate from the stable :mod:`isaaclab.managers` API and from the experimental Warp manager
fork (``isaaclab_experimental.managers`` and ``--frontend warp``). Neither is changed by it. See
:doc:`mdp_runtime_results` for measurements and :doc:`mdp_runtime_migration` for what moving off the Warp
frontend involves.

.. warning::

   This is an experimental API. It may change without deprecation.


Why a new runtime
-----------------

The inventory of the stable MDP stack, the experimental Warp frontend, and the sibling ``warp-rl`` project
found the following limitations, which the runtime is designed to remove.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Limitation of the existing systems
     - How the runtime addresses it
   * - The stable ``ManagerBasedRLEnv.step`` selects resets with ``reset_buf.nonzero()`` and ``len(...)``, a
       host synchronization and a dynamic shape on every step. The Warp manager env inherits this.
     - Resets are boolean masks everywhere. No operation depends on how many environments reset.
   * - Stable reset events, reward logging, and interval events index with dynamic ``env_ids``
       (``RewardManager.reset`` gathers with ``env_ids``; ``EventManager`` builds interval ids with
       ``nonzero``).
     - Every reset and event kernel runs over all environments and is predicated on the mask.
   * - The Warp frontend captures per manager (``WarpGraphCache`` keyed by stage name). The scene write,
       ``sim.step``, ``scene.update``, commands, curricula, and recorders stay outside the graphs. Scene
       capture is capped because the actuator model uses Torch.
     - One program step includes actions, physics, terminations, rewards, resets, events, and observations,
       and can be captured as a single graph, together with a learner.
   * - Warp term parameters (``dt``, clip bounds, interval ranges) are frozen as Python scalars at first
       capture. A reward weight changing between zero and non-zero needs ``invalidate_wp_graphs()``, which
       nothing calls.
     - Structure is fixed at compile time and a zero weight is still computed. Reward weights live in device
       memory and can be changed without recapture.
   * - Several Warp terms cache ranges and scratch buffers on the module function object, so two terms using
       the same function with different parameters share state.
     - Binders capture per-term parameters in closures and recorded launches. Nothing is stored on functions.
   * - Warp terms are found by mirroring module and class names of stable terms (``_swap_mdp``). Stable and
       Warp parameters are not validated against one shared declaration.
     - A term is declared once (:class:`TermSpec`), and each backend registers an implementation of that
       declaration by name.
   * - ``STABLE`` mode of ``ManagerCallSwitch`` fails for most stages because no ``stable_call`` is passed.
     - Backend choice is one argument of :func:`compile_plan` and applies to every term.
   * - ``warp-rl`` reaches Isaac Lab through private attributes of two direct task classes
       (``CartpoleWarpEnv``, ``AntWarpEnv``) and reimplements their reset logic.
     - :class:`MdpEnv` exposes any program through warp-rl's fixed-buffer contract.


Lifecycle
---------

.. code-block:: python

   from isaaclab_experimental.mdp_runtime import MdpEnv, NewtonArticulationPhysics, compile_plan
   from isaaclab_tasks_experimental.mdp_runtime.cartpole import DECIMATION, CartpoleMdpCfg

   physics = NewtonArticulationPhysics(scene, "robot", DECIMATION)  # or PointMassPhysics(...)
   plan = compile_plan(CartpoleMdpCfg(), physics, backend="warp")   # validate + resolve, no device memory
   inputs, state, outputs = plan.allocate()                           # explicit buffers
   program = plan.bind(inputs, state, outputs)                        # record every operation once

   program.reset()                  # all environments (host-built mask, outside capture)
   graph = program.capture()        # one eager warm-up step, then capture one step
   for _ in range(1000):
       inputs.actions.assign(...)   # or write the array in place from a learner
       graph.replay()               # outputs.observations["policy"], outputs.reward, ...

The three phases have separate owners:

* :func:`compile_plan` owns **validation and resolution** and returns an immutable :class:`ExecutionPlan`.
* :meth:`ExecutionPlan.allocate` owns **allocation**: :class:`MdpInputs`, :class:`MdpState`, and
  :class:`MdpOutputs`. Callers may allocate them separately, or pass shared views to
  :meth:`ExecutionPlan.allocate_outputs`.
* :meth:`ExecutionPlan.bind` owns **binding**. It returns an :class:`MdpProgram` whose operations all refer to
  those buffers. A plan can be bound to several buffer sets, e.g. for independent rollouts.


Configuration
-------------

A configuration is an :class:`MdpCfg` of dictionaries. Dictionary order is execution order and column order.

.. code-block:: python

   @configclass
   class CartpoleMdpCfg(MdpCfg):
       episode_length_s: float = 5.0
       actions = {"joint_effort": ActionTermCfg(term="joint_effort", params={"joints": ["slider_to_cart"]}, scale=100.0)}
       observations = {"policy": ObservationGroupCfg(terms={
           "joint_pos_rel": ObservationTermCfg(term="joint_pos_rel"),
           "joint_vel_rel": ObservationTermCfg(term="joint_vel_rel"),
       })}
       rewards = {"alive": RewardTermCfg(term="is_alive", weight=1.0), ...}
       terminations = {"time_out": TerminationTermCfg(term="time_out", time_out=True), ...}
       events = {"reset_cart_position": EventTermCfg(term="reset_joints_by_offset", params={...}), ...}

Shared configuration fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every field is backend-independent. Backends never read a field that is not listed here.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Class
     - Field
     - Semantics (identical on both backends)
   * - :class:`TermCfg`
     - ``term``, ``params``
     - Registered term name and its keyword parameters. ``joints`` holds joint-name regular expressions
       resolved against the binding's ``joint_names`` in the given order (``preserve_order=True``).
   * - :class:`ActionTermCfg`
     - ``scale``, ``offset``, ``clip``
     - ``processed = clamp(raw * scale + offset, *clip)`` per action column, as in the stable joint actions.
   * - :class:`ObservationTermCfg`
     - ``scale``, ``clip``
     - ``value = clamp(value, *clip) * scale``, the stable order (clip, then scale).
   * - :class:`ObservationGroupCfg`
     - ``terms``
     - Terms concatenated along the last dimension into one ``(N, width)`` float32 buffer.
   * - :class:`RewardTermCfg`
     - ``weight``
     - ``reward = sum_k(value_k * weight_k * step_dt)`` in declaration order, as in the stable
       ``RewardManager``.
   * - :class:`TerminationTermCfg`
     - ``time_out``
     - A ``time_out`` term sets ``truncated``; any other term sets ``terminated``. Both may be set.
   * - :class:`EventTermCfg`
     - ``mode``, ``interval_range_s``
     - ``"reset"`` runs for resetting environments. ``"interval"`` runs per environment when its timer expires.
   * - :class:`MdpCfg`
     - ``episode_length_s``, ``compute_final_observations``, ``seed``
     - ``max_episode_length = ceil(episode_length_s / step_dt)``. Final observations are pre-reset copies.
       ``seed`` seeds the per-environment random streams.

Validation and errors
^^^^^^^^^^^^^^^^^^^^^

:func:`compile_plan` checks the whole configuration and raises one :class:`MdpConfigError` listing every
problem as ``<path>: <reason>``, for example ``rewards.pole_pos: missing required parameters ['target']``. It
checks:

* the backend name, a positive environment count, and a positive ``episode_length_s``;
* that each term name is registered, has the expected kind, and is implemented for the chosen backend;
* unknown and missing parameters, ``(lower, upper)`` ordering of ``*_range``, ``bounds``, and ``clip``;
* that ``joints`` patterns match joint names;
* that every field a term reads or writes exists, and that runtime fields are only read after their producing
  stage (see `Ordering and dependencies`_);
* at least one action term, at least one observation group, no empty group, positive widths, finite reward
  weights, and the ``mode``/``interval_range_s`` combination of events.

Errors at run time are explicit: buffers of the wrong shape passed to :meth:`ExecutionPlan.bind` raise
``ValueError``; :meth:`MdpProgram.reset` with host ``env_ids`` raises ``RuntimeError`` during capture;
:meth:`MdpProgram.set_reward_weight` raises during capture; :class:`MdpEnv` rejects actions of the wrong shape or
dtype; capture on a non-CUDA device raises ``RuntimeError``.


Term interface
--------------

A term has **one shared declaration** and **one implementation per backend**.

.. code-block:: python

   from isaaclab_experimental.mdp_runtime import REQUIRED, Stage, define_term, implement

   define_term(
       "joint_vel_l1",
       Stage.REWARD,
       params={"joints": ".*"},       # name -> default, or REQUIRED
       reads=("joint_vel",),          # physics or runtime fields
       doc="L1 norm of the selected joint velocities.",
   )

   @implement("joint_vel_l1", "warp")
   def _(ctx):
       return wp.launch(kernel, dim=ctx.num_envs,
                        inputs=[ctx.fields["joint_vel"], ctx.indices["joint_ids"], ctx.out],
                        device=ctx.device, record_cmd=True).launch

   @implement("joint_vel_l1", "torch")
   def _(ctx):
       joint_vel, ids, out = ctx.fields["joint_vel"], ctx.indices["joint_ids"], ctx.out
       return lambda: torch.sum(torch.abs(joint_vel[:, ids]), dim=1, out=out)

The declaration (:class:`TermSpec`) fixes what both backends share:

* the **kind**: action, observation, reward, termination, or event;
* the **parameters** and their defaults;
* the **fields** it reads and writes. Only action and event terms may write, and only physics fields;
* the **output**: a width function for observation and action terms, otherwise one value per environment.

An implementation is a **binder**: ``binder(ctx: TermContext) -> Callable[[], None]``. It runs once in
:meth:`ExecutionPlan.bind` and returns the callable run every step. The :class:`TermContext` passes only the
fields the term declared, the resolved parameters, prepared index arrays (``joint_ids``), and the output view
(observation columns, a reward row, a termination row), the processed action columns, or the event mask and
random stream.

The binder must not allocate, synchronize, or branch on device data inside the returned callable. The Warp
convention is a recorded launch (``record_cmd=True``). The Torch convention is out-parameter or in-place tensor
operations with masked updates (``torch.where``, ``masked_fill_``) instead of index selection.

Built-in terms (:mod:`isaaclab_experimental.mdp_runtime.builtin_terms`) mirror the stable functions of the
same name:

* **Actions:** ``joint_effort``.
* **Observations:** ``joint_pos_rel``, ``joint_vel_rel``.
* **Rewards:** ``is_alive``, ``is_terminated``, ``joint_pos_target_l2``, ``joint_vel_l1``, ``joint_vel_l2``,
  ``action_rate_l2``.
* **Terminations:** ``time_out``, ``joint_pos_out_of_manual_limit``.
* **Events:** ``reset_joints_by_offset``, ``push_joints_by_velocity``.

:func:`registered_terms` lists every term and the backends that implement it.


State and buffer layout
-----------------------

``N`` is the number of environments, ``A`` the number of action columns, ``K`` the number of reward terms,
``T`` the number of termination terms, and ``E`` the number of interval events. Warp buffers are ``wp.array``;
Torch buffers are ``torch.Tensor`` on the same device.

.. list-table::
   :header-rows: 1
   :widths: 20 28 17 35

   * - Owner
     - Buffer
     - Shape, dtype
     - Content
   * - :class:`MdpInputs`
     - ``actions``
     - ``(N, A)`` float32
     - Raw actions, written by the caller before a step.
   * - :class:`MdpState`
     - ``action``, ``prev_action``
     - ``(N, A)`` float32
     - Raw action of this and the previous step; zeroed on reset.
   * -
     - ``processed_actions``
     - ``(N, A)`` float32
     - ``clamp(raw * scale + offset)``; action terms read their columns.
   * -
     - ``episode_length``
     - ``(N,)`` int32
     - Steps since reset.
   * -
     - ``rng``
     - ``(N,)`` uint32 (Torch: int64)
     - Per-environment PCG32 stream shared by events and timers.
   * -
     - ``termination_values``, ``reward_values``
     - ``(T, N)`` bool, ``(K, N)`` float32
     - Per-term values; each term writes one contiguous row.
   * -
     - ``episode_sums``, ``reward_weights``
     - ``(K, N)``, ``(K,)`` float32
     - Weighted episode sums (zeroed on reset) and device-resident weights.
   * -
     - ``interval_time_left``, ``interval_fired``
     - ``(E, N)`` float32, bool
     - Per-environment interval timers [s] and the masks of this step.
   * -
     - ``reset_request``
     - ``(N,)`` bool
     - Mask used by :meth:`MdpProgram.reset`.
   * - :class:`MdpOutputs`
     - ``observations[group]``, ``final_observations[group]``
     - ``(N, width)`` float32
     - Post-reset and pre-reset observations; terms write column slices.
   * -
     - ``reward``; ``terminated``, ``truncated``, ``reset_mask``
     - ``(N,)`` float32; bool
     - Step outputs. ``reset_mask = terminated | truncated``.

Physics fields are owned by the :class:`PhysicsBinding`, which must keep their addresses. The built-in terms
use a joint-space schema documented in :mod:`isaaclab_experimental.mdp_runtime.physics`: ``joint_pos``,
``joint_vel``, ``default_joint_pos``, ``default_joint_vel``, ``soft_joint_pos_limits`` ``(N, J, 2)``,
``soft_joint_vel_limits``, and ``joint_effort_target``. Two bindings are provided:

* :class:`NewtonArticulationPhysics` binds an Isaac Lab articulation simulated by Newton. Its fields are the
  articulation data arrays, plus a binding-owned effort buffer.
* :class:`PointMassPhysics` is an analytic integrator for physics-free tests and the heterogeneous example.

A binding implements ``step()`` (apply ``joint_effort_target``, advance one control period) and
``commit(mask)`` (make event writes of masked environments effective in the solver).


Ordering and dependencies
-------------------------

:meth:`MdpProgram.step` runs this list. :attr:`MdpProgram.op_names` returns it for inspection.

#. ``action.process``, then each action term (writes physics command fields).
#. ``physics.step``.
#. ``runtime.episode_length`` (increment), each termination term, ``termination.reduce``.
#. Each reward term, ``reward.reduce``.
#. If ``compute_final_observations``: each observation term, then ``final_observation.<group>`` (copy).
#. Reset events on ``reset_mask``, ``physics.commit.reset``, ``runtime.reset_state``, and interval-timer
   resampling for the reset environments.
#. For each interval event: ``tick`` (produces its mask), the event, ``physics.commit.interval.<term>``.
#. Each observation term, then ``observation.<group>.post`` if any term scales or clips.

The order matches the stable ``ManagerBasedRLEnv.step``, except that commands, curricula, and recorders do
not exist in the runtime. Within a stage, terms run in declaration order.

A term may read a physics field in any stage. It may read a runtime field only if the producing stage comes
strictly earlier:

.. list-table::
   :header-rows: 1

   * - Runtime field
     - Produced by
     - Readable by
   * - ``action``, ``prev_action``, ``episode_length``
     - action stage (``episode_length`` right after physics)
     - terminations, rewards, events, observations
   * - ``terminated``, ``truncated``
     - termination stage
     - rewards, events, observations

For example, ``is_alive`` sees the terminations of the same step, as in the stable implementation. A
termination that reads ``terminated`` fails compilation.


Reset semantics
---------------

Resets are masks of shape ``(N,)``. In a step, ``reset_mask = terminated | truncated`` selects the
environments to reset after rewards (and final observations) have been computed. :meth:`MdpProgram.reset`
takes ``env_ids``, which builds ``reset_request`` on the host and is rejected during capture, or a device
``mask``, which is copied on the device and is capture-safe.

For each masked environment, a reset:

#. runs the reset events in order. They consume draws from that environment's random stream;
#. commits the written joint state to the solver (``physics.commit``);
#. zeroes ``episode_length``, ``action``, ``prev_action``, and ``episode_sums``;
#. resamples interval timers from ``interval_range_s``;
#. recomputes observations.

Unmasked environments are not modified by any of these steps (tested bitwise).

Random draws come from one PCG32 stream per environment. Warp and Torch implement the same integer
arithmetic, so both backends draw bitwise identical values. Terms draw in a documented order, e.g.
``reset_joints_by_offset`` draws all position offsets, then all velocity offsets, in ``joints`` order.


Backend dispatch
----------------

The backend is chosen once, in :func:`compile_plan`. It selects the implementation of every term and of the
runtime-owned operations (action processing, reductions, observation post-processing, reset bookkeeping,
timers):

* **Warp:** ``wp.array`` buffers. Every operation is a launch recorded at bind time and replayed with fixed
  arguments. Work runs on the current Warp stream.
* **Torch:** ``torch.Tensor`` buffers. Physics fields are zero-copy views of the binding's Warp arrays. Work runs
  on the current Torch stream, and Warp physics launches are redirected to that stream, so both kinds of work
  stay ordered.

Agreement is tested on the same inputs: flags and episode lengths are identical. Float outputs agree within
``1e-5`` absolute on analytic physics and within ``1e-4`` after 40 Newton Cartpole steps. The residual comes from
float contraction (e.g. fused multiply-add) in Warp kernels, amplified by chaotic dynamics.


Graph lifetime
--------------

:meth:`MdpProgram.capture` (and :func:`capture_step`) captures one step:

* **Warp** programs capture with :class:`warp.ScopedCapture` on the current Warp stream.
* **Torch** programs capture with :class:`torch.cuda.CUDAGraph`. Warp is registered as an external capture on
  the same stream (``wp.capture_begin(external=True)``), with one persistent :class:`warp.Stream` object per
  Torch stream. Without that registration, MJWarp's graph-conditional solver loop (``wp.capture_while``)
  synchronizes and invalidates the capture.

A graph remains valid while its owner (program or population) and the physics binding keep their buffers. The
:class:`CapturedStep` holds a reference to its owner. What a replay sees:

* **Picked up without recapture:** new contents of any bound buffer (actions, physics state, reward weights set
  with :meth:`MdpProgram.set_reward_weight`).
* **Requires a new plan, program, and capture:** changes to terms, parameters, widths, environment counts, or
  buffer addresses.
* **Warm-up:** ``capture(warmup=True)`` runs one eager step first, which advances the state. Use
  ``warmup=False`` after an eager step has already compiled the kernels.
* **Host state:** host-side counters of the simulator (simulation time) do not advance during replays. Physics
  state and the runtime's device counters do.

Learners that capture themselves (e.g. warp-rl's ``OnPolicyRunner``) call :meth:`MdpEnv.step` inside their own
capture; the same rules apply.


Heterogeneous environments
--------------------------

**Definition.** A population is heterogeneous when its environments differ in *agent type*, and so may differ
in the MDP term set, observation widths, action widths, joint count, or physics. A batch of identical
environments whose observations are zero-padded to a common width is **not** heterogeneous, and the runtime
does not present it as such.

**Layout.** :class:`HeterogeneousProgram` partitions the population into contiguous per-type row blocks:

* each type has its own compiled plan and dense ``(N_g, width_g)`` observation and ``(N_g, A_g)`` action
  buffers. Nothing is padded;
* per-environment scalars (``reward``, ``terminated``, ``truncated``, ``reset_mask``) are population-wide
  ``(N,)`` buffers, and type ``g`` writes rows ``[offset_g, offset_g + N_g)`` through views;
* :meth:`HeterogeneousProgram.packed_observations` adds an explicit extra operation that copies every type into
  a zero-padded ``(N, max_width)`` buffer, and returns each type's valid width. Padding carries no information.

A step runs every type on one stream, so the whole population is captured in one graph.
:mod:`isaaclab_tasks_experimental.mdp_runtime.point_mass` defines two types: a 1-joint slider (observation 2,
action 1) and a 3-joint gantry (observation 6, action 3) with extra rewards, an extra termination, and an
interval event.

**Not supported yet:** types that share one physics model (e.g. two articulation kinds in one Newton scene) need
a binding that exposes per-type sub-views and a single shared physics step. Each type currently needs its own
binding, and the program rejects shared bindings. Learners that consume several types (e.g. multi-agent PPO)
are out of scope.


Learner integration
-------------------

:class:`MdpEnv` exposes one observation group of a program through the fixed-buffer contract of the
``warp-rl`` learner (``observations``, pre-reset ``final_observations``, ``rewards``, ``terminated``,
``truncated`` as ``wp.array``, plus ``step(actions)`` and ``reset()``). It copies the learner's action array into
the program input, so the learner may pass a different array every step.

``scripts/benchmarks/benchmark_mdp_runtime_capture.py`` trains Cartpole with warp-rl PPO with physics, MDP, and
learning in one graph. See :doc:`mdp_runtime_results`.


Current limitations
-------------------

* Terms cover joint-space Cartpole-style tasks. There are no commands, curricula, sensors, body or root-state
  terms, observation noise, history, or delays.
* :class:`NewtonArticulationPhysics` supports one articulation with stateless implicit actuators, and calls
  private ``NewtonManager`` methods (``_simulate_full``, ``forward``). There is no PhysX binding.
* There is no Gym registration and no ``isaaclab train`` integration. The runtime is driven from Python or
  through :class:`MdpEnv`.
* No logging extras: episode sums are kept on the device, but no ``extras["log"]`` is produced.
