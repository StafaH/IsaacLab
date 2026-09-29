Experimental MDP Runtime
========================

.. currentmodule:: isaaclab_experimental.mdp_runtime

:mod:`isaaclab_experimental.mdp_runtime` executes an MDP specification (actions, commands, observations, rewards,
terminations, and events) over preallocated buffers in a fixed order. The same configuration runs on a **Warp**
or a **Torch** backend. On Warp, all terms of a step are fused into three generated kernels. A whole step, with
or without physics, can be captured into one CUDA graph and replayed, alone or inside a learner's graph.

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
   * - Stable reset events, reward logging, and interval events index with dynamic ``env_ids``.
     - Every reset and event runs per environment, predicated on the mask.
   * - The Warp frontend captures per manager (``WarpGraphCache`` keyed by stage name). The scene write,
       ``sim.step``, ``scene.update``, commands, curricula, and recorders stay outside the graphs.
     - One program step (actions, physics, terminations, rewards, resets, commands, events, observations) can
       be captured as a single graph, together with a learner.
   * - Every manager term is at least one kernel launch, plus reductions: tens of launches per step for a
       locomotion task.
     - Warp terms are per-environment functions inlined into three generated kernels per step, independent of
       the number of terms.
   * - Warp term parameters are frozen as Python scalars at first capture; a reward weight changing between
       zero and non-zero needs ``invalidate_wp_graphs()``, which nothing calls.
     - Structure is fixed at compile time and a zero weight is still computed. Reward weights live in device
       memory and can be changed without recapture.
   * - Several Warp terms cache ranges and scratch buffers on the module function object, so two terms using
       the same function with different parameters share state.
     - Each term's parameters are baked into its own generated function. Nothing is stored on functions.
   * - Warp terms are found by mirroring module and class names of stable terms (``_swap_mdp``). Stable and
       Warp parameters are not validated against one shared declaration.
     - A term is declared once (:class:`TermSpec`), and each backend registers an implementation of that
       declaration by name.
   * - Derived articulation data (body-frame velocities, projected gravity, heading) is computed lazily on
       property access, which a captured graph only replays if the access happened during capture.
     - Terms compute derived quantities from the raw root state in the same fused kernel.
   * - ``warp-rl`` reaches Isaac Lab through private attributes of two direct task classes and reimplements
       their reset logic.
     - :class:`MdpEnv` exposes any program through warp-rl's fixed-buffer contract.


Lifecycle
---------

.. code-block:: python

   from isaaclab_experimental.mdp_runtime import MdpEnv, NewtonPhysics, compile_plan
   from isaaclab_tasks_experimental.mdp_runtime.go2_velocity import STABLE_TASK, Go2FlatVelocityMdpCfg
   from isaaclab_tasks_experimental.mdp_runtime.stable import stable_physics_cfgs

   sim_cfg, scene_cfg, decimation = stable_physics_cfgs(STABLE_TASK, num_envs=4096)
   # ... create the SimulationContext and the scene, then sim.reset() ...
   physics = NewtonPhysics(scene, "robot", decimation, contact_sensor="contact_forces")
   plan = compile_plan(Go2FlatVelocityMdpCfg(), physics, backend="warp")  # validate + resolve, no device memory
   inputs, state, outputs = plan.allocate()                               # explicit buffers
   program = plan.bind(inputs, state, outputs)                            # generate and record the kernels

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
* :meth:`ExecutionPlan.bind` owns **binding**: it builds the backend executor for those buffers and returns an
  :class:`MdpProgram`. A plan can be bound to several buffer sets.


Configuration
-------------

A configuration is an :class:`MdpCfg` of dictionaries. Dictionary order is execution order and column order.

.. code-block:: python

   @configclass
   class Go2FlatVelocityMdpCfg(MdpCfg):
       episode_length_s: float = 20.0
       commands = {"base_velocity": CommandTermCfg(term="uniform_velocity", resampling_time_range=(10.0, 10.0),
                                                   params={"lin_vel_x": (-1.0, 1.0), ...})}
       actions = {"joint_pos": ActionTermCfg(term="joint_position", scale=0.25, params={"use_default_offset": True})}
       observations = {"policy": ObservationGroupCfg(terms={
           "base_lin_vel": ObservationTermCfg(term="base_lin_vel", noise=(-0.1, 0.1)),
           "velocity_commands": ObservationTermCfg(term="generated_commands", params={"command": "base_velocity"}),
           ...
       })}
       rewards = {"feet_air_time": RewardTermCfg(term="feet_air_time", weight=0.25,
                                                 params={"command": "base_velocity", "threshold": 0.5,
                                                         "contact_bodies": ".*_foot"}), ...}
       terminations = {"base_contact": TerminationTermCfg(term="illegal_contact",
                                                          params={"threshold": 1.0, "contact_bodies": "base"}), ...}
       events = {"push_robot": EventTermCfg(term="push_by_setting_velocity", mode="interval",
                                            interval_range_s=(10.0, 15.0), params={...}), ...}

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
     - Registered term name and its keyword parameters. Name parameters are resolved at compile time:
       ``joints``, ``bodies``, ``contact_bodies`` (regular expressions matched in the given order against the
       binding's joint, body, and contact-body names), ``command`` (a command name, resolved to its columns),
       ``terms`` (termination names, reward terms only).
   * - :class:`ActionTermCfg`
     - ``scale``, ``offset``, ``clip``
     - ``processed = clamp(raw * scale + offset, *clip)`` per action column.
   * - :class:`CommandTermCfg`
     - ``resampling_time_range``
     - Per-environment resampling interval [s]; commands also resample on reset.
   * - :class:`ObservationTermCfg`
     - ``noise``, ``clip``, ``scale``
     - ``value = clamp(value + U(noise), *clip) * scale``, the stable order. Noise draws one value per column.
   * - :class:`ObservationGroupCfg`
     - ``terms``
     - Terms concatenated along the last dimension. Groups are column ranges of one output buffer.
   * - :class:`RewardTermCfg`
     - ``weight``
     - ``reward = sum_k(value_k * weight_k * step_dt)`` in declaration order, as in the stable ``RewardManager``.
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
* unknown and missing parameters, and ``(lower, upper)`` ordering of ``*_range`` (also per axis of range
  dictionaries), ``bounds``, ``clip``, and ``noise``;
* that name parameters match: joints, bodies, contact bodies, command names, termination names;
* that every field a term reads or writes exists, and that runtime fields are only read after their producing
  stage (see `Ordering and dependencies`_);
* at least one action term and one observation group, no empty group, positive widths, finite weights, and the
  ``mode``/``interval_range_s`` combination of events.

Errors at run time are explicit: buffers of the wrong shape passed to :meth:`ExecutionPlan.bind` raise
``ValueError``; :meth:`MdpProgram.reset` with host ``env_ids`` and :meth:`MdpProgram.set_reward_weight` raise
``RuntimeError`` during capture; :class:`MdpEnv` rejects actions of the wrong shape or dtype; capture on a
non-CUDA device raises ``RuntimeError``.


Term interface
--------------

A term has **one shared declaration** and **one implementation per backend**:

.. code-block:: python

   from isaaclab_experimental.mdp_runtime import REQUIRED, Stage, define_term, implement
   from isaaclab_experimental.mdp_runtime.wp_math import root_lin_vel_b

   define_term(
       "track_lin_vel_xy_exp",
       Stage.REWARD,
       params={"std": REQUIRED, "command": REQUIRED},   # name -> default, or REQUIRED
       reads=("root_pose_w", "root_vel_w", "commands"),  # physics or runtime fields
       doc="exp(-|v_cmd_xy - v_xy|^2 / std^2).",
   )

   @implement("track_lin_vel_xy_exp", "warp")
   def _(ctx):                                            # runs once, at bind time
       col, variance = ctx.params["command_columns"][0], ctx.params["std"] ** 2

       @wp.func
       def term(env: int, f: Any) -> float:             # one environment; f holds every field
           v = root_lin_vel_b(f, env)
           dx = f.commands[env, col] - v[0]
           dy = f.commands[env, col + 1] - v[1]
           return wp.exp(-(dx * dx + dy * dy) / wp.static(variance))

       return term

   @implement("track_lin_vel_xy_exp", "torch")
   def _(ctx):                                            # runs once, at bind time
       f, out, col = ctx.fields, ctx.out, ctx.params["command_columns"][0]
       ...
       return lambda: torch.exp(-error(f, col) / variance, out=out)   # all environments

The declaration (:class:`TermSpec`) fixes what both backends share: the **kind** (action, command, observation,
reward, termination, or event), the **parameters** and defaults, the **fields** read and written (only action
and event terms may write, and only physics fields), and the **width** of action, command, and observation
terms.

**Warp implementations** are factories returning a per-environment :func:`warp.func`. Parameters, column
offsets, and index lists are Python values captured by the closure, so they are compile-time constants of the
generated code. Signatures by kind:

========================= ======================================================================
Kind                      Warp function
========================= ======================================================================
action                    ``(env, f)``: reads ``f.processed_actions`` columns, writes fields
observation               ``(env, f, out)``: writes its columns of ``out``
reward                    ``(env, f) -> float``
termination               ``(env, f) -> bool``
event                     ``(env, f, state: wp.uint32) -> wp.uint32``: draws from and returns the stream
command                   ``(resample(env, f, state) -> wp.uint32, update(env, f))``
========================= ======================================================================

**Torch implementations** are binders returning a callable over all environments, from a :class:`TermContext`
holding only the declared fields, the output view, and index tensors. Event callables take the environment
mask: ``run(mask)``. Command binders return ``(resample(mask), update())``. Masked callables leave unmasked
environments and their random streams unchanged.

Built-in terms (:mod:`isaaclab_experimental.mdp_runtime.builtin_terms`) mirror the stable functions of the same
name. :func:`registered_terms` lists every term and its backends.

* **Actions:** ``joint_effort``, ``joint_position``.
* **Commands:** ``uniform_velocity`` (with heading control and standing environments), ``uniform_pose``.
* **Observations:** ``joint_pos_rel``, ``joint_vel_rel``, ``last_action``, ``generated_commands``,
  ``base_lin_vel``, ``base_ang_vel``, ``projected_gravity``.
* **Rewards:** ``is_alive``, ``is_terminated``, ``termination_term``, ``joint_pos_target_l2``,
  ``joint_vel_l1``, ``joint_vel_l2``, ``joint_acc_l2``, ``joint_torques_l2``, ``action_l2``, ``action_rate_l2``,
  ``lin_vel_z_l2``, ``ang_vel_xy_l2``, ``flat_orientation_l2``, ``track_lin_vel_xy_exp``,
  ``track_ang_vel_z_exp``, ``feet_air_time``, ``position_command_error``, ``orientation_command_error``.
* **Terminations:** ``time_out``, ``joint_pos_out_of_manual_limit``, ``illegal_contact``,
  ``pose_command_success``.
* **Events:** ``reset_joints_by_offset``, ``reset_joints_by_scale``, ``push_joints_by_velocity``,
  ``reset_root_state_uniform``, ``push_by_setting_velocity``.


State and buffer layout
-----------------------

``N`` is the number of environments, ``A`` action columns, ``C`` command columns, ``K`` reward terms, ``T``
termination terms, ``E`` interval events, and ``W`` the total observation width. Warp buffers are ``wp.array``;
Torch buffers are ``torch.Tensor`` on the same device.

.. list-table::
   :header-rows: 1
   :widths: 20 30 17 33

   * - Owner
     - Buffer
     - Shape, dtype
     - Content
   * - :class:`MdpInputs`
     - ``actions``
     - ``(N, A)`` float32
     - Raw actions, written by the caller before a step.
   * - :class:`MdpState`
     - ``action``, ``prev_action``, ``processed_actions``
     - ``(N, A)`` float32
     - Raw action of this and the previous step (zeroed on reset), and the processed action.
   * -
     - ``episode_length``
     - ``(N,)`` int32
     - Steps since reset.
   * -
     - ``rng``
     - ``(N,)`` uint32 (Torch: int64)
     - Per-environment PCG32 stream shared by events, commands, timers, and noise.
   * -
     - ``commands``, ``command_state``, ``command_time_left``
     - ``(N, C)``, ``(N, S)``, ``(num_commands, N)``
     - Command values (each command owns columns), private command state, resampling timers [s].
   * -
     - ``termination_values``, ``reward_values``, ``episode_sums``
     - ``(T, N)`` bool, ``(K, N)``, ``(K, N)``
     - Per-term values, and weighted episode sums (zeroed on reset).
   * -
     - ``reward_weights``
     - ``(K,)`` float32
     - Device-resident weights.
   * -
     - ``interval_time_left``, ``interval_fired``
     - ``(E, N)`` float32, bool
     - Interval timers [s] and the masks of this step.
   * -
     - ``reset_request``, ``commit_mask``
     - ``(N,)`` bool
     - Mask of :meth:`MdpProgram.reset`; environments whose physics state events changed.
   * - :class:`MdpOutputs`
     - ``observation_buffer``, ``final_observation_buffer``
     - ``(N, W)`` float32
     - Post-reset and pre-reset observations; ``observations[group]`` are column views.
   * -
     - ``reward``; ``terminated``, ``truncated``, ``reset_mask``
     - ``(N,)`` float32; bool
     - Step outputs. ``reset_mask = terminated | truncated``.

On the Warp backend, every buffer above and every physics field is a member of one generated
:func:`warp.struct` (``f`` in the term functions), so a term reads any field by name.

Physics fields are owned by the :class:`PhysicsBinding`, which must keep their addresses. The schema
(documented in :mod:`isaaclab_experimental.mdp_runtime.physics`) covers joint state, targets, and limits; root
pose and COM velocity; default root state and environment origins; link poses; and contact-sensor forces and
air times. A binding provides the subset its simulator supports:

* :class:`NewtonPhysics` binds an Isaac Lab articulation, and optionally a contact sensor, simulated by Newton.
  Fields are the articulation and sensor data arrays, plus binding-owned command buffers.
* :class:`PointMassPhysics` is an analytic integrator for physics-free tests and the heterogeneous example.

A binding implements ``prepare(reads, writes)`` (called once at bind time with the fields the terms use, so it
applies only the needed commands and refreshes only the needed data), ``step()``, ``reset(mask)``
(simulator-internal state such as sensors), and ``commit(mask)`` (make event writes effective in the solver).

:class:`NewtonPhysics` steps physics as Isaac Lab environments do. When Newton runs every actuator inside the
solver step (Newton-native actuators), it sets the decimation, submits targets, runs the whole decimation loop in
one solver call, and updates the scene data once per control step. Otherwise it applies targets, steps, and
updates the data once per physics step. Only the commands that terms write are applied, and contact and body
data are refreshed only if terms read them.


Ordering and dependencies
-------------------------

:meth:`MdpProgram.step` runs, in this order (:attr:`MdpProgram.schedule` lists it by term):

#. Action processing, then action terms. **Warp kernel 1.**
#. ``physics.step``.
#. Episode-length increment, terminations, rewards, and, with ``compute_final_observations``, pre-reset
   observations. **Warp kernel 2** (steps 3 to 5).
#. For resetting environments: reset events, runtime-state reset, interval timers and commands resampled.
#. Command timers and updates, then interval events.
#. ``physics.reset`` of the resetting environments, and ``physics.commit`` of the environments events changed.
#. Observations. **Warp kernel 3.**

The order matches the stable ``ManagerBasedRLEnv.step``; curricula and recorders do not exist in the runtime.
Within a stage, terms run in declaration order. Because every term is per environment, fusing a stage into one
kernel does not change results.

A term may read a physics field in any stage. It may read a runtime field only if the producing stage comes
strictly earlier:

.. list-table::
   :header-rows: 1

   * - Runtime field
     - Produced by
     - Readable by
   * - ``action``, ``prev_action``, ``episode_length``
     - action stage (``episode_length`` right after physics)
     - terminations, rewards, events, commands, observations
   * - ``terminated``, ``truncated``, ``termination_values``
     - termination stage
     - rewards, events, commands, observations
   * - ``commands``
     - command stage (state)
     - every stage; terminations and rewards see the previous step's command, as in the stable environment

For example, ``is_alive`` sees the terminations of the same step. A termination that reads ``terminated``
fails compilation.


Reset semantics
---------------

Resets are masks of shape ``(N,)``. In a step, ``reset_mask = terminated | truncated`` selects the
environments to reset after rewards (and final observations). :meth:`MdpProgram.reset` takes ``env_ids``, which
builds ``reset_request`` on the host and is rejected during capture, or a device ``mask``, which is copied on
the device and is capture-safe.

For each masked environment, a reset:

#. runs the reset events in order; they draw from that environment's random stream;
#. zeroes ``episode_length``, ``action``, ``prev_action``, and ``episode_sums``;
#. resamples interval timers, then each command's timer and value;
#. resets simulator-internal state (``physics.reset``) and commits the written state (``physics.commit``);
#. recomputes observations.

Unmasked environments are not modified (tested bitwise). Random draws come from one PCG32 stream per
environment; Warp and Torch implement the same integer arithmetic and the same draw order, so both backends
draw bitwise identical values. Terms document their draw order, e.g. ``reset_joints_by_offset`` draws all
position offsets, then all velocity offsets, in ``joints`` order.


Backend dispatch
----------------

The backend is chosen once, in :func:`compile_plan`. It selects the implementation of every term and the
executor of the runtime-owned work (action processing, reductions, noise, clip and scale, resets, timers):

* **Warp:** generates the field struct and three kernels per program (``pre``, ``post``, ``observe``) plus one
  for external resets, with ``wp.static`` loops unrolled over the term functions. Kernels are recorded once
  with ``record_cmd=True``; a step is three launches plus the physics binding's work. A program's kernels live
  in a Warp module named after a signature of everything baked into them (terms, resolved parameters, columns,
  configuration values, field layout). Warp's own module hash does not cover ``wp.static`` values in nested
  functions, so without this, programs that differ only in constants would reuse each other's cached kernels.
  Identical programs still share the cache.
* **Torch:** records one tensor callable per term and per runtime operation, in the same order. Physics fields
  are zero-copy views of the binding's Warp arrays. Work runs on the current Torch stream, and Warp physics
  launches are redirected to that stream.

Agreement is tested on identical inputs: flags, episode lengths, and random draws are identical; float outputs
agree within ``1e-5`` absolute on analytic physics and ``1e-4`` on Newton scenes. The residual comes from float
contraction (e.g. fused multiply-add) in Warp kernels.


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
* **Reproducibility:** replay equals eager execution bitwise for the MDP, and for Cartpole physics. With contacts
  (Go2), Newton's collision pipeline is not bitwise reproducible between runs, eager or replayed.

Learners that capture themselves (e.g. warp-rl's ``OnPolicyRunner``) call :meth:`MdpEnv.step` inside their own
capture; the same rules apply.


Performance model
-----------------

A runtime step costs the physics binding's work plus three MDP kernels. The MDP kernels are one thread per
environment, reading fields once and keeping intermediate values in registers. Their cost is measured
separately (``mdp_only`` in :doc:`mdp_runtime_results`): on Go2 with 4,096 environments they take about 1% of
the step. The remaining time is the physics step, so no faster MDP implementation can speed this step up by
more than that share. Further speedups must come from the physics configuration (solver substeps, iterations,
contact capacity) or from the binding.


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


Tasks
-----

:mod:`isaaclab_tasks_experimental.mdp_runtime` provides MDPs that run on the scenes of stable tasks
(:func:`~isaaclab_tasks_experimental.mdp_runtime.stable.stable_physics_cfgs` resolves the stable task on
Newton/MJWarp):

* ``cartpole.CartpoleMdpCfg``: ``Isaac-Cartpole``, all terms except the logging-only success-rate reward.
* ``go2_velocity.Go2FlatVelocityMdpCfg``: ``Isaac-Velocity-Flat-UnitreeGo2``, all active terms. Omitted: the
  startup randomization of friction, base mass, and base COM, and the zero reset external force.
* ``franka_reach.FrankaReachMdpCfg``: ``Isaac-Reach-Franka`` with joint-position actions. Omitted: the
  reward-weight curriculum. The success termination uses the current pose error instead of the error from the
  previous command update.

``scripts/benchmarks/validate_mdp_runtime_parity.py`` checks each task against its stable task on identical
states (see :doc:`mdp_runtime_results`).


Learner integration
-------------------

:class:`MdpEnv` exposes one observation group of a program through the fixed-buffer contract of the
``warp-rl`` learner (``observations``, pre-reset ``final_observations``, ``rewards``, ``terminated``,
``truncated`` as ``wp.array``, plus ``step(actions)`` and ``reset()``). It copies the learner's action array into
the program input, so the learner may pass a different array every step. With ``action_bounds``, the copy clamps the actions,
e.g. to ``(-5, 5)`` for unbounded Gaussian policies: the program then processes, stores, and observes the bounded
action, while the learner keeps its sample for likelihoods.


Current limitations
-------------------

* No curricula, recorders, logging extras, observation histories or delays, sensors other than contact
  sensors, or startup events.
* :class:`NewtonPhysics` binds one articulation and at most one contact sensor, requires stateless actuators
  (implicit or DC motor), assumes gravity along ``-z`` for projected gravity, and calls private
  ``NewtonManager`` methods (``_simulate_full``, ``forward``). There is no PhysX binding.
* There is no Gym registration and no ``isaaclab train`` integration. The runtime is driven from Python or
  through :class:`MdpEnv`.
