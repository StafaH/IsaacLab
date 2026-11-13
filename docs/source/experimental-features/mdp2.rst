.. _mdp2:

MDP2: Compiled MDPs
===================

.. note::

   MDP2 lives in ``isaaclab_experimental.mdp2``, and its tasks in ``isaaclab_tasks.mdp2``. It is an experimental
   feature: expect its API to change.

MDP2 describes a task's MDP (actions, observations, rewards, terminations, commands, events, and quantities) as
configuration, and compiles it into a few kernels per step. With the Warp backend, a whole environment step, physics
included, replays as one CUDA graph. :class:`~isaaclab_experimental.envs.MDP2RLEnv` runs an MDP2 task on Newton with
the Gym interface of :class:`~isaaclab.envs.ManagerBasedRLEnv`, so the RL wrappers work unchanged. MDP2 tasks run on
Newton only and do not import Isaac Sim.


Quick Start
-----------

The tasks ``Isaac-Cartpole-MDP2``, ``Isaac-Velocity-Flat-UnitreeGo2-MDP2``,
``Isaac-Velocity-Flat-WalkStand-UnitreeGo2-MDP2``, ``Isaac-Lift-KukaAllegro-MDP2``, and the motion tracking tasks
``Isaac-Tracking-Humanoid-Walk-MDP2``, ``-Run-MDP2``, and ``-Dance-MDP2`` train with RSL-RL:

.. code-block:: bash

   uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
       --task Isaac-Cartpole-MDP2 presets=newton_mjwarp


Terms and uses
--------------

A term is a calculation with an implementation per backend: a batched Torch function, and a ``wp.func`` for one
world that MDP2 calls from the kernels it generates. A use binds a term to the data it reads and to its role in the
task. Inputs are named explicitly; nothing is gathered behind the term's back.

.. code-block:: python

   import warp as wp
   from isaaclab_experimental import mdp2


   @wp.func
   def _joint_vel_l1(world: int, joint_vel: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)) -> float:
       total = float(0.0)
       for k in range(joint_ids.shape[0]):
           total += wp.abs(joint_vel[world, joint_ids[k]])
       return total


   joint_vel_l1 = mdp2.Term(torch=lambda joint_vel, joint_ids: joint_vel[:, joint_ids].abs().sum(-1), warp=_joint_vel_l1)

   robot = mdp2.Entity("robot")
   cart_vel = mdp2.Reward(
       joint_vel_l1,
       inputs={"joint_vel": robot.joint_vel, "joint_ids": robot.joints("slider_to_cart")},
       weight=-0.01,
   )

The built-in terms are in :mod:`isaaclab_experimental.mdp2.mdp` and keep the names of their counterparts in
:mod:`isaaclab.envs.mdp`. A :class:`~isaaclab_experimental.mdp2.TaskCfg` groups the uses by role, and
:class:`~isaaclab_experimental.envs.MDP2RLEnvCfg` holds the task with its scene and simulation. See
``isaaclab_tasks/mdp2/cartpole`` for a complete task.

- **State.** A term can keep :class:`~isaaclab_experimental.mdp2.State` across steps, reset with its world's
  episode unless it is persistent. Other terms read it with :func:`~isaaclab_experimental.mdp2.term_state`.
- **Quantities.** A :class:`~isaaclab_experimental.mdp2.Quantity` computes a value per world once per step, which
  rewards, terminations, and events read as ``mdp2.quantities.<name>``.
- **Events** write scene fields in place. Before an event with ``commit=True``, and after all reset events, the scene
  makes the simulation see the written fields: joint and root state, fixed-base root poses, and gravity.
- **Randomness** comes from per-world streams that each use owns (``Term(random=True)`` passes ``rng``), so the Torch
  and Warp backends draw the same numbers.


Building and inspecting
-----------------------

:class:`~isaaclab_experimental.mdp2.MDPBuilder` turns a task into an immutable
:class:`~isaaclab_experimental.mdp2.MDP`. Between adding the task and finalizing, the schedule can be inspected and
reordered within a phase. ``finalize`` reports every error at once, and ``mdp.describe()`` prints each op's layout and
each runtime function's steps.

.. code-block:: python

   builder = mdp2.MDPBuilder(scene_data.schema)
   builder.add_task(task)
   mdp = builder.finalize(device="cuda:0", backend="warp")
   print(mdp.describe())

:class:`~isaaclab_experimental.mdp2.MDPBuilder.replicate` runs different task configs, with their own rewards and
weights, in different worlds of one scene.


Different robots in different worlds
------------------------------------

The worlds of one environment can also hold different robots, each with its own task. The scene declares every
robot and the combinations of assets that a world may hold, the physics keeps the assets outside a world's
combination asleep, and the environment takes one task per combination, in their order:

.. code-block:: python

   @configclass
   class G1Go2FlatSceneCfg(InteractiveSceneCfg):
       g1: ArticulationCfg = replace(G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/G1")
       go2: ArticulationCfg = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Go2")
       clone_cfg: CloneCfg = CloneCfg(clone_combinations=[InclusionSet(assets=["g1"]), InclusionSet(assets=["go2"])])


   physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(sleep_inactive_assets=True))
   task = [g1_task, go2_task]

MuJoCo Warp simulates worlds of one structure. With
:attr:`~isaaclab_newton.physics.MJWarpSolverCfg.sleep_inactive_assets`, every world holds every robot and keeps the
ones outside its combination always asleep: they do not move or collide, are not rendered, and take no room in the
constraint solver. Each robot then has data for every world, and a task reads and writes its robot in the worlds of
its combination. Observations and actions are padded to those of the widest task; ``mdp.world_configs`` holds the
worlds of each task and the columns that it uses. Uses of one name share their state between the tasks, so the
rewards and terminations of robots whose actions differ in size need names of their own.

``Isaac-Velocity-Flat-G1-Go2-MDP2`` trains a Unitree G1 in half of the worlds and a Unitree Go2 in the other half
to follow velocity commands, with the tasks ``Isaac-Velocity-Flat-G1-MDP2`` and
``Isaac-Velocity-Flat-UnitreeGo2-MDP2`` as they are and a policy per robot:

.. code-block:: bash

   uv run python scripts/reinforcement_learning/mdp2/train_velocity_g1_go2.py --budget 800

``--policy shared`` trains one network for both robots: an encoder and a decoder per robot around a trunk that the
robots share. ``--style 0.5`` gives each robot a second task in half of its worlds, so the simulation runs four
tasks. The worlds of a style task follow the same commands and start on frames of motion clips, of a person walking
for the G1 and of a dog trotting for the Go2. In place of the rewards that shape the gait by hand, they get a style
reward from a discriminator, which the learner trains to tell the motion of the robot from that of the clips. The
policy of a robot learns from both of its tasks. ``Isaac-Velocity-Flat-G1-Style-MDP2`` and
``Isaac-Velocity-Flat-UnitreeGo2-Style-MDP2`` are the two tasks of one robot alone. The clips are downloaded, see
the ``motions`` folders of ``isaaclab_tasks/mdp2/tracking/config/g1`` and
``isaaclab_tasks/mdp2/velocity_style/config/go2``.


Backends and capture
--------------------

- **Warp** fuses the terms' ``wp.func`` s into a few generated kernels per runtime function. It is the fast path.
- **Torch** runs each term as a batched Torch function. It is the reference, and convenient for prototyping.

A Warp MDP runs the Torch hooks of a term that has no Warp ones between its kernels, so a task can start in Torch and
move its expensive terms to Warp. Both backends, and such mixed MDPs, are captured: ``MDP2Env`` replays each step
after the first as CUDA graphs. Torch work in a graph runs on Warp's stream and allocates from a private memory pool,
so it must not copy data from the host or draw from Torch's random generators; use the term's ``rng`` instead.

**Eager terms** (``Term(eager=fn)``) are plain Python functions with MDP1-style access to the scene, called as
``fn(ctx, **params)``. They may synchronize with the host or call any library, so they never run inside a graph:
the environment captures the work before and after them, physics included, and runs them between the graphs.
``Term(init=fn)`` fills a term's state once at startup in the same way.


Limitations
-----------

- Newton only. Different robots in different worlds need MuJoCo Warp contacts and
  :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.sleep_inactive_assets`. The scene and the solver keep the data
  of every robot for every world, also where it sleeps, so a step costs more than the steps of the robots simulated
  apart: 1.3 times for a G1 and a Go2.
- Asset resets with a mask reset the actuator state of every world, so the environments support implicit, ideal PD,
  and DC motor actuators only, which keep no state, and refuse delayed, actuator-network, and BAM actuators.
- Joint orderings (``ArticulationCfg.joint_ordering``) are not supported: terms see the solver's joint order.
- Asset data that is derived on demand, such as ``root_lin_vel_b``, is not refreshed by a captured step. Terms read
  the fields of the scene data, which alias the solver state; code outside the MDP that reads asset data between
  steps should use those too, for example ``root_com_vel_w`` rotated by ``root_link_pose_w``.
