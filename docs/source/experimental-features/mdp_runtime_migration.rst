MDP Runtime Migration Note
==========================

This note lists every part of the experimental Warp environment system, i.e. the Warp frontend
(``--frontend warp``), the Warp manager fork, and the direct Warp tasks. For each part it states whether
:doc:`mdp_runtime` replaces it, and what removing the old system would break.

.. important::

   **Status: the Warp frontend system is retained.** The runtime replaces the Cartpole task and the warp-rl
   training path. It does not yet replace the other 17 task paths or the RL-library entry points listed
   below. Removing the old system now would be a deliberate loss of support for all of them, so it is not
   done in this change. The `Removal plan`_ lists the gates.


What is replaced
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Old entry point
     - Replacement
     - Status
   * - ``Isaac-Cartpole`` and ``Isaac-Cartpole-Direct`` with ``--frontend warp`` (Newton/MJWarp)
     - ``isaaclab_tasks_experimental.mdp_runtime.cartpole.CartpoleMdpCfg`` +
       :class:`~isaaclab_experimental.mdp_runtime.NewtonArticulationPhysics`, Warp or Torch backend
     - Replaced. Same terms and parameters as the stable manager-based task, except the logging-only
       ``success_rate`` reward.
   * - warp-rl ``integrations/isaaclab.py`` (``IsaacLabDirectEnv``) for Cartpole
     - :class:`~isaaclab_experimental.mdp_runtime.MdpEnv`, used by
       ``scripts/benchmarks/benchmark_mdp_runtime_capture.py --boundary training``
     - Replaced for Cartpole (physics, MDP, and PPO in one graph).
   * - Per-manager graph capture (``ManagerCallSwitch``, ``WarpGraphCache``)
     - :meth:`MdpProgram.capture <isaaclab_experimental.mdp_runtime.MdpProgram.capture>` (whole step)
     - Replaced for runtime tasks.
   * - Mask-based resets (``reset_mask_wp``, ``InteractiveSceneWarp.reset(env_mask=)``)
     - :meth:`MdpProgram.reset <isaaclab_experimental.mdp_runtime.MdpProgram.reset>` and in-step
       ``reset_mask``
     - Replaced for runtime tasks.


What is not replaced
--------------------

Removing the old system would drop these **task paths**. The frontend derives them from stable
registrations; there are no ``gym.register`` calls in ``isaaclab_tasks_experimental``.

* **Manager-based** (``ManagerBasedRLEnvWarp`` through ``WarpFrontend.adapt_cfg``, pinned in
  ``source/isaaclab_experimental/test/envs/test_frontend_cfg_conversion.py``):

  * ``Isaac-Ant``, ``Isaac-Humanoid``
  * ``Isaac-Reach-Franka``, ``Isaac-Reach-UR10``
  * ``Isaac-Velocity-Flat-AnymalD``, ``Isaac-Velocity-Flat-Cassie``, ``Isaac-Velocity-Flat-G1``,
    ``Isaac-Velocity-Flat-H1``, ``Isaac-Velocity-Flat-UnitreeGo2``
  * ``IsaacContrib-Velocity-Flat-AnymalB``, ``IsaacContrib-Velocity-Flat-AnymalC``,
    ``IsaacContrib-Velocity-Flat-UnitreeA1``, ``IsaacContrib-Velocity-Flat-UnitreeGo1``

  Missing in the runtime: commands, root and body state terms, contact-sensor terms, curricula, noise models,
  and event terms on rigid bodies and materials.

* **Direct** (``DirectRLEnvWarp`` subclasses, resolved by ``_mirror_direct_warp_class``):

  * ``Isaac-Ant-Direct``: ``core/locomotion/ant/ant_warp_env.py:AntWarpEnv``
  * ``Isaac-Humanoid-Direct``: ``core/locomotion/humanoid/humanoid_warp_env.py:HumanoidWarpEnv``
  * ``Isaac-Reorient-Cube-Allegro-Direct``: ``core/reorient/reorient_warp_env.py:ReorientDirectWarpEnv``

  Missing in the runtime: root-state bindings, multi-body observations, and the in-hand reorientation terms.

Removing the old system would also drop these **entry points**, which the runtime does not replace (it has
no Gym environment):

* The ``--frontend {torch,warp}`` flag. It is defined by ``add_frontend_args`` and dispatched by
  ``create_isaaclab_env`` in ``source/isaaclab_rl/isaaclab_rl/entrypoints/common.py``, and used by:

  * the train and play backends ``train_{rl_games,rsl_rl,sb3,skrl,torchrl}.py`` and
    ``play_{rl_games,rsl_rl,sb3,skrl,torchrl}.py`` in ``source/isaaclab_rl/isaaclab_rl/entrypoints/backends/``;
  * the benchmark backends ``benchmark_{train,play}_{rl_games,rsl_rl,sb3,skrl}.py`` in
    ``source/isaaclab/isaaclab/benchmark/entrypoints/backends/``.

* Environment-type checks and wrapper type hints that name the Warp env classes:

  * ``source/isaaclab_rl/isaaclab_rl/utils/env_types.py``
  * ``rsl_rl/vecenv_wrapper.py``, ``rl_games/rl_games.py``, ``sb3.py``, ``torchrl/vecenv_wrapper.py``

* The "Warp" column of the environment browser: ``tools/environ_docs.py`` (``_supports_warp_frontend``) and
  ``docs/source/_static/css/environment-browser.js`` (``--frontend warp`` command strings).


Every affected file
-------------------

Replacing the old system requires changing or deleting the following. Paths are relative to the repository
root.

**Old implementation** (deleted on removal):

* ``source/isaaclab_experimental/isaaclab_experimental/envs/``:

  * ``frontend.py``, ``manager_based_env_warp.py``, ``manager_based_rl_env_warp.py``, ``direct_rl_env_warp.py``,
    ``interactive_scene_warp.py``, ``__init__.pyi``
  * ``mdp/`` (Warp term twins and actions), ``utils/io_descriptors.py``

* ``source/isaaclab_experimental/isaaclab_experimental/managers/`` (the whole Warp manager fork).
* ``source/isaaclab_experimental/isaaclab_experimental/utils/``:

  * ``manager_call_switch.py``, ``warp_graph_cache.py``, ``torch_utils.py``
  * ``buffers/``, ``modifiers/``, ``noise/``, ``warp/``

* ``source/isaaclab_tasks_experimental/isaaclab_tasks_experimental/core/``:

  * ``cartpole``, ``locomotion`` (Ant, Humanoid, shared ``locomotion_env_warp.py``), ``reach``, ``reorient``,
    and ``velocity``: the Warp envs and their ``mdp`` twins.

**Imports and registrations outside the packages** (each must be edited on removal):

* ``import isaaclab_tasks_experimental`` (registration side effect, wrapped in ``contextlib.suppress``):

  * ``source/isaaclab_rl/isaaclab_rl/entrypoints/simple_agents.py`` and the ten train/play backends above
  * ``source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py`` and the eight benchmark backends above
  * ``source/isaaclab/isaaclab/cli/commands/list_envs.py``
  * ``scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py``,
    ``scripts/tutorials/07_visualizers/run_video_recording.py``
  * ``tools/update_environments_rst.py``

* ``from isaaclab_experimental.envs import ...Warp``:

  * ``source/isaaclab_rl/isaaclab_rl/utils/env_types.py``
  * the four wrapper modules above

* ``from isaaclab_experimental.envs.frontend import WarpFrontend``:

  * ``source/isaaclab_rl/isaaclab_rl/entrypoints/common.py``
  * ``tools/environ_docs.py``

* Package lists:

  * ``source/isaaclab/isaaclab/cli/commands/install.py`` (``CORE_ISAACLAB_SUBMODULES``)
  * ``source/isaaclab/isaaclab/benchmark/entrypoints/startup.py``
  * ``pyproject.toml`` (uv sources)
  * ``docs/conf.py`` (``sys.path``)

  These stay, because the runtime lives in the same packages.

**Tests:**

* Deleted with the implementation:

  * ``source/isaaclab_experimental/test/envs/`` (``test_frontend.py``, ``test_frontend_cfg_conversion.py``,
    ``mdp/test_*_warp_parity.py``, ``mdp/test_capture_safety.py``, ``mdp/parity_helpers.py``)
  * ``source/isaaclab_experimental/test/utils/test_manager_call_switch.py``, ``test_warp_graph_cache.py``

* Edited:

  * ``source/isaaclab_rl/test/test_entrypoints_common.py`` (frontend flag and dispatch)
  * ``source/isaaclab/test/benchmark/test_api.py`` (``args.frontend`` default)
  * ``tools/test/test_environ_docs.py`` (``supports_warp_frontend``)
  * ``source/isaaclab_tasks/test/core/test_env_cfg_no_forbidden_imports.py`` and
    ``test_direct_scene_ownership.py`` (paths into ``isaaclab_tasks_experimental``)
  * ``source/isaaclab/test/envs/_env_cfgs_fresh_process.py``,
    ``source/isaaclab/test/install_ci/cli/test_cli_install_core_in_uvenv_correctness.py`` (package names;
    unchanged by removal)

**Documentation:**

* ``docs/source/concepts/warp_environments.rst`` and ``docs/source/concepts/warp_environment_migration.rst``
  (toctree in ``docs/source/concepts/physics_backends.rst``), plus their redirect stubs under
  ``docs/source/overview/core-concepts/physical-backends/newton/``
* ``docs/source/api/lab_experimental/isaaclab_experimental.{envs,envs.mdp.actions,managers,utils}.rst`` and
  ``docs/source/api/index.rst``
* mentions in ``docs/source/setup/ecosystem.rst`` and ``docs/source/refs/troubleshooting.rst``

**CI and ownership:** ``.github/workflows/build.yaml`` (job ``test-isaaclab-experimental``, filter
``isaaclab_experimental``: it also runs the runtime tests and stays), ``.github/CODEOWNERS``, and
``.github/workflows/license-exceptions.json`` (package entries; unchanged).

**External:** the sibling ``warp-rl`` project imports ``WarpFrontend``, ``CartpoleWarpEnv``, ``AntWarpEnv``,
``reset``, ``reset_actions``, ``reset_joints``, ``reset_root``, ``zero_mask_int32``, ``add_to_env``, and
``InteractiveSceneWarp`` in ``src/warp_rl/integrations/{factory,isaaclab,go2}.py``. Its Cartpole path is
replaced by :class:`~isaaclab_experimental.mdp_runtime.MdpEnv`. Its Ant and Go2 paths would break.


Porting a task
--------------

#. Declare the MDP as an :class:`~isaaclab_experimental.mdp_runtime.MdpCfg`. Map each stable
   ``ObsTerm``/``RewTerm``/``DoneTerm``/``EventTerm`` to a runtime term cfg with ``term="<name>"``. Replace
   ``SceneEntityCfg("robot", joint_names=[...])`` with ``params={"joints": [...]}``: the asset is the binding.
#. For every term without a runtime implementation, call ``define_term`` once, then ``implement`` it for the
   backends you need. Declare every field it reads and writes.
#. Provide a physics binding exposing the fields your terms need.
#. Check Torch/Warp agreement and eager/replay equality as in
   ``source/isaaclab_experimental/test/mdp_runtime/``.


Removal plan
------------

Remove the old system when each item below has a runtime replacement, or when a release decides to drop it
explicitly:

#. Root-state, body-state, command, and contact-sensor fields and terms. This unlocks Ant, Humanoid,
   locomotion velocity tasks, and Reach.
#. A Gym-compatible wrapper around :class:`~isaaclab_experimental.mdp_runtime.MdpProgram`, so the ``isaaclab_rl``
   entry points can select it instead of ``--frontend warp``.
#. A public, capture-safe stepping API in the Newton manager, replacing the private ``_simulate_full`` call.
#. Then delete the files listed under "Old implementation", edit the imports above, and remove
   ``--frontend``.
