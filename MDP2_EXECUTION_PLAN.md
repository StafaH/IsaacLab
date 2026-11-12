# MDP2 research: execution plan

The plan to execute, tracked in the branch `feature/mdp2-examples` so that the work can continue on any machine. It
is the only plan document in the branch. The literature review behind it is an untracked working document on the
workstation where the project started and is not needed to execute.

Work through the packages of section 7 in the order of section 8. Keep **Status** and **Log** at the end current
and commit the file with every update.

This branch carries an unpublished research plan: do not push it to a public remote unless the owner says so.

## 1. Goal

Massively parallel imitation learning fixes the mixture of training worlds by hand: a tracking stage followed by a
50/50 split (HIL, arXiv 2505.12619), masks drawn from fixed distributions followed by distillation (HOVER,
MaskedMimic), teacher-to-student ratios that are constant or ramped on a clock. We treat the mixture as the variable
to optimise. Every world runs one of several MDP variants, called arms, that differ in what the policy is told, in
objective, in perturbation or in scene. A few percent of the worlds are evaluation worlds inside the same
simulation: they measure competence per arm on fixed starts with deterministic actions. An allocator moves the
training worlds between arms from that signal. MDP2 is what makes a different MDP per world cheap to write and run.

- **Target: RSS 2027.** Six-page abstract on 4 December 2026, written only if Gate 1 passes. Full paper on 16 April
  2027, with results on a Unitree G1. Fallback: CoRL 2027 with the same plan.
- **Not the claim: speed or scale.** Tracking one clip in seconds with a captured environment step is published
  (InstantMimic, arXiv 2609.09821), and the largest generalists train on 64 to 128 GPUs (SONIC, Scaling BFM). Our
  ground is the mixture, measured with seeds, with one GPU per run.

## 2. Hypotheses and gates

| | Hypothesis | Decided by |
|---|---|---|
| H1 | At 10 billion samples or fewer, a uniform fixed mixture from scratch loses to two-stage distillation at equal samples | Gate 1 |
| H2 | Some schedule from teacher-heavy to relaxed closes most of that gap | Gate 1 |
| H3 | The allocator matches the best mixture or schedule found by a sweep, in one run, without tuning per problem | Gate 2 |
| H4 | Signals from evaluation worlds beat signals from training rollouts (biased by exploration noise, start sampling and perturbations) | Gate 2 |
| H5 | Transfer between arms is asymmetric and changes over training | Analysis (W5) |
| H6 | Tracking plus adversarial worlds from scratch fails with a fixed split and trains with allocation | Gate 2 |

**Measures.** Per interface, the held-out success: the share of evaluation episodes on held-out clips that track for
the whole 10 s. The final value of a run is the mean of its last ten log rows. S of a method is that value averaged
over the interfaces and then over the seeds. An interface whose specialist stays below 20% at the budget is left out
of S and reported separately (it stays in the task: an arm that cannot be learned must not absorb worlds). Two
methods are *separated* when their S differ by 0.10 or more and the seeds of one all lie above the seeds of the
other.

**Gate 1 (26 November)** compares, at equal total samples and three seeds: U, the uniform mixture; D, two-stage
distillation; B, the best single-stage alternative (schedules, other fixed mixtures, self-distillation).

| Outcome | Condition | What follows |
|---|---|---|
| The mixture matters | B is separated from U | Allocator (W3) on this problem, write the abstract |
| Only distillation works | D is separated from U and from B | W3 as the test of whether allocation closes the gap; abstract only if the allocator is within 0.05 of D by 2 December |
| Flat | Nothing is separated from U | Mixing is easy on this problem. Move to the harder arms and problems of W4 and repeat the comparison there. If that is flat too, the paper becomes the large study (many mixtures, schedules and critic designs with seeds), the allocator a secondary result, and no abstract is sent |

**Gate 2 (17 December).** The allocator, with one setting of its constants, reaches S within 0.03 of the best
result of the sweep on at least two problems. If not, the study and the transfer analysis are the paper and the
allocator is reported as a negative result.

## 3. What exists

| Path | Content |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/mdp2/interfaces/` | Task `Isaac-Interfaces-G1-MDP2`: the G1 tracks clips and is told its targets through one interface per episode |
| `.../interfaces/config/g1/g1_env_cfg.py` | `TRAIN_CLIPS` (25), `TEST_CLIPS` (9, held out by subject), `INTERFACES`, physics limits |
| `.../tracking/mdp/events.py` | `reset_to_clips`: starts drawn towards frames where training worlds fail, fixed starts for evaluation worlds, frames lifted out of the ground |
| `scripts/reinforcement_learning/mdp2/ppo.py` | PPO whose update is one CUDA graph; privileged critic; imitation loss |
| `scripts/reinforcement_learning/mdp2/train_interfaces.py` | The learner: shares, a share switch, teacher labels, self-distillation, evaluation worlds, JSON log |
| `source/isaaclab_tasks/test/mdp2/test_interfaces.py` | Test of the task |
| `.../mdp2/hil/`, `train_hil.py` | Tracking worlds plus goal worlds with a discriminator's style reward, on the 28-DoF humanoid: start of problem 2 |
| `.../mdp2/recovery/` | Pushes in half of the worlds: start of problem 3 |

**The task.** Every world tracks the whole clip and is rewarded for all of it; the interface only decides what the
policy is told. Targets are given 1, 5 and 15 steps ahead (a step is 1/30 s). Actions are position targets of the
29 body joints, offset from the default pose. The critic sees the full reference, privileged state and the
interface of its world.

| Interface | The policy is told |
|---|---|
| `full` | Positions of root, torso, hands, feet, knees and elbows, and all joint positions |
| `key_points` | The body positions only |
| `three_point` | Torso and hands (the G1 has no head body) |
| `upper_body` | Root position and the joints of waist and arms |

The learner steers the task through four state arrays: `shares` (one per interface, read when an episode starts),
`interface` (the interface of every world), `fixed` (a world's interface, or -1 to draw it) and `fixed_start` (a
world's start frame, or -1 to draw it). No change to the MDP2 runtime was needed for this.

**Protocol P1** (everything in section 4 and W1). 8192 worlds, of which the last 512 are evaluation worlds: 64 per
interface and split (training clips, held-out clips), start frames spread evenly over the split, deterministic
actions, no data for the learner. Episodes last 10 s. An iteration is 24 steps of the 7680 training worlds, 184,320
samples. The script writes a log row every 50 iterations (9.2 million samples) to the JSON file of `--out`:
`{"args": ..., "history": [row, ...]}`, a row holding `samples_m`, `time`, `reward`, `shares`, and for every
interface and split `<interface>/<train|test>/success` and `<interface>/<train|test>/error_mm` (mean distance of
the bodies to the clip).

**Throughput.** 0.56 million samples per second on an RTX PRO 6000 Blackwell, so 1000 million samples take 30
minutes; about 8 GB of GPU memory.

## 4. Results so far

Pilot on the workstation: one seed, 1000 million samples per run, protocol P1. Held-out success with the mean body
error in brackets. Preliminary: one seed, and no run has converged.

| Run | `full` | `key_points` | `three_point` | `upper_body` |
|---|---|---|---|---|
| Specialist `full` | 70% (68 mm) | | | |
| Specialist `three_point` | | | 69% (83 mm) | |
| Uniform mixture | 48% (79 mm) | 72% (76 mm) | 61% (87 mm) | 3% (128 mm) |
| Schedule: `full`, then uniform from 500 M | 70% (76 mm) | 76% (77 mm) | 28% (100 mm) | 2% (138 mm) |
| Specialists `key_points`, `upper_body` | pending | | | |
| Two-stage distillation (1000 M teacher + 1000 M student) | pending | | | |
| Self-distillation | pending | | | |

Reading so far:

- The uniform mixture does not fail wholesale. With a quarter of the samples on `three_point` it is 8 points behind
  the specialist of that interface, so the interfaces teach each other.
- It pays on `full` (22 points behind the specialist) and it fails `upper_body`. Whether `upper_body` can be learned
  alone is the pending specialist run.
- The schedule trades the other way: `full` stays at the specialist's level and `key_points` is the best so far,
  but `three_point` has had 125 M samples of its own and is at 28%, still rising. Neither the uniform mixture nor
  this schedule is better on every interface, which is the situation an allocator is meant for.
- The specialists are still improving at 1000 M (`full`: 20%, 40%, 59%, 70% at 250, 500, 750 and 1000 M), so the
  comparison needs longer budgets. Success stays at 0 until about 140 M.
- Halving the initial exploration noise (`--init_std 0.5`) was slower.
- 808 frames of the clips put bodies in the ground and the simulation diverged when a world started there. The
  frames are now lifted (`ground_clearance`), and a world whose state is not finite ends its episode.

## 5. Bring-up on a new machine

1. **Code.** Check out `feature/mdp2-examples`, which sits on `main` of isaac-sim/IsaacLab. Then `uv sync`: the
   default environment has Newton and no Isaac Sim, which is all this work uses.
2. **Clips.** The G1 files of the LAFAN1 retargeting dataset are not in the repository (non-commercial, not
   redistributable; never commit them):

   ```bash
   M=source/isaaclab_tasks/isaaclab_tasks/mdp2/tracking/config/g1/motions
   URL=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1
   .venv/bin/python -c "
   from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import TEST_CLIPS, TRAIN_CLIPS
   print('\n'.join((*TRAIN_CLIPS, *TEST_CLIPS)))" 2>/dev/null | while read name; do
       curl -fsSL -o $M/$name.csv $URL/$name.csv
   done
   ls $M/*.csv | wc -l    # 34
   ```

3. **Test.** `uv run --extra test python -m pytest source/isaaclab_tasks/test/mdp2/test_interfaces.py
   source/isaaclab_tasks/test/mdp2/test_tracking.py` reports 4 passed.
4. **Smoke run.** `.venv/bin/python scripts/reinforcement_learning/mdp2/train_interfaces.py --budget 20
   --log_interval 20`. The first run compiles the Warp kernels, which takes minutes. It prints
   `SETUP obs 305 critic 466 actions 29 train worlds 7680 eval block 64`, a few `LOG` lines and
   `TRAIN ... graphs 1`.
5. **Reproduce.** `mkdir -p runs/research`, then the same script with `--shares full=1 --budget 1000 --out
   runs/research/repro_full.json`. Expect the `full` row of section 4: no success until about 140 M, held-out
   success near 70% at 1000 M. Seeds and GPUs differ, so within about 10 points. Record the throughput of the
   machine's GPU in the log of this file.

## 6. Running batches

One run per GPU. A batch is a text file with one run per line: a name, then the arguments of
`train_interfaces.py`. This function runs it over the GPUs and skips runs that already finished (dry-run tested):

```bash
launch() {  # launch <batch> <number of GPUs>, with the runs in runs/research/<batch>.txt
    mkdir -p runs/research/$1
    grep -v '^#' runs/research/$1.txt | PHASE=$1 xargs -P $2 --process-slot-var=GPU -L 1 bash -c '
        out=runs/research/$PHASE/$0
        grep -qs "^TRAIN" $out.log && exit 0
        echo "COMMIT $(git rev-parse --short HEAD) GPU $(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader)" > $out.log
        CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python scripts/reinforcement_learning/mdp2/train_interfaces.py \
            --out $out.json --save $out.pt "$@" >> $out.log 2>&1'
}
```

- **Outputs** go to `runs/research/<batch>/<name>.{json,log,pt}`. `runs/` is ignored by git; copy what must be kept
  to storage that outlives the machine, and put the numbers that matter into section 4.
- **Pin the code of a batch.** A run imports the task and the learner when it starts, so editing the checkout
  changes the later runs of a batch. Launch batches from their own worktree with their own environment, and develop
  in the main checkout: `git worktree add --detach ../mdp2-batch HEAD`, then `uv sync` and a copy of the clips
  there.
- **Compile once.** Do the smoke run of section 5 in that worktree before launching, so that the parallel runs find
  the kernels in the cache. On a cluster with a shared home directory, point `WARP_CACHE_PATH` at a local disk.
- **Use `.venv/bin/python` for runs**, not `uv run`, which may re-sync the environment under running jobs.
- **One protocol per comparison.** Runs are comparable only if they share the commit of the task and the learner.
  A change to `ppo.py`, `train_interfaces.py`, `mdp2/interfaces/` or `mdp2/tracking/mdp/` starts a new protocol
  (P2, ...): note it in the log and rerun the baselines it is compared with.
- **Timing claims** only from runs that had their GPU to themselves, with the GPU model named.
- The script takes the device `cuda:0`, so `CUDA_VISIBLE_DEVICES` selects the GPU. Training one run on several
  GPUs is not built and not needed before W6.

## 7. Work packages

GPU-hours are for the throughput of section 3.

### W0. Bring-up

Section 5 on the multi-GPU machine. Done when the reproduction matches and its throughput is in the log.

### W1. Gate 1 batch (no new code)

Protocol P1, 2000 million samples per method in total, seeds 0 to 2. About 44 GPU-hours: 11 hours on four GPUs.

```bash
mkdir -p runs/research
ALL=full=1,key_points=1,three_point=1,upper_body=1
for s in 0 1 2; do
    for i in full key_points three_point upper_body; do
        echo "${i}_s$s --seed $s --budget 2000 --shares $i=1"
    done
    echo "uniform_s$s --seed $s --budget 2000 --shares $ALL"
    echo "self_distill_s$s --seed $s --budget 2000 --shares $ALL --self_distill"
    for at in 250 500 1000; do
        echo "schedule${at}_s$s --seed $s --budget 2000 --shares full=1 --shares_after $ALL --switch_at $at"
    done
    echo "mix_full10_s$s --seed $s --budget 2000 --shares full=1,key_points=3,three_point=3,upper_body=3"
    echo "mix_full40_s$s --seed $s --budget 2000 --shares full=2,key_points=1,three_point=1,upper_body=1"
    echo "mix_full70_s$s --seed $s --budget 2000 --shares full=7,key_points=1,three_point=1,upper_body=1"
    echo "mix_upper40_s$s --seed $s --budget 2000 --shares full=1,key_points=1,three_point=1,upper_body=2"
    echo "teacher_s$s --seed $s --budget 1000 --shares full=1"
done > runs/research/gate1.txt
launch gate1 4

# The students need the teachers of the first batch.
DISTILL="--rl 0 --entropy 0 --init_std 0.1 --budget 1000 --shares $ALL"
for s in 0 1 2; do
    echo "two_stage_s$s --seed $s $DISTILL --teacher runs/research/gate1/teacher_s$s.pt"
    echo "two_stage_long_s$s --seed $s $DISTILL --teacher runs/research/gate1/full_s$s.pt"
done > runs/research/gate1b.txt
launch gate1b 4
```

| Runs | Stand for |
|---|---|
| `full`, `key_points`, `three_point`, `upper_body` | Specialists: what all samples on one interface buy |
| `uniform` | U: HOVER's from-scratch baseline, the mode sampling of single-stage generalists |
| `teacher` + `two_stage` | D at equal samples: 1000 M teacher on `full`, then 1000 M of imitating its labels in every interface |
| `full` + `two_stage_long` | Distillation with a head start: 2000 M teacher, 3000 M in total |
| `schedule250`, `schedule500`, `schedule1000` | Hand schedules: `full` alone, then uniform (HIL's shape) |
| `mix_*` | Other fixed mixtures: what a sweep buys |
| `self_distill` | The policy's own action from full targets as the label in the other interfaces |

Then, on one seed, check that the ranking holds at 8000 M: `uniform`, two-stage with 4000 M each, the best
schedule, and the `upper_body` specialist (16 GPU-hours).

Done when the table of section 4 holds these runs with mean and range over seeds, and the outcome of Gate 1 is
written into the log with the numbers behind it.

### W2. Tooling (while W1 runs)

1. **Summary script.** Reads `runs/research/<batch>/*.json`, groups runs by name without the seed, and prints per
   method the final held-out success and error per interface (mean and range over seeds), S, and the samples to
   reach 50%. Markdown output, for section 4.
2. **Checkpoints.** `--checkpoint` and `--save_every` with the whole learner state: networks, normalisers,
   optimiser moments, learning rate. Done when 500 M plus a resumed 500 M is within seed range of 1000 M.
3. **Final evaluation.** A mode that loads a model and uses every world as an evaluation world (1024 per interface
   and split) for three episodes. Reported numbers come from this, not from the 64-world blocks.
4. **Training-side statistics per interface** in the log: mean reward and episode length of the training worlds.
   The baselines of H4 need them.
5. **Return normalisation per arm** in the critic, behind a flag. Needed once arms differ in reward (W4).

### W3. Allocator, version 1

A pure function from the competence history to shares, so that it can be tested without a simulation, called from
`train_interfaces.py` at every log row.

```
c[i]    competence of arm i: the share of the 10 s that its evaluation worlds on training clips track
        (mean episode length / 300 steps: it moves from the first iterations, success does not)
p[i]    absolute learning progress: |fast average of c[i] - slow average of c[i]|
open    arm i is open once all its parents have c >= 0.5, and closes again below 0.4
        (full -> key_points -> three_point, upper_body)
w       proportional to p over the open arms; then at least 0.05 per open arm and 0.2 for the anchor (full)
shares  <- 0.8 shares + 0.2 w; a world takes its arm when its next episode starts
```

- Absolute progress, so that forgetting attracts practice. Progress, not failure, so that an arm that cannot be
  learned does not absorb worlds.
- The allocator never reads the held-out split; that is for reporting only.
- The constants above are set once and kept for every problem. That is the claim of H3.

Test (one, on synthetic competence curves): an arm that never improves ends at the floor; a rising arm gets the
largest share; a falling arm regains share; a closed arm has none.

Runs, three seeds at 2000 M each (13 variants, about 40 GPU-hours):

| Runs | Purpose |
|---|---|
| Allocator on the evaluation signal | H3, against the best of W1 |
| The same rule on the training-world statistics | H4 |
| Shares proportional to failure, 1 - c | The clip samplers' rule, lifted to arms |
| TSCL and ALP on training returns, a PLR-style value-loss score | Curriculum baselines |
| Without the graph, the floor, the anchor; 1%, 3%, 12% evaluation worlds; update every 4 rows | Ablations |

Done when the summary table holds them and Gate 2's condition is checked on this problem.

### W4. Harder arms and the other problems

Start while W1 runs. This is where the literature reports that fixed mixtures fail, and where a flat Gate 1 sends
us.

1. **More interface arms** (task configuration only): `root_path` (root position only); `upper_body` with the root
   velocity if its specialist fails; the six fall-and-get-up clips back in as a hard clip group.
2. **Robustness arms (problem 3).** Pushes and randomised dynamics switched per world by a state that the learner
   writes, as `shares` is today. Clip and phase bins as arms, from the failure statistics of `reset_to_clips`.
3. **Hybrid imitation without a pre-stage (problem 2).** Tracking worlds and goal-plus-style worlds from scratch,
   on the 28-DoF humanoid with HIL's heading task (`mdp2/hil`) and on the G1. Fixed splits run today
   (`MDP2EnvCfg.task_worlds`). Changing the split during training needs one of:
   - quick: every term in every world, with reward weights per world (they are state already) and a per-world arm
     flag for terminations and observations;
   - clean: the world's configuration as state that takes effect when its episode ends. This is a change to the
     MDP2 runtime and goes to the pull request (section 9).

   Do the quick one for the science and the clean one for the systems measurements.

For each problem: specialists, uniform, a sweep of fixed mixtures, hand schedules, two-stage where it applies, then
the allocator with unchanged constants.

### W5. Transfer probes

From checkpoints at several times of a uniform run, train short bursts (50 M samples) with 0.3 extra share on one
arm, plus a control burst with unchanged shares, and measure the change of every arm's competence against the
control. The result is a matrix (arm trained, arm measured) over training time: H5, and the check that the
allocator's cheap signal agrees with measured transfer. Needs W2.2 and W2.3. A burst per arm and a control, at six
times, for three seeds: 90 short runs with four arms, a few GPU-hours.

### W6. Scale

PHUMA (73 h, Apache-2.0) for the scale result and for anything released. Convert to the format of the clips (30
fps; root position, root quaternion, 29 joint positions), check for frames in the ground, split by sequence into
training, validation and test. Clips are stored on the GPU: estimate the memory before loading 73 h, and start with
10 h. If one GPU is too slow at this scale, average gradients over several processes; decide then.

### W7. Systems measurements

Step time against the number of distinct arms (1 to 64 and beyond) and against a simulation with one arm; fixed
against changing membership; the cost of the evaluation worlds; the same mixture written with the MDP1 managers
(every term in every world, masked). Idle GPU, one GPU model, three repeats.

### W8. Hardware

Randomised dynamics (friction, masses, motor strength, delays), observation noise, sim-to-sim in MuJoCo, export of
the policy. The deployment stack and who runs the robot are open (section 10).

### W9. Writing

Abstract (4 December, if Gate 1 passes): the concept figure, the table of W1 with the allocator's row, shares over
training, a first transfer matrix. Paper: the study with at least 3 seeds on the G1 and 8 on the humanoid,
interquartile means with bootstrap intervals, samples and GPU-hours for every result, hardware.

## 8. Schedule

| Dates | Work |
|---|---|
| 13 to 16 Nov | W0; launch W1; W2 |
| 17 to 25 Nov | W3 implemented and running; W4.1 and W4.2 built; W1 at 8000 M |
| 26 Nov | **Gate 1** |
| 27 Nov to 4 Dec | Allocator against the sweep on problem 1; abstract |
| 5 to 17 Dec | Ablations, W5, W4.3. **Gate 2** on 17 December |
| 18 Dec to 31 Jan | W6, the full study with seeds, W7, sim-to-sim |
| Feb to mid Mar | W8 on the G1 |
| Mid Mar to 16 Apr | Figures, video, writing |

## 9. Working rules

- MDP2 runs on Newton only: the `newton_mjwarp` physics preset and the Newton Warp renderer. No Isaac Sim, and a
  task configuration must load without importing `pxr`, `omni`, `carb` or `isaacsim`.
- Cartpole is not a benchmark; do not report it.
- Changes to the MDP2 runtime (`source/isaaclab_experimental/isaaclab_experimental/mdp2/`, `envs/mdp2_*.py`) are
  committed separately from tasks and learners and cherry-picked onto `feature/mdp2`, the branch of pull request
  isaac-sim/IsaacLab#9016. Tasks, learners and this file stay on `feature/mdp2-examples`.
- Code style: plain code, no helpers that wrap one line, short docstrings, names that RL and Isaac Lab users know.
- Commits: imperative capitalised subject, no attribution lines; `uvx ruff format` and `uvx ruff check` first;
  tests with `uv run --extra test python -m pytest`.
- Data: LAFAN1-derived clips are for evaluation and development only. Release only on PHUMA and OmniRetarget (MIT).
- Numbers of other papers quoted in this project were extracted by reading agents; check each against the PDF
  before it is cited.
- Record what did not work in the log, with the numbers.

## 10. Open questions

| Question | Needed for |
|---|---|
| How jobs are launched on the cluster (scheduler, shared file system) | The study with seeds (from 18 December); W1 to W5 fit on four GPUs |
| Who runs the G1, and with which deployment stack | W8 |
| Where this branch may be pushed | Sharing between machines |

## Status

| Item | Status | Notes |
|---|---|---|
| Pilot on the workstation (1000 M, one seed) | DOING | 4 of 8 runs done, section 4 |
| W0 Bring-up on the multi-GPU machine | TODO | |
| W1 Gate 1 batch | TODO | |
| W2 Tooling | TODO | |
| W3 Allocator, version 1 | TODO | |
| W4 Harder arms, problems 2 and 3 | TODO | |
| W5 Transfer probes | TODO | |
| W6 Scale | TODO | |
| W7 Systems measurements | TODO | |
| W8 Hardware | TODO | |
| W9 Abstract and paper | TODO | |
| Gate 1 | 26 Nov | |
| Gate 2 | 17 Dec | |

## Log

- 2026-11-12: task, learner and test committed. Pilot batch of eight runs started on the workstation (RTX PRO 6000
  Blackwell, 0.56 M samples/s); `full`, `uniform`, `three_point` and the schedule done, see section 4. This plan
  written.
