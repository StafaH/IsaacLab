# MDP2 research: execution plan

The plan to execute, tracked in the branch `feature/mdp2-examples` so that the work can continue on any machine. It
is written to be enough on its own: section 11 says how to get or reproduce everything the branch does not hold,
and sections 12 and 13 record the background and the lessons so far. `MDP2_RESEARCH_PLAN.md` beside it is the full
research plan it was derived from: the claim, the literature with titles and links, the method and the design of
the experiments.

Work through the packages of section 7 in the order of section 8. Keep **Status** and **Log** at the end current
and commit the file with every update.

The branch is on the owner's fork at the owner's request. It carries an unpublished research plan: do not open a
pull request from it or push it anywhere else without asking.

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
minutes; 6.7 GB of GPU memory per run.

## 4. Results so far

Pilot: seed 0, 1000 million samples per run, protocol P1, on one RTX PRO 6000 Blackwell. These are the runs of W1
at half the budget with one schedule. The pilot was stopped on 12 November: four of eight runs are complete, the
student of the two-stage run was stopped at 101 M, and three runs never started (specialists `key_points` and
`upper_body`, self-distillation). W1 repeats all of them with seeds. Preliminary: one seed, and no run has
converged.

To reproduce the runs below (30 minutes each; the last needs the model of the first):

```bash
T=scripts/reinforcement_learning/mdp2/train_interfaces.py
ALL=full=1,key_points=1,three_point=1,upper_body=1
P=runs/research/pilot && mkdir -p $P
.venv/bin/python $T --budget 1000 --out $P/full.json --shares full=1 --save $P/full.pt
.venv/bin/python $T --budget 1000 --out $P/three_point.json --shares three_point=1
.venv/bin/python $T --budget 1000 --out $P/uniform.json --shares $ALL
.venv/bin/python $T --budget 1000 --out $P/schedule.json --shares full=1 --shares_after $ALL --switch_at 500
.venv/bin/python $T --budget 1000 --out $P/two_stage.json --shares $ALL --teacher $P/full.pt \
    --rl 0 --entropy 0 --init_std 0.1
```

Final held-out success (mean of the last ten log rows), with the mean body error in brackets:

| Run | `full` | `key_points` | `three_point` | `upper_body` |
|---|---|---|---|---|
| Specialist `full` | 70% (68 mm) | | | |
| Specialist `three_point` | | | 69% (83 mm) | |
| Uniform mixture | 48% (79 mm) | 72% (76 mm) | 61% (87 mm) | 3% (128 mm) |
| Schedule: `full`, then uniform from 500 M | 70% (76 mm) | 76% (77 mm) | 28% (100 mm) | 2% (138 mm) |

On the training clips the same runs give 63%; 50%; 58 / 58 / 40 / 6%; 52 / 41 / 24 / 3%. The two splits hold
different clips, so the difference between them is not a generalisation gap by itself.

Held-out success [%] over training, as `full` / `key_points` / `three_point` / `upper_body` (mean of the five rows
up to the mark, so the last column differs a little from the table above):

| Run | 250 M | 500 M | 750 M | 1000 M |
|---|---|---|---|---|
| Specialist `full` | 20 / 0 / 0 / 0 | 40 / 0 / 0 / 0 | 59 / 0 / 0 / 0 | 70 / 0 / 0 / 0 |
| Specialist `three_point` | 0 / 0 / 33 / 0 | 0 / 0 / 61 / 0 | 0 / 0 / 73 / 0 | 0 / 0 / 66 / 0 |
| Uniform mixture | 15 / 22 / 14 / 2 | 39 / 63 / 36 / 2 | 47 / 71 / 44 / 3 | 43 / 69 / 62 / 3 |
| Schedule (switch at 500 M) | 21 / 0 / 0 / 0 | 59 / 0 / 0 / 0 | 57 / 51 / 2 / 0 | 72 / 77 / 29 / 2 |

- **First success.** Held-out success passes 5% at 175 M (`full`) and 111 M (`three_point`) for the specialists, and
  at 175 M and 157 M for the same interfaces in the uniform mixture. Runs shorter than 200 M say nothing.
- **Two-stage distillation, unfinished.** The student imitates the labels of the `full` specialist (the 1000 M run
  above) in every interface. At 101 M of its own samples it has 4 / 8 / 5 / 0% held-out success with errors of 109 /
  107 / 116 / 137 mm, where the uniform mixture with reinforcement learning has 0% everywhere and 113 / 119 / 121 /
  121 mm at the same count. Too early to say anything about its final level.
- **Earlier single runs.** `full` alone reached 32% on training clips and 35% held-out at 295 M in a 300 M run.
  With `--init_std 0.5` it had 10% and 5% at 200 M, slower than the default of 1.0.
- **Noise.** The curve of one run moves by several points between marks (`three_point`: 73, then 66), which is why
  the gates ask for seeds and a separation of 10 points.

Reading so far:

- The uniform mixture does not fail wholesale. With a quarter of the samples on `three_point` it is 8 points behind
  the specialist of that interface, so the interfaces teach each other.
- It pays on `full` (22 points behind the specialist) and it fails `upper_body`. Whether `upper_body` can be learned
  alone is not known: its specialist never ran. Look at that run of W1 first.
- The schedule trades the other way: `full` stays at the specialist's level and `key_points` is the best so far,
  but `three_point` has had 125 M samples of its own and is at 28%, still rising. Neither the uniform mixture nor
  this schedule is better on every interface, which is the situation an allocator is meant for.
- After the switch, the schedule reached 51% on `key_points` within 250 M (62 M samples of that interface): what
  `full` learned carries over to it almost at once, and far less to `three_point`. That asymmetry is H5.
- H1 is open: the two-stage run did not finish. The uniform mixture's numbers make a wholesale loss unlikely on
  this problem, so expect the outcome of Gate 1 to turn on `full`, on `upper_body` and on the fixed-mixture sweep.
- No run has converged: the `full` specialist gained 11 points over its last 250 M. W1 therefore doubles the
  budget and checks the ranking at 8000 M.

## 5. Bring-up on a new machine

1. **Code.** Check out `feature/mdp2-examples` from the owner's fork; it sits on `main` of isaac-sim/IsaacLab. Then
   `uv sync`: the default environment has Newton and no Isaac Sim, which is all this work uses.
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
5. **Machine notes (4 x RTX 6000 Ada, 12 November).**
   - Set `TMPDIR` to a folder of your own (`export TMPDIR=$PWD/runs/tmp`): assets are downloaded to the temporary
     folder, and `/tmp/https` belongs to another user there. Without it every environment fails with a
     `PermissionError`.
   - The kernel module of the NVIDIA driver is 595.84 and the libraries are 595.99, until the machine is rebooted.
     CUDA works; `nvidia-smi` and the RTX renderer (ovrtx) do not. Without root: download
     `NVIDIA-Linux-x86_64-595.84.run` from download.nvidia.com, unpack it with `--extract-only`, link every
     `lib*.so.595.84` by its soname and its bare name into one folder, and put that folder first in
     `LD_LIBRARY_PATH` (here `runs/nvidia/lib`). Training does not need this.
   - Runs with the same seed are not identical: two runs differ from the first log row on.
6. **Reproduce.** `mkdir -p runs/research`, then the same script with `--shares full=1 --budget 1000 --out
   runs/research/repro_full.json`. Expect the `full` row of section 4: no success until about 140 M, held-out
   success near 70% at 1000 M. Seeds and GPUs differ, so within about 10 points. Record the throughput of the
   machine's GPU in the log of this file.

## 6. Running batches

One run per GPU, from a queue. The queue is the file `runs/research/queue.txt`, with one run per line: the
checkout that pins its code, the script (`train_interfaces.py` or `evaluate_interfaces.py`), `<batch>/<name>`, then
the arguments of the script.
`scripts/reinforcement_learning/mdp2/run_queue.sh <GPU>` is a worker: it takes the first line of the file whenever
its GPU is free, until the file is empty, and skips runs that already finished. Start one worker per GPU; edit the
file at any time to add, drop or reorder runs.

```bash
echo "../mdp2-batch train_interfaces.py gate1/uniform_s0 --seed 0 --budget 2000 --shares $ALL" >> runs/research/queue.txt
for g in 0 1 2 3; do
    setsid nohup scripts/reinforcement_learning/mdp2/run_queue.sh $g > runs/research/worker$g.log 2>&1 &
done
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
- **Do not test on a GPU that runs a batch.** Throughput is bound by the GPU: two runs on one GPU get half each,
  and 4096, 8192 and 16384 worlds give the same samples per second.
- The script takes the device `cuda:0`, so `CUDA_VISIBLE_DEVICES` selects the GPU. Training one run on several
  GPUs is not built and not needed before W6.

## 7. Work packages

GPU-hours are for the throughput of section 3.

### W0. Bring-up

Section 5 on the machine that runs the batches. Done when the reproduction matches and its throughput is in the
log.

### W1. Gate 1 batch (no new code)

Protocol P1, 2000 million samples per method in total, seeds 0 to 2. About 44 GPU-hours: 11 hours on four GPUs.
The batch contains every run of the pilot, including the four that did not finish there.

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
   velocity if its specialist fails; the six fall-and-get-up clips back in as a hard clip group (download
   `fallAndGetUp1_subject1`, `fallAndGetUp1_subject4`, `fallAndGetUp1_subject5`, `fallAndGetUp2_subject2`,
   `fallAndGetUp2_subject3` and `fallAndGetUp3_subject1` as in section 5).
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
  isaac-sim/IsaacLab#9016. Tasks, learners and this file stay on `feature/mdp2-examples`. Today the runtime code
  of the two branches is identical.
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

Decided by the owner on 12 November: the venue is RSS 2027; a G1 is available; compute is a 4-GPU machine and a
cluster with hundreds of GPUs; only public datasets are used.

## 11. What is not in the branch, and how to get it

| What | How to get or reproduce it |
|---|---|
| The clips | Download them from the [LAFAN1 retargeting dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) with the command of section 5; never commit them |
| Logs and models of the pilot | Rerun the commands of section 4; W1 reruns them with seeds anyway |
| The texts of the papers | `MDP2_RESEARCH_PLAN.md`, section 12, lists the title and the link of every text that was read |
| The design of the MDP2 runtime | The guide `docs/source/experimental-features/mdp2.rst` in the branch, and pull request [isaac-sim/IsaacLab#9016](https://github.com/isaac-sim/IsaacLab/pull/9016) |
| The showcase results quoted in section 13 | The learners in `scripts/reinforcement_learning/mdp2/`; section 13 names the command behind each number |

## 12. Background from the literature review

Seven reading tracks over the publications at <https://xbpeng.github.io/> and related work, well over a hundred
papers. `MDP2_RESEARCH_PLAN.md`, section 12, gives the title and the link of each; an arXiv number N below stands
for `https://arxiv.org/abs/N`. Numbers were extracted by reading agents, in places through a summarising model:
check each against the PDF before it is cited.

**What the literature establishes.**

| Finding | Evidence |
|---|---|
| Tracking is precise but rigid; adversarial imitation adapts but collapses onto few modes, more so with more data | HIL, AMP, ASE, CALM, SMP |
| Mixing objectives across worlds helps, but only two modes were tried, at a fixed split, after a tracking stage | HIL (arXiv 2505.12619): 4 billion samples of tracking, then 2 billion at 50/50; no ablation of the ratio, no seeds, no wall time; four V100 GPUs |
| Mixing from the start fails on dynamic skills without a curriculum | Wang et al., RSS 2026 (arXiv 2602.20375): success 0 / 0 without their hand-set schedule (imitation probability from 1.0 to 0.5), one policy per behaviour |
| Masked multi-mode control from scratch loses to distillation at modest compute | HOVER (worse on 32 of 32 metrics), the BFM ablation, PULSE (72% / 33% without distillation) |
| It works in one stage only with a hand curriculum on a small library, or at very large compute | MHC (75 clips, three stages), Scaling BFM (arXiv 2607.15163, eight mask modes, up to 64 GPUs), SONIC (128 GPUs for 7 days) |
| A sparse interface with the dense full-reference reward is learnable from scratch | PULSE's three-point baseline (98.8% / 93.4%), SONIC. Our pilot agrees |
| The field is moving from distillation to single-stage training | SONIC, TWIST2, RGMT, Scaling BFM, SuperMimic (model-based, CoRL 2026) |
| Every imitation sampler adapts only which clip or phase to start from, from training failures | PHC, MaskedMimic, BeyondMimic, GMT, SONIC, ZEST |
| Ratios of teacher to student worlds are constants or clocks | CTS (3:1), S-TS (ramp after N iterations); MaskAdapt (arXiv 2603.29272) distils within one network for state masks |
| With two reward structures in one batch the critic must know the mode | HIL (5 times the critic loss without it) |
| Results rest on one to five runs | DeepMimic, HIL, most tracking papers; the one "what matters" study (arXiv 2607.19903) uses at most two seeds |

**The gap.** We found no paper that does any of the following. Confidence is medium to low, because the terminology
is diffuse and OpenReview and Google Scholar were not queried directly.

- Allocates shares of parallel worlds across imitation objectives or levels of supervision from a signal.
- Drives a choice about the training distribution from held-out evaluation inside the training simulation.
- Adapts more than two axes of the training distribution jointly.
- Measures which objective teaches which (HIL's explanation of why mixing works is qualitative).
- Measures the cost of a different MDP per world (multi-task benchmarks such as MTBench do not).

**Method beyond version 1 of the allocator.**

| Axis | Arms |
|---|---|
| Specification | Full-body targets; head and hands; upper body plus root velocity; root path only; keyframes; goal plus style (adversarial, no reference) |
| Clip and phase | Bins of the reference set |
| Perturbation | Clean; pushed; randomised dynamics |
| Scene (stretch) | Reference scene; widened obstacle and goal ranges (OmniRetarget clips) |

- The critic sees the full reference and the arm, with returns normalised per arm.
- Version 2 of the allocator uses the transfer matrices of W5: allocate to the arms with the largest measured gain
  on the arms that matter at deployment.
- Forks: K copies of the policy in one simulation, each with its own worlds. They exist in `train_reward_search.py`
  and are the one-GPU form of the transfer probes and of population-based training.

**Baselines the paper needs,** all in our runtime at equal samples and equal wall time.

| Baseline | Stands for |
|---|---|
| Uniform fixed mixture | HOVER's from-scratch baseline, Scaling BFM's mode sampling |
| Best fixed mixture from a grid | What a sweep buys |
| Hand schedules | HIL (tracking, then 50/50), Wang et al. (linear to 0.5), MHC (three stages), S-TS (ramp after N iterations) |
| Two-stage distillation | HOVER, MaskedMimic |
| Fixed-ratio concurrent teacher and student | CTS (3:1) |
| Clip and phase sampler only | BeyondMimic, GMT |
| Curriculum allocators on training signals | TSCL, ALP, PLR-style value-loss scores |
| Population-based training over the shares | K policies in one simulation, K times the cost |

**Metrics.** Per arm on held-out clips: success at 0.5 m and at 0.25 m, global and local keypoint error in mm; for
the hybrid problem HIL's direction, facing and skill-fidelity measures. Curves against samples and wall time.

**Data** (public sets only).

| Set | Size | Licence | Use |
|---|---|---|---|
| [LAFAN1 retargeted to the G1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | 40 clips, 2.45 h | Non-commercial, not redistributable | Development and the standard comparison |
| PHUMA ([arXiv 2510.26236](https://arxiv.org/abs/2510.26236)) | 73 h | Apache-2.0 | Scale, and anything released |
| OmniRetarget ([arXiv 2509.26633](https://arxiv.org/abs/2509.26633)) | 4 h | MIT | The scene problem |
| [AMASS](https://amass.is.tue.mpg.de/) retargets | | CC-BY-4.0 plus the AMASS licence | Alternative for scale |

**Other deadlines.** ICML 22 January 2027; SIGGRAPH 28 January 2027 (simulation only, gives up the hardware
result); CoRL 2027 not announced (28 May in 2026).

**Risks.**

| Risk | Response |
|---|---|
| Uniform mixing simply works at our scale | Gate 1, then the harder problems of W4, then the study |
| The allocator only equals a good hand schedule | Claim one rule across problems against a sweep per problem |
| Groups with far more compute are one step away | Move fast; our ground is allocation, seeds and one GPU per run |
| Small data and small networks look unconvincing | PHUMA, standard metrics, sim-to-sim against released checkpoints, hardware |
| The discriminator destabilises mixed training | Gradient penalty; a frozen score prior as the style arm |
| Reviewers see population-based training or meta-gradients under a new name | Both are baselines; the transfer matrices show the signal is different |

## 13. Lessons from the build so far

- **The research task is not the showcase tracker.** That one acts as a correction on the reference pose and
  observes global targets, which hands the full reference to every interface. The task here acts from the default
  pose and its actor sees only what the robot measures plus its interface. Training takes minutes to hours, not
  seconds.
- **Bad frames.** 808 frames of the clips have bodies in the ground (for example in `walk3_subject1`); rewards
  became NaN when a world started there. The loader now lifts such frames (`ground_clearance`) and a world whose
  state is not finite ends its episode. Check new data for this before training on it.
- **Physics limits.** The constraint and contact buffers overflowed when hands and knees touched the ground; the
  task sets `njmax=300` and `nconmax=120`.
- **Shared exploration noise.** The action noise of PPO is one vector for all worlds. In the Go2 mixture of walking
  and standing (`Isaac-Velocity-Flat-WalkStand-UnitreeGo2-MDP2`) a penalty that is sensitive to noise in one arm
  starved exploration in the other. When one arm of a mixture fails, train it alone with the same code before
  suspecting the mixture.
- **Saved models.** On the manipulation showcase a saved model evaluated in a fresh simulation gave 34% and 32%
  where training reported 97% and 98%; the cause was not found. Reproduce with `train_manipulation.py --iterations
  4000 --log_dir <dir>`, then the same script with `--checkpoint <model in dir>`. Check W2.2 and W2.3 against the
  evaluation worlds of the training run before trusting them.
- **MDP2.** A `Quantity` takes `term=`; a quantity that reads another is listed after it. A logged quantity that
  only some world configurations compute used to be averaged over all worlds (fixed in the runtime).
- **Warp.** A variable that starts as a constant and changes inside a loop of dynamic length must be declared with
  a type (`chosen = int(0)`). Do not define kernels in `python -c`.
- **Tests** run with `uv run --extra test`; a plain `uv run` removes pytest from the environment again.
- **Step time of the showcase tasks,** MDP1 managers against MDP2 at 4096 worlds, one measurement each [ms]: push
  recovery 3.89 / 2.34, reward search 4.29 / 2.42, hybrid imitation 4.50 / 2.47, distillation 5.39 / 3.64, skills
  8.01 / 4.11, tracking 3.82 / 2.41. Physics is the floor: the MDP is about 0.07 ms of the 2.41 ms. None has more
  than four arms, which is what W7 has to add. The MDP1 side is `mdp1_baselines.py <recovery | reward_search | hil
  | distillation> 4096`, `skills_mdp1_baseline.py 4096` and `train_tracking.py --runtime mdp1`. The MDP2 side is
  the same loop on the MDP2 environment of each task: 60 warm-up steps, then 300 timed calls of `env.step` with
  random actions.
- **Unsolved cases** that problem 3 can use: the cartwheel that is unstable in the six-clip tracking policy
  (`train_tracking.py --motion walk,run,dance,hop,cartwheel,backflip`), the get-up and handstand skills of the Go2
  (`train_skills.py --skills walk,hind_legs,hind_walk,get_up`; the handstand is defined in the task but not among
  its skills), and the reorientation skill of the Kuka Allegro (`train_manipulation.py --skills
  lift,place,reorient`: 3% success).

## Status

| Item | Status | Notes |
|---|---|---|
| Pilot (1000 M, one seed) | STOPPED | 4 of 8 runs complete, section 4; the rest is part of W1 |
| W0 Bring-up | DONE | 4 x RTX 6000 Ada, 0.29 M samples/s per GPU (half of the pilot's GPU); the reproduction is the `full` run of W1 |
| W1 Gate 1 batch | DOING | Seed 0 running since 12 November, 15 runs, about 7 hours |
| W2 Tooling | DOING | Done: summary (`summarize_interfaces.py`), checkpoints (`--save_every`), final evaluation (`evaluate_interfaces.py`), statistics of the training worlds per interface. Open: resuming a checkpoint, return normalisation per arm |
| W3 Allocator, version 1 | DOING | Built and tested (`interfaces/allocation.py`, `--allocator`), with the baselines; no runs yet |
| W4 Harder arms, problems 2 and 3 | DOING | Pushed worlds as arms built (`Isaac-Interfaces-Robust-G1-MDP2`); no runs yet |
| W5 Transfer probes | TODO | |
| W6 Scale | TODO | |
| W7 Systems measurements | TODO | |
| W8 Hardware | TODO | |
| W9 Abstract and paper | TODO | |
| Gate 1 | 26 Nov | |
| Gate 2 | 17 Dec | |

## Log

- 2026-11-12: task, learner and test committed. Pilot batch of eight runs started on one RTX PRO 6000 Blackwell
  (0.56 M samples/s); `full`, `uniform`, `three_point` and the schedule done, see section 4. This plan written.
- 2026-11-12: pilot stopped by the owner to continue with several GPUs. The two-stage student was at 101 M; the
  specialists `key_points` and `upper_body` and self-distillation had not started. Results, background and lessons
  recorded in sections 4, 12 and 13. Branch pushed to the owner's fork. Next: W0.
- 2026-11-12: references to local files replaced by commands and links. Section 4 has the commands of the pilot,
  section 11 says how to get everything that is not in the branch, section 13 names the command behind each
  showcase number, and `MDP2_RESEARCH_PLAN.md` lists the title and the link of every text that was read.
- 2026-11-12: W0 on a machine with four RTX 6000 Ada. Tests pass (with `TMPDIR` set, section 5). Throughput is
  0.29 M samples/s per GPU with 4096, 8192 or 16384 worlds, and 0.14 each for two runs on one GPU: half of the
  pilot's GPU, so 2000 M samples take 1.9 hours and the four GPUs give about 100 billion samples per day. GPU 0
  runs at 90 degrees and is throttled to about 0.2 M samples/s. Same-seed runs are not identical.
- 2026-11-12: the learner now also logs, per interface and split, the success and the episode length averaged
  over the start frames (`start_success`, `length`), and per interface the reward, episode length, failure share
  and absolute advantage of the training worlds. Reason: `success` is a share of episodes, and a start frame that
  fails after one second ends ten times as many episodes as one that succeeds, so it understates the share of
  start frames that are tracked. `success` stays as the measure of the gates; report `start_success` beside it.
  None of this changes training. W1 seed 0 launched at commit e72952d64 with the teacher taken from the `full` run
  at 1000 M (`--save_every 250`) instead of a separate teacher run.
- 2026-11-12: built while W1 runs: the allocator of W3 as `isaaclab_tasks/mdp2/interfaces/allocation.py` with
  its test and the baselines (failure, TSCL, ALP, ranked advantage), `--allocator` in the learner;
  `evaluate_interfaces.py` (W2.3); pushed worlds as arms (`Isaac-Interfaces-Robust-G1-MDP2`: every interface a
  second time with pushes of up to 1 m/s every 1 to 3 s, switched per world by the arm, W4.2). The RTX renderer
  runs with the libraries of section 5.
- 2026-11-12, evening, seed 0 at 2000 M (success per start frame, 1024 starts per cell with
  `evaluate_interfaces.py`, training clips / held-out clips):
  - Uniform mixture: `full` 94 / 88, `key_points` 94 / 93, `three_point` 92 / 92, `upper_body` 42 / 21.
    Specialists: `key_points` 97 / 91, `upper_body` 92 / 18, `full` 94 / 87 (log, 1410 M). The mixture equals the
    specialists of the first three interfaces with a quarter of their samples; `upper_body` is the only interface
    that wants its own samples, and on held-out clips nobody tracks it (a single root position and the arm joints
    leave the legs open).
  - The last row of every log covered only the iterations after the last full interval and lowered the mean of
    the last ten rows by up to 8 points. The learner no longer writes it and `summarize_interfaces.py` leaves it
    out; the summary reads `start_success` by default.
  - Allocator pilots. With the unlock graph, `key_points` opened at 220 M with a share of 3 to 30% and stayed at
    0% until its share passed 50% at 1550 M, then reached 94% within 200 M; the two interfaces behind it never
    opened. Without the graph the progress rule gave `upper_body` less than a quarter of the worlds until 1000 M
    (it progresses slowest) and half of them by the end: 52% on training clips against 40% for the uniform
    mixture, the rest equal. Consequences: the graph is off by default; the rule `frontier` was added, the mean
    of p (1 - p) over the fixed start frames of the evaluation worlds; `--obs_clip` was added, because an input
    that was constant (the flag of an interface without worlds) is divided by a standard deviation of zero.
    None of this changes runs without the new arguments.
  - Transfer probes (W5): 50 M on one interface alone from checkpoints of the uniform run, against 50 M of the
    uniform mixture. At 500 M, `full` alone keeps `key_points` within 2 points of the control and `three_point`
    within 4; `three_point` alone keeps `key_points` within 1 and loses 7 on `full`; `upper_body` alone gains 6
    on itself and costs the others 10; every other interface alone costs `upper_body` 3 to 14. H5 holds on this
    problem.
  - Pushed worlds (8 arms, uniform, 1000 M): the clean interfaces reach the level of the four-arm uniform run
    with half of its samples per interface; the pushed ones reach 26 to 33% on held-out clips and still rise.
- 2026-11-13, night, seed 0 at 2000 M (success per start frame from the log, training clips / held-out clips, as
  `full`, `key_points`, `three_point`, `upper_body`, then the mean):
  - Problem 1. Uniform 95 94 89 40, 79.4 / 89 95 91 22, 74.4. Rule `headroom` 91 91 88 73, 85.5 / 86 93 87 23,
    72.4. Mixture with 70% `upper_body` 90 89 87 80, 86.5 / 83 89 86 20, 69.7. Two-stage (final evaluation) 86 88
    81 29 / 75 78 73 15. Every method that moves worlds to `upper_body` gains on the training clips and loses 1 to
    5 points on the held-out clips: `upper_body` on training clips is memorized, not learned. On held-out clips the
    methods of problem 1 are within 3 points of the uniform mixture, except the 70% mixture, which is 5 below. Judge
    by the held-out clips from here on.
  - Pushed worlds (8 arms). Uniform: clean 93 98 93 22, pushed 56 66 58 9 on held-out clips. Pushed arms only: clean
    96 97 92 22, pushed 84 89 82 17. The pushed arms teach the clean ones for free, and the mean over the eight
    arms rises from 61.9 to 72.3: this is the problem on which the shares matter.
  - Throughput is the same with 1, 4 or 8 arms and with or without an allocator: 0.285 to 0.288 M samples/s on one
    GPU for a specialist, the rules on four arms and the failure rule on eight.
  - Replaying the rules on the logs of the uniform runs: the gate of `headroom` (progress of 0.01) is at the level
    of the noise of 64 evaluation worlds, so the rule is the failure rule in practice and sends most worlds to
    `upper_body`; `progress` gives the clean arms most worlds for the first 800 M, because they rise fastest.
  - Changes. The floor is now a share that the open arms divide (0.2), not a share per arm: twelve arms kept 60% at
    uniform. `--signal val` reads the evaluation worlds on every second held-out clip (`VAL_CLIPS`), and
    `evaluate_interfaces.py` reports the other held-out clips as `unseen`. New rule `gain`: the competence that an
    arm gains within 100 updates at its present progress, at most what it lacks. Runs with `gain` and the
    validation signal are queued for the three problems, with the same rule on training clips and on rollouts.
  - Literature, checked against the arXiv full texts: Wang et al. (2602.20375) adapt the share of imitation worlds
    from 1.0 to 0.5 with one scalar that follows performance, and report 0% on two skills without it: the closest
    work, and not a hand-set schedule as written before. DeXtreme (2210.13702) keeps 40% of its worlds for
    evaluation in the batch to adapt randomisation ranges. LP-ACRL (2601.17428) and Success Guided Sampling
    (2610.12465) reallocate worlds over terrain, command and reset cells. SONIC: 128 GPUs for 7 days, 611 h of
    motion. Scaling BFM: 32 to 64 GPUs. HIL: 4 billion samples of tracking, then 2 billion at a fixed half and
    half. HOVER reports an RL baseline from scratch that is worse on 32 of 32 metrics. The score p (1 - p) is from
    Tzannetos et al. (2304.12877) and Rutherford et al. (2408.15099).
