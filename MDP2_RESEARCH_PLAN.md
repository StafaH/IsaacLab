# Research plan: which worlds to simulate

The research plan behind `MDP2_EXECUTION_PLAN.md`: the claim, the literature, the method and the design of the
experiments. Written 2026-11-12 from a full-text read of the literature (see "Sources" at the end). Progress and
results are logged in the execution plan; change this file when the claim or the design changes.

## 1. The claim in one paragraph

Massively parallel imitation learning fixes the mixture of training worlds by hand: HIL trains tracking alone for
4 billion samples and then splits worlds 50/50; HOVER and MaskedMimic sample masks from fixed distributions and need
a distillation stage; concurrent teacher-student methods fix the ratio at 3:1 or ramp it on a clock. We propose to
treat the mixture as the thing to optimise. Every world runs one of many MDP variants ("arms") that differ in
objective, in how much of the reference the policy is told, in perturbation and in scene. A small share of
evaluation worlds inside the same simulation measures held-out competence per arm, and an allocator moves worlds
between arms from that signal. The paper shows that this makes single-stage training of a multi-interface,
hybrid-imitation controller work on one GPU, where fixed mixtures are reported to fail, and it measures why with
transfer experiments that no prior work has run.

Working title: "Which Worlds to Simulate? Evaluation-Driven Allocation of Heterogeneous Worlds for Single-Stage
Imitation".

## 2. What the literature establishes

| Finding | Evidence |
|---|---|
| Tracking is precise but rigid; adversarial imitation adapts but collapses onto few modes, more so with more data | HIL, AMP, ASE, CALM, SMP |
| Mixing objectives across worlds helps, but only two modes have been tried, at a fixed split, after a tracking pre-stage | HIL (4B + 2B samples, 50/50, no ratio ablation, no seeds, no wall time) |
| Mixing from the start fails on dynamic skills without a curriculum | Wang et al. RSS 2026: success 0 / 0 without their hand-set schedule |
| Masked multi-mode control from scratch loses to distillation at modest compute | HOVER (worse on 32 of 32 metrics), BFM ablation, PULSE (72% / 33% without distillation) |
| It works in one stage only with a hand curriculum on a small library, or at very large compute | MHC (75 clips, three stages), Scaling BFM (up to 64 GPUs), SONIC (128 GPUs for 7 days) |
| A sparse interface with the dense full-reference reward is learnable from scratch | PULSE's three-point baseline (98.8% / 93.4%), SONIC |
| The field is moving from distillation to single-stage training | SONIC, TWIST2, RGMT, Scaling BFM, SuperMimic |
| Every imitation sampler adapts only which clip or phase to start from, from training failures | PHC, MaskedMimic, BeyondMimic, GMT, SONIC, ZEST |
| With two reward structures in one batch the critic must know the mode | HIL (5x critic loss without it); multi-task benchmarks locate gradient conflict in the critic |
| Results in this area rest on one to five runs | DeepMimic, HIL, most of the tracking literature; the one "what matters" study uses at most two seeds |

## 3. The gap

We found no paper that does any of the following (search caveats in "Sources"):

- allocates shares of parallel worlds across imitation objectives or levels of supervision from a signal;
- drives any training-distribution choice from held-out evaluation inside the training simulation;
- adapts more than two axes of the training distribution jointly;
- measures which objective teaches which (HIL's explanation of why mixing works is qualitative);
- measures the cost of running a different MDP per world, or runs independent policies in one humanoid simulation.

## 4. What we will not claim

- **Fastest tracking.** InstantMimic (SIGGRAPH Asia 2026) reports 1.5 to 4.5 s per clip, about 0.9 to 1.9 s in its
  repository, on an RTX 5090 with the same class of stack. Our 2.4 to 5 s is matched or beaten.
- **First captured environment step.** InstantMimic's released code captures the whole step, and Isaac Lab's public
  experimental Warp path partly does.
- **Largest generalist.** SONIC and Scaling BFM train on 500+ hours with 64 to 128 GPUs.

What stays ours is generality: a different MDP per world from a declarative config, at a cost we can measure.

## 5. Method

**Arms.** An arm is a complete MDP variant. The axes:

| Axis | Arms |
|---|---|
| Specification | full-body targets; head and hands; upper body plus root velocity; root path only; keyframes; goal plus style (adversarial, no reference) |
| Clip and phase | bins of the reference set (our failure-driven starts) |
| Perturbation | clean; pushed; randomised dynamics |
| Scene (stretch) | reference scene; widened obstacle and goal ranges |

The policy sees its targets and mask bits. Every arm that has a reference keeps the dense full-reference reward and
only masks what the policy is told. The critic is privileged: it sees the full reference and the arm, with return
normalisation per arm. Exploration noise must not be one shared vector across arms (our own walk-and-stand lesson:
a noise-sensitive penalty in one arm starved exploration in the others).

**Evaluation worlds.** Three to five percent of the worlds, split across arms: deterministic actions, canonical
starts, held-out clips, no perturbation unless the arm is a robustness arm. They give a competence c in [0, 1] per
arm continuously. Prior work pays for separate evaluation passes (PHC evaluates the whole dataset between stages;
DemoStart discards its probe episodes).

**Allocator, version 1.** Every few seconds:

1. Priority of an unlocked arm is its absolute learning progress, the change of its smoothed competence. Absolute,
   so that forgetting attracts practice; progress rather than failure, so that impossible arms do not absorb worlds.
2. Every arm keeps a floor share, and designated anchor arms (full tracking) a minimum share.
3. Arms form a graph of relaxations (full targets, then sparse targets, then goal only; clean, then pushed). An arm
   unlocks when its parents pass a competence threshold, with hysteresis.
4. Shares change slowly, and a world changes arm only when its episode ends.

**Transfer probes.** The policy is forked into K copies that share one simulation. Each copy trains for a short
burst with extra worlds on one arm, and the evaluation worlds measure the change in every arm's competence. The
result is a K by K transfer matrix over training time. It is an analysis in its own right, it validates the cheap
signal of version 1, and it gives version 2 of the allocator: allocate to the arms with the largest measured gain on
the arms that matter at deployment.

**Self-distillation (component).** In masked-specification worlds the full reference is present, so the policy's own
full-information action on the same state is a supervised target for its masked mode. MaskAdapt does this for state
masks, and CTS and SONIC in latent form; none replaces a teacher stage for target masks. Ablated on and off.

## 6. Hypotheses and decision gates

| | Hypothesis | Decided by |
|---|---|---|
| H1 | At 10 billion samples or fewer, a uniform fixed mixture from scratch loses to two-stage distillation at equal samples | Gate 1 |
| H2 | Some schedule from teacher-heavy to relaxed closes most of that gap | Gate 1 |
| H3 | The allocator matches the best mixture or schedule found by a sweep, in one run, without per-problem tuning, on at least three problems | Gate 2 |
| H4 | Held-out in-simulation signals beat training-rollout signals (biased by exploration noise, start sampling and perturbations) | Gate 2 |
| H5 | Transfer between arms is asymmetric and changes over training (full tracking helps sparse modes early, less later) | Analysis |
| H6 | Tracking plus adversarial worlds from scratch fails with a fixed split and trains with allocation | Gate 2 |

- **Gate 1 (26 November).** If a uniform mixture already matches distillation at our scale, H1 is false. The paper
  then becomes the large-scale study (section 9, fallback A) and the allocator a secondary result.
- **Gate 2 (17 December).** If the allocator does not match the best sweep result on at least two problems, keep
  the study and the transfer analysis as the paper and report the allocator as a negative result.

## 7. Experiments

**Problems.**

1. **Multi-interface tracking on the Unitree G1.** Arms: full body, head and hands, upper body plus velocity, root
   only. First on the 40 LAFAN1 clips (2.4 h), then on about 30 h (PHUMA or an AMASS retarget), with a held-out split.
2. **Hybrid imitation without the pre-stage.** Tracking plus goal-and-style worlds, on the 28-DoF humanoid with
   HIL's heading task and metrics, and on the G1.
3. **Robustness and the long tail.** Share of perturbed worlds, and clip and phase bins, as arms. Our unresolved
   cases (the unstable cartwheel in the six-clip policy, the get-up skill) are test cases.
4. **Stretch: expansion from demonstrations.** Platform and box clips of the G1 (OmniRetarget) with per-dimension
   widening of scene parameters and a tracking anchor.

**7a. The G1 task has to be rebuilt for this.** The showcase tracking task acts as a residual on the reference
joint positions and observes global targets and the base velocity. That leaks the full reference into every
interface and is not deployable. The paper's task follows the convention of the literature (BeyondMimic, GMT):
actions offset from the default pose; targets relative to an anchor body; an actor that sees only what the robot
measures plus its commanded interface; a privileged critic; pose-error termination; and randomised dynamics for the
hardware stage. Expect training times of minutes to hours, not the seconds of the showcase.

**Baselines, all in our runtime at equal samples and equal wall time.**

| Baseline | Stands for |
|---|---|
| Uniform fixed mixture | HOVER's from-scratch baseline, Scaling BFM's mode sampling |
| Best fixed mixture from a grid | what a sweep buys |
| Hand schedules | HIL (tracking, then 50/50), Wang et al. (linear to 0.5), MHC (three stages), S-TS (ramp after N iterations) |
| Two-stage distillation | HOVER, MaskedMimic |
| Fixed-ratio concurrent teacher-student | CTS (3:1) |
| Clip and phase sampler only | BeyondMimic, GMT |
| Curriculum allocators on training signals | TSCL, ALP, PLR-style value-loss scores |
| Population-based training over shares | K policies in one simulation, K times the cost |

**Ablations.** No unlock graph; no floor; no anchor; failure versus progress signal; share of evaluation worlds;
reallocation period; critic without the arm, with the arm, with one head per arm; self-distillation on and off.

**The study.** Fixed mixtures, schedules and the allocator, on every problem, with at least 8 seeds at humanoid
scale and 3 at G1 scale. Report interquartile means with bootstrap intervals and performance profiles.

**Metrics.** Per arm on held-out clips: success at 0.5 m and at 0.25 m, global and local keypoint error in mm, HIL's
direction, facing and skill-fidelity measures for the hybrid problem. Curves against samples and wall time.
Sim-to-sim in MuJoCo against released checkpoints where they exist. Samples and GPU-hours for every result.

**Systems measurements.** Step time with 1 to 64+ distinct arms against a homogeneous simulation; static against
changing membership; cost of evaluation worlds; the same task mix on MDP1 managers.

**Compute.** Development and single runs on one GPU: every reported result should train on one GPU, which is
the contrast with the 64 to 128 GPUs of the single-stage generalists. Sweeps, seeds and the scale-up run in
parallel on the 4-GPU machine and the cluster, one run per GPU. How jobs are launched there is still to be agreed.

**Data (public sets only).** LAFAN1 retargeted to the G1 (40 clips, 2.4 h) for development and as the standard
comparison set; PHUMA (73 h, Apache-2.0) for scale and for anything released; OmniRetarget (4 h, MIT) for the
stretch problem. LAFAN1 itself is non-commercial and not redistributable, so it is used for evaluation only.

## 8. Runtime work

Core changes go to the MDP2 pull request; learners and tasks stay on the examples branch.

| Item | State today |
|---|---|
| Change a world's arm at the end of an episode | Done without a core change for arms that differ in what the policy is told: the interface is a command drawn at reset from a shared `shares` state that the learner writes. Arms with different terms (objective, perturbation) still need writable membership: today they are fixed ranges set at build time |
| Evaluation worlds as a first-class setting | Done in the pilot task: per-world `fixed` interface and `fixed_start` frame states, set by the learner |
| Forks: K learners in one simulation, each with its own worlds | Exists in the reward-search learner; needs generalising |
| Metrics per arm | Fixed on 12 November (logged quantities are averaged over their own worlds) |
| Learner: privileged critic with the arm and per-arm normalisation, larger networks, distillation baseline, self-distillation loss | Privileged critic, imitation loss (teacher labels or the policy's own full-target action) and 1024-512-256 networks are in `ppo.py` and `train_interfaces.py`; per-arm return normalisation is not built |
| Overhead benchmark for heterogeneous worlds | Not built; six demos measured at up to four arms |

## 9. Venue and schedule

**Target: RSS 2027** (confirmed by the user on 2026-11-12), now that a G1 is available. A six-page abstract is due 4 December 2026 and the
full paper 16 April 2027, which leaves time for hardware results. The direct precedent (Wang et al.) was an RSS 2026
paper. The abstract is written only if Gate 1 passes.

| Dates | Work |
|---|---|
| 12 to 26 Nov | Public data; a literature-standard G1 tracking task with interface arms (section 7a); learner with a privileged critic; pilot of specialists, uniform mixture, hand schedule and two-stage baseline. **Gate 1** |
| 27 Nov to 4 Dec | Allocator version 1 on the pilot problem; RSS abstract |
| 5 Dec to 17 Dec | Ablations, transfer probes, hybrid problem. **Gate 2** |
| 18 Dec to 31 Jan | Scale to PHUMA on several GPUs, baselines at scale, the full study with seeds, systems measurements, MuJoCo sim-to-sim |
| Feb to mid Mar | Randomised dynamics, hardware on the G1 (run by the team), stretch problem |
| Mid Mar to 16 Apr | Figures, video, writing, buffer |

**Alternatives.** If the abstract is not written or not invited: CoRL 2027 (not announced; 28 May in 2026) with the
same hardware plan. SIGGRAPH 2027 (28 January) only as a simulated result, which gives up the hardware advantage.

**Fallbacks.**

- **A. The study.** "What matters in physics-based motion imitation": objectives, mixtures, schedules, critic
  designs and samplers across characters and clip difficulty, thousands of runs with seeds. The one existing study
  varies one factor at a time with runs of about 25 hours and at most two seeds.
- **B. Test-time specialisation.** Fine-tune a generalist on one out-of-distribution generated motion in seconds.
  The concept is published (FRoM-W1, UniTracker, FAST); a stated time budget and a system-level evaluation are open.

## 10. Risks

| Risk | Response |
|---|---|
| Uniform mixing simply works at our scale | Gate 1; fallback A |
| The allocator only equals a good hand schedule | Claim one rule across problems against a sweep per problem; otherwise fallback A with the transfer analysis |
| Well-resourced groups are one step away (HIL's authors, SuperMimic, NVIDIA's METATRON) | Move fast; our distinct ground is allocation, seeds and one GPU, not scale |
| Small data and small networks look unconvincing | PHUMA at 73 h on several GPUs, standard metrics, sim-to-sim against released checkpoints, hardware |
| The discriminator destabilises mixed training | Gradient penalty and the VDB bottleneck; a frozen score prior (SMP) as the alternative style arm |
| Licences | LAFAN1-derived data is non-commercial and not redistributable; release only on PHUMA (Apache-2.0) and OmniRetarget (MIT) |
| Reviewers see population-based training or meta-gradients under a new name | Include both as baselines; the transfer matrices are the evidence that the signal is different |

## 11. Decisions

| Question | Answer (user, 2026-11-12) |
|---|---|
| Hardware | A G1 is available |
| Compute | Hundreds of GPUs, and a 4-GPU machine |
| Data | Public sets |

Still open: how to launch jobs on the other GPUs; who runs the robot and with which deployment stack.

## 12. Sources

Seven reading tracks, well over a hundred papers. The texts of what was read are in a session directory on the
first workstation, not in the repository.

- **Read in full:** HIL (with appendix), Wang et al. RSS 2026, InstantMimic (paper and repository), MaskedMimic,
  PHC, PULSE, SuperPADL, SONIC, HOVER, GMT, BeyondMimic, MHC, TWIST, TWIST2, Meta Motivo, BFM-Zero, Scaling BFM, CTS,
  DemoStart.
- **Method, experiments and limitations read:** ADD, SMP, AMP, ASE, CALM, GPC, MaskedManipulator, PARC, CLoSD, the
  curriculum papers (ADR, DORAEMON, PLR, ALP-GMM, TSCL), and the 45 other works on Peng's page (each at least
  abstract, method and limitations; many in full).
- **Abstract or secondary source only:** SuperMimic's arXiv number, METATRON (project page only), Athena-WBC
  (withdrawn), several 2026 follow-ups named by the prior-art track.
- **Caveats.** Numbers were extracted by reading agents, in places through a summarising model; check each against
  the PDF before citing it. Two tracks disagree on SONIC's held-out success (98.5% and 99.6%). The prior-art search
  did not query OpenReview or Google Scholar directly, and its confidence that evaluation-driven allocation is open
  is medium to low because the terminology is diffuse.

## Status

| Item | Status | Notes |
|---|---|---|
| Literature read and plan | DONE | 2026-11-12 |
| Decisions in section 11 | DONE | Job launching and hardware stack still open |
| Runtime items | DOING | See section 8 |
| Pilot for Gate 1 | DOING | Task built (`isaaclab_tasks/mdp2/interfaces`, `Isaac-Interfaces-G1-MDP2`); first batch of eight runs at 1 billion samples running |
| Execution plan | DONE | `MDP2_EXECUTION_PLAN.md`, tracked on `feature/mdp2-examples` for the multi-GPU machine: bring-up, launcher, work packages W0 to W9, numeric rules for the gates. Results and progress are logged there from now on |

## Log

- 2026-11-12: plan written. Candidate ideas checked against prior art: single-stage masked control, concurrent
  self-distillation and expansion from demonstrations each have a 2026 paper one step away; specialist farms,
  test-time fine-tuning and mixed perturbation exist in substance; evaluation-driven allocation had no direct hit.
- 2026-11-12: user answered the decisions. Venue moved to RSS 2027 (abstract 4 December if Gate 1 passes). The G1
  task is to be rebuilt in the literature's convention before the pilot.
- 2026-11-12: pilot task built and committed on the examples branch. 34 LAFAN1 clips (25 for training, 9 held out
  by subject; the six fall-and-get-up clips left out), four interfaces (full, key points, head and hands, upper
  body), actions offset from the default pose, targets 1, 5 and 15 steps ahead, 512 evaluation worlds inside the
  simulation. What did not work at first: 808 frames of the clips put bodies in the ground and the simulation
  diverged when a world started there (frames now lifted, diverged worlds end their episode). First result, full
  interface alone: 32% of training-clip and 35% of held-out evaluation episodes track for 10 s after 295 million
  samples (8.6 min at 0.57 M steps/s), still rising; halving the initial exploration noise was slower. Gate 1 batch
  running: four specialists, uniform mixture, full-then-uniform schedule, two-stage distillation, self-distillation,
  1 billion samples each, one seed.
- 2026-11-12: first pilot numbers (held-out success, mean of the last ten log rows, one seed, 1000 M). Specialists:
  `full` 70%, `three_point` 69%. Uniform mixture: 48% / 72% / 61% / 3% on full / key points / three point / upper
  body. So uniform mixing does not fail wholesale here: it is 8 points behind the three-point specialist with a
  quarter of that interface's samples, pays 22 points on `full`, and fails `upper_body`. H1 is in doubt on this
  problem; the execution plan therefore defines a "flat" outcome of Gate 1 that moves to harder arms and to the
  hybrid problem, and adds a sweep of fixed mixtures to measure the headroom an allocator has. Execution plan
  written and committed (`MDP2_EXECUTION_PLAN.md`).
- 2026-11-12: pilot stopped at the user's request; experiments continue on the multi-GPU machine. Complete:
  `full`, `three_point`, uniform, schedule (70 / 76 / 28 / 2% held-out for the schedule). The two-stage student
  stopped at 101 M; the `key_points` and `upper_body` specialists and self-distillation never ran. Raw outputs
  copied to `runs/research/pilot/` (ignored by git). The execution plan now also holds the background of this
  file and the build lessons, and `feature/mdp2-examples` is pushed to the fork. Progress is logged in the
  execution plan from here on.
