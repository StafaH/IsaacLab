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

Seven reading tracks, well over a hundred papers. Every text is public. The references at the end of this section
give the title and the link of each, which is all that is needed to read them again.

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

### References

Titles and identifiers are those of the pages that were fetched. Rows marked with * were not saved as text: their
identifiers come from the reading agents' reports and were confirmed against the title of the arXiv page on
2026-11-12. An arXiv number N stands for `https://arxiv.org/abs/N`.

**Hybrid and adversarial imitation.**

| Short name | Title | Link |
|---|---|---|
| HIL | HIL: Hybrid Imitation Learning for Dynamic Athletic Control | [2505.12619](https://arxiv.org/abs/2505.12619), [project page](https://xbpeng.github.io/projects/HIL/index.html) |
| Wang et al. (RSS 2026) | Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework | [2602.20375](https://arxiv.org/abs/2602.20375) |
| DeepMimic | DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills | [1804.02717](https://arxiv.org/abs/1804.02717) |
| AMP | AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control | [2104.02180](https://arxiv.org/abs/2104.02180) |
| AMP on robots | Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions | [2203.15103](https://arxiv.org/abs/2203.15103) |
| ASE | ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters | [2205.01906](https://arxiv.org/abs/2205.01906) |
| CALM | CALM: Conditional Adversarial Latent Models for Directable Virtual Characters | [2305.02195](https://arxiv.org/abs/2305.02195) |
| ADD | Physics-Based Motion Imitation with Adversarial Differential Discriminators | [2505.04961](https://arxiv.org/abs/2505.04961) |
| SMP | SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control | [2512.03028](https://arxiv.org/abs/2512.03028) |
| Composite | Composite Motion Learning with Task Control | [2305.03286](https://arxiv.org/abs/2305.03286) |
| PMP | PMP: Learning to Physically Interact with Environments using Part-wise Motion Priors | [2305.03249](https://arxiv.org/abs/2305.03249) |
| State-dependent AMP | Unified Walking, Running, and Recovery for Humanoids via State-Dependent Adversarial Motion Priors | [2605.18611](https://arxiv.org/abs/2605.18611) |
| Wen et al. | Constrained Style Learning from Imperfect Demonstrations under Task Optimality | [2507.09371](https://arxiv.org/abs/2507.09371) |
| Lee et al. 2021 | Learning a family of motor skills from a single motion clip | [doi 10.1145/3450626.3459774](https://doi.org/10.1145/3450626.3459774) |
| MimicKit | MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control | [2510.13794](https://arxiv.org/abs/2510.13794) |

**Masked and multi-interface control, distillation, teacher and student.**

| Short name | Title | Link |
|---|---|---|
| MaskedMimic | MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting | [2409.14393](https://arxiv.org/abs/2409.14393) |
| MaskedManipulator | MaskedManipulator: Versatile Whole-Body Manipulation | [2505.19086](https://arxiv.org/abs/2505.19086) |
| HOVER | HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots | [2410.21229](https://arxiv.org/abs/2410.21229) |
| MHC | Learning Multi-Modal Whole-Body Control for Real-World Humanoid Robots | [2408.07295](https://arxiv.org/abs/2408.07295) |
| PHC | Perpetual Humanoid Control for Real-time Simulated Avatars | [2305.06456](https://arxiv.org/abs/2305.06456) |
| PULSE | Universal Humanoid Motion Representations for Physics-Based Control | [2310.04582](https://arxiv.org/abs/2310.04582) |
| PADL | PADL: Language-Directed Physics-Based Character Control | [2301.13868](https://arxiv.org/abs/2301.13868) |
| SuperPADL | SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation | [2407.10481](https://arxiv.org/abs/2407.10481) |
| H2O | Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation | [2403.04436](https://arxiv.org/abs/2403.04436) |
| OmniH2O | OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning | [2406.08858](https://arxiv.org/abs/2406.08858) |
| ExBody2 | ExBody2: Advanced Expressive Humanoid Whole-Body Control | [2412.13196](https://arxiv.org/abs/2412.13196) |
| TWIST | TWIST: Teleoperated Whole-Body Imitation System | [2505.02833](https://arxiv.org/abs/2505.02833) |
| TWIST2 | TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System | [2511.02832](https://arxiv.org/abs/2511.02832) |
| BFM (the "BFM ablation") | Behavior Foundation Model for Humanoid Robots | [2509.13780](https://arxiv.org/abs/2509.13780) |
| Scaling BFM | Scaling Behavior Foundation Model for Humanoid Robots | [2607.15163](https://arxiv.org/abs/2607.15163) |
| Meta Motivo | Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models | [2504.11054](https://arxiv.org/abs/2504.11054) |
| BFM-Zero | BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning | [2511.04131](https://arxiv.org/abs/2511.04131) |
| BumbleBee | From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots | [2506.12779](https://arxiv.org/abs/2506.12779) |
| EAGLE | Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control | [2602.02960](https://arxiv.org/abs/2602.02960) |
| HANDOFF | HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers | [2606.06493](https://arxiv.org/abs/2606.06493) |
| HoRD | HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation | [2602.04412](https://arxiv.org/abs/2602.04412) |
| AnyBody * | AnyBody: Free-Form Whole-Body Humanoid Control from Arbitrary Keypoint Guidance | [2606.29209](https://arxiv.org/abs/2606.29209) |
| InterPrior * | InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions | [2602.06035](https://arxiv.org/abs/2602.06035) |
| MaskAdapt * | MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters | [2603.29272](https://arxiv.org/abs/2603.29272) |
| CTS | CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion | [2405.10830](https://arxiv.org/abs/2405.10830) |
| L2T | Learn to Teach: Sample-Efficient Privileged Learning for Humanoid Locomotion over Real-World Uneven Terrain | [2402.06783](https://arxiv.org/abs/2402.06783) |
| S-TS * | Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction | [2512.07464](https://arxiv.org/abs/2512.07464) |
| METATRON | No arXiv number was found; an NVIDIA project page by Tessler, Jiang, Otto, Shi, Iqbal, Chechik and Peng | Search by name |
| Athena-WBC | Withdrawn from arXiv | [2607.04837](https://arxiv.org/abs/2607.04837) |

**Motion tracking for humanoids.**

| Short name | Title | Link |
|---|---|---|
| GMT | GMT: General Motion Tracking for Humanoid Whole-Body Control | [2506.14770](https://arxiv.org/abs/2506.14770) |
| BeyondMimic | BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion | [2508.08241](https://arxiv.org/abs/2508.08241) |
| SONIC | SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control | [2511.07820](https://arxiv.org/abs/2511.07820) |
| RGMT | Robust and Generalized Humanoid Motion Tracking | [2601.23080](https://arxiv.org/abs/2601.23080) |
| OmniTrack | OmniTrack: General Motion Tracking via Physics-Consistent Reference | [2602.23832](https://arxiv.org/abs/2602.23832) |
| UniTracker | UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots | [2507.07356](https://arxiv.org/abs/2507.07356) |
| Any2Track | Track Any Motions under Any Disturbances | [2509.13833](https://arxiv.org/abs/2509.13833) |
| ASAP | ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills | [2502.01143](https://arxiv.org/abs/2502.01143) |
| KungfuBot | KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills | [2506.12851](https://arxiv.org/abs/2506.12851) |
| AMS | Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data | [2511.17373](https://arxiv.org/abs/2511.17373) |
| Humanoid-GPT | Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking | [2606.03985](https://arxiv.org/abs/2606.03985) |
| GPC | GPC: Large-Scale Generative Pretraining for Transferable Motor Control | [2606.29148](https://arxiv.org/abs/2606.29148) |
| Motion generation and tracking | Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking | [2604.17335](https://arxiv.org/abs/2604.17335) |
| "What matters" | What Matters in Humanoid General Motion Tracking? An Empirical Study | [2607.19903](https://arxiv.org/abs/2607.19903) |
| InstantMimic | InstantMimic: A High Performance System for Learning Physics-based Skills in Seconds | [2609.09821](https://arxiv.org/abs/2609.09821), [code](https://github.com/Scripter36/InstantMimic), [project page](https://scripter36.github.io/projects/instantmimic/) |
| ZEST * | ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control | [2602.00401](https://arxiv.org/abs/2602.00401) |
| SuperMimic * | SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model | [2610.28480](https://arxiv.org/abs/2610.28480) |
| FAST * | General Humanoid Whole-Body Control via Pretraining and Fast Adaptation | [2602.11929](https://arxiv.org/abs/2602.11929) |
| FRoM-W1 * | FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions | [2601.12799](https://arxiv.org/abs/2601.12799) |

**Character control with generative models and scene interaction.**

| Short name | Title | Link |
|---|---|---|
| CLoSD | CLoSD: Closing the Loop between Simulation and Diffusion for Multi-task Character Control | [2410.03441](https://arxiv.org/abs/2410.03441) |
| Diffuse-CLoC | Diffuse-CLoC: Guided Diffusion for Physics-based Character Look-ahead Control | [2503.11801](https://arxiv.org/abs/2503.11801) |
| PDP | PDP: Physics-Based Character Animation via Diffusion Policy | [2406.00960](https://arxiv.org/abs/2406.00960) |
| UniPhys | UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control | [2504.12540](https://arxiv.org/abs/2504.12540) |
| ControlVAE | ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based Characters | [2210.06063](https://arxiv.org/abs/2210.06063) |
| MoConVQ | MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations | [2310.10198](https://arxiv.org/abs/2310.10198) |
| MoCapAct | MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control | [2208.07363](https://arxiv.org/abs/2208.07363) |
| InterMimic | InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions | [2502.20390](https://arxiv.org/abs/2502.20390) |
| TokenHSI | TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization | [2503.19901](https://arxiv.org/abs/2503.19901) |
| PARC | PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers | [2505.04002](https://arxiv.org/abs/2505.04002) |

**Curricula, allocation and robustness.**

| Short name | Title | Link |
|---|---|---|
| ADR * | Solving Rubik's Cube with a Robot Hand | [1910.07113](https://arxiv.org/abs/1910.07113) |
| DORAEMON * | Domain Randomization via Entropy Maximization | [2311.01885](https://arxiv.org/abs/2311.01885) |
| PLR * | Prioritized Level Replay | [2010.03934](https://arxiv.org/abs/2010.03934) |
| ALP-GMM * | Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments | [1910.07224](https://arxiv.org/abs/1910.07224) |
| TSCL * | Teacher-Student Curriculum Learning | [1707.00183](https://arxiv.org/abs/1707.00183) |
| DemoStart * | DemoStart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots | [2409.06613](https://arxiv.org/abs/2409.06613) |
| ERFI * | Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization | [2209.12878](https://arxiv.org/abs/2209.12878) |
| Cha et al. * | Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection | [2504.06585](https://arxiv.org/abs/2504.06585) |

**Simulation with many tasks, and systems.**

| Short name | Title | Link |
|---|---|---|
| MTBench * | Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks | [2507.23172](https://arxiv.org/abs/2507.23172) |
| Hebero and DGPO * (called MT-Libero in our notes) | A GPU-Parallel Framework for Heterogeneous Multi-Task Reinforcement Learning | [2606.03335](https://arxiv.org/abs/2606.03335) |
| Isaac Lab * | Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning | [2511.04831](https://arxiv.org/abs/2511.04831) |

**Data.**

| Short name | Title or content | Link |
|---|---|---|
| LAFAN1 retargeted to the G1 | LAFAN1 Retargeting Dataset (Unitree H1, H1-2 and G1) | [Hugging Face](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) |
| LAFAN1 | Ubisoft La Forge Animation Dataset | [GitHub](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) |
| PHUMA | PHUMA: Physically Reliable Humanoid Locomotion Dataset | [2510.26236](https://arxiv.org/abs/2510.26236) |
| OmniRetarget * | OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction | [2509.26633](https://arxiv.org/abs/2509.26633) |
| HumanX * | HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos | [2602.02473](https://arxiv.org/abs/2602.02473) |
| DemoHLM * | DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation | [2510.11258](https://arxiv.org/abs/2510.11258) |
| AMASS | Archive of Motion Capture as Surface Shapes | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) |

**The other works on the publications page,** <https://xbpeng.github.io/>. Each is at
`https://xbpeng.github.io/projects/<folder>/index.html`, with its PDF in the same folder.

| Folder | Title |
|---|---|
| A1Goalkeeping | Creating a Dynamic Quadrupedal Robotic Goalkeeper with Reinforcement Learning |
| A1SoccerShooting | Hierarchical Reinforcement Learning for Precise Soccer Shooting Skills using a Quadrupedal Robot |
| ActionSpace | Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter? |
| AMDM | Interactive Character Control with Auto-Regressive Motion Diffusion Models |
| AWR | Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning |
| Cassie_Jump | Robust and Versatile Bipedal Jumping Control through Reinforcement Learning |
| Cassie_Locomotion | Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control |
| Cassie_Walking | Reinforcement Learning for Robust Parameterized Locomotion Control of Bipedal Robots |
| CIC | Contrastive Intrinsic Control for Unsupervised Reinforcement Learning |
| CICP | Reinforcement Learning with Competitive Ensembles of Information-Constrained Primitives |
| CondMDI | Flexible Motion In-betweening with Diffusion Models |
| DeepLoco | DeepLoco: Dynamic Locomotion Skills Using Hierarchical Deep Reinforcement Learning |
| DeepMimic | DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills |
| DeepTerrainRL | Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning |
| DiffuseLoco | DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets |
| Diffusion_Humanoid_Manipulation | Generalizable Humanoid Manipulation with 3D Diffusion Policies |
| DMP | DMP: Directable Motion Retargeting through Motion Paraphrasing |
| Finetuning_Locomotion | Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the Real World |
| GenLoco | GenLoco: Generalized Locomotion Controllers for Quadrupedal Robots |
| HiLMa-Res | HiLMa-Res: A General Hierarchical Framework via Residual RL for Combining Quadrupedal Locomotion and Manipulation |
| InterPhys | Synthesizing Physical Character-Scene Interactions |
| Kimodo | Kimodo: Scaling Controllable Human Motion Generation |
| Learn_to_Move | Deep reinforcement learning for modeling human locomotion control in neuromechanical simulation |
| MACAW | Offline Meta-Reinforcement Learning with Advantage Weighting |
| MCP | MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies |
| MotionBricks | MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives |
| MSc_Thesis | Developing Locomotion Skills with Deep Reinforcement Learning |
| PhD_Thesis | Acquiring Motor Skills Through Motion Imitation and Reinforcement Learning |
| RCP | Reward-Conditioned Policies |
| RoboPianist | RoboPianist: Dexterous Piano Playing with Deep Reinforcement Learning |
| RoboSaber | Robo-Saber: Generating and Simulating Virtual Reality Players |
| Robotic_Imitation | Learning Agile Robotic Locomotion Skills by Imitating Animals |
| SFV | SFV: Reinforcement Learning of Physical Skills from Videos |
| SimToReal | Sim-to-Real Transfer of Robotic Control with Dynamics Randomization |
| StableMotion | StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data |
| STMC | Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation |
| SymLoco | On Learning Symmetric Locomotion |
| TerrainRL | Dynamic Terrain Traversal Skills Using Reinforcement Learning |
| TeSMo | Generating Human Interaction Motions in Scenes with Text Control |
| Trace_Pace | Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion |
| Trajeglish | Trajeglish: Traffic Modeling as Next-Token Prediction |
| TWiRL | Learning and Adapting Agile Locomotion Skills by Transferring Experience |
| VDB | Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow |
| Vid2Player3D | Learning Physically Simulated Tennis Skills from Broadcast Videos |
| VIPER | Video Prediction Models as Rewards for Reinforcement Learning |

## Status

| Item | Status | Notes |
|---|---|---|
| Literature read and plan | DONE | 2026-11-12 |
| Decisions in section 11 | DONE | Job launching and hardware stack still open |
| Runtime items | DOING | See section 8 |
| Pilot for Gate 1 | DOING | Task built (`isaaclab_tasks/mdp2/interfaces`, `Isaac-Interfaces-G1-MDP2`); first batch of eight runs at 1 billion samples running |
| Execution plan | DONE | `MDP2_EXECUTION_PLAN.md`, tracked on `feature/mdp2-examples`: bring-up, launcher, work packages W0 to W9, numeric rules for the gates. Results and progress are logged there from now on |

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
- 2026-11-12: pilot stopped at the user's request. Complete: `full`, `three_point`, uniform, schedule (70 / 76 /
  28 / 2% held-out for the schedule). The two-stage student stopped at 101 M; the `key_points` and `upper_body`
  specialists and self-distillation never ran. The commands that reproduce these runs are in the execution plan,
  section 4. The execution plan now also holds the background of this file and the build lessons, and
  `feature/mdp2-examples` is pushed to the fork. Progress is logged in the execution plan from here on.
- 2026-11-12: references added to section 12, with the title and the link of every text that was read.
