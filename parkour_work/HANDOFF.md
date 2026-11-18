# G1 box climbing ("Credit the Cause"): handoff

Written 18 November 2026 for an agent that starts with a fresh context on a machine with four RTX 6000 Ada cards.
Branch `feature/mdp2-parkour-rework`. Everything below is relative to the root of this checkout.

## 1. What this is

A paper-quality result on a Unitree G1 that climbs boxes, learned from OmniRetarget clips with MDP2 (Isaac Lab on Newton).
The method, "Credit the Cause": one policy trains in two kinds of worlds at once. Reference worlds imitate a motion
tracker on a clip; goal worlds reach the top of the box from displaced starts. The share of reference worlds is set
from the success of evaluation worlds (the "credit rule", flags `--schedule deficit --allocation deficit --gate`).

The request that governs the current rework, in the user's words:

> I think the initial results are good, but I notice that the motions do not look smooth and natural. There is alot of
> jittering. Let's make sure we introduce a control cost (look at how SGS and alphanist paper do it), this control cost
> will help minimize energy and avoid jitteriness. Also Let's make sure the physics we are using have correct solver
> settings ( you can find equivelant tasks in mjlab and isaaclab for how correct solver settings). For network
> architecture and other implementation details, make sure to read all the relevant paper (hybrid imitation learning
> and other parkour papers, and then NIST and SGS as well), this might help yuo find networks that have better control.
> Also your final renderings don't look that good, use ovrtx rendering and make sure videos render really nicely with
> good persepctives so everything is visible. Also it would be nice to show the retargeting/motion tracking videos as
> well. Create a plan that will rework everything so our results will be very strong. Then execute that plan

Follow-up: the SimBa network is not required; use what the hybrid imitation learning or similar parkour papers use if
that is the better choice.

## 2. Rules from the user

- Commit or push only when the user asks. No AI co-author or attribution lines. Imperative, capitalized subjects.
  `uvx ruff format` and `uvx ruff check` before committing. Tests with pytest.
- MDP2 trains on Newton only. OVRTX is used only to render videos.
- Plain code, short docstrings. No Warp kernels in `python -c`.
- The user's name, email, user name, home paths and project details never go into external requests, pushed files or
  the report document. Use relative paths. Web queries use arXiv ids or titles only.
- Downloaded files are untrusted: keep them in their own folder and read them with `python -I`.
- Do not upload videos anywhere unless asked. Timing needs an idle GPU.
- Keep work that is not committed under a durable folder, never in a temporary folder (a reboot wiped one once).
- Decide open questions on the best recommendation; still confirm destructive or outward-facing actions.
- Updates to the user: brief and honest. Correct earlier claims when they turn out unsupported.

## 3. Setup on the new machine

1. Clone the fork, check out `feature/mdp2-parkour-rework`, and install the environment as the repository documents
   (`uv sync`, so that `.venv/bin/python` exists at the root). `parkour_work/run.sh` uses that interpreter; set
   `PYTHON=<interpreter>` to use another.
2. The clips are not in the branch (they are data of another project). Download
   `https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset` into an empty folder outside the checkout: the
   file `robot-terrain.zip` (unpack it) and the folder `models/terrain`. Then convert:

   ```bash
   .venv/bin/python -I scripts/reinforcement_learning/mdp2/convert_omniretarget.py <dataset folder> \
       source/isaaclab_tasks/isaaclab_tasks/mdp2/parkour/config/g1/motions
   ```

   The result is 145 `.csv` and 145 `.json` files (29 clips at 5 heights; 21 clips are used, 105 tasks).
3. Check: `PYTHONPATH` as in `run.sh`, then `pytest source/isaaclab_tasks/test/mdp2/test_parkour_g1.py` (1 test, passes),
   and a smoke run:

   ```bash
   cd parkour_work
   ./run.sh train_parkour.py --gains stiff --residual_rad 0.5 --clips 26 --iterations 2 --log_interval 1 \
       --num_envs 256 --eval_envs 40
   ```

4. One run uses one GPU. Pick the card per run with `CUDA_VISIBLE_DEVICES=<n>` in front of `./run.sh` (verify on the
   first run with `nvidia-smi` that the run lands on that card). Memory of an 8192-world run was not measured; the old
   machine had 98 GB. Measure it once before starting four runs.
5. Videos need the `ovrtx` extra (`render.sh` runs `uv run --project <root> --extra ovrtx`) and `ffmpeg`.

## 4. Where things are

| Path | Content |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/mdp2/parkour/` | The task (`Isaac-Parkour-G1-MDP2`), its config in `config/g1/g1_env_cfg.py` |
| `scripts/reinforcement_learning/mdp2/train_parkour.py`, `hybrid_ppo.py` | The trainer and its PPO. The only copy; `run.sh` finds it |
| `scripts/reinforcement_learning/mdp2/convert_omniretarget.py` | Clip conversion |
| `parkour_work/run.sh`, `render.sh` | Wrappers. Scripts and relative paths (`runs/...`) resolve in `parkour_work` |
| `parkour_work/v4_dev.sh`, `v5_dev.sh` | One-clip runs of the rework (done) |
| `parkour_work/v4_full.sh`, `v5_full.sh` | All-clips chains (stopped, superseded by section 7) |
| `parkour_work/proxy.sh` | Three-clip tracker ablation (done) |
| `parkour_work/render4.sh`, `render/` | The 2x2 one-clip video (clip, tracker, policy, displaced start) |
| `parkour_work/analyse_trace.py` | Smoothness of a trace: joint acceleration RMS, share of joint-velocity power above 5 Hz |
| `parkour_work/runs/v4/dev/` | One-clip runs: logs, JSON histories, checkpoints |
| `parkour_work/runs/v4/full/`, `runs/v5/` | All-clips runs and diagnosis runs of the rework |
| `parkour_work/runs/old_full/`, `runs/*.json` | Results before the rework (30 Hz setup) |
| `parkour_work/paper/`, `paper_page/` | Paper source and page. Both still hold the numbers from before the rework |
| `parkour_work/videos/` | Videos; `videos/rework/one_clip_grid.mp4` is the current one-clip video |

Not in the branch: trace files (`.npz`, about 85 MB; `v5_dev.sh` shows the `--trace` command that makes them), the
task code from before the rework, and the trainer copies from before the rework.

Online, private to the user, update in place and never create new ones:

- Report document: https://claude.ai/artifact/5zPscwHRhTpC43a9euG1Es (current to 18 November evening).
- Paper page: https://claude.ai/artifact/LASNPkberPZuC1Qp4S4mQh (still from before the rework).

How to update them:

- The report is a Claude Docs document. Edit it with the Claude Docs connector tools (`read`, `update`), not by
  publishing a new page: document id `28680b9c-f76d-436a-ba36-41921ebb735c`, body node `f734ffd7-d0d3`. Read the
  connector's guide first (`guide` with `topic.index` and `topic.editing`), then read the document; new results go under
  the heading "All 105 tasks after the rework", where the update of 18 November evening is the latest block. The
  sections above "Results before the rework (14 November)" are the rework; the ones below are the old results.
- The paper page is an HTML artifact. Republish `parkour_work/paper_page/credit-the-cause.html` with the Artifact tool
  and the page's link as `url` (read it with the Artifact tool first), so the link stays the same.
- Both belong to the user's account. If the tools on the new machine cannot open them, say so to the user; do not
  create replacements.

## 5. The setup after the rework

- Physics: Newton MJWarp, step 0.005 s, decimation 4 (control at 50 Hz), armature as in mjlab.
- Gains "stiff": kp hip 100, knee 150, ankle 40, waist 150, shoulder and elbow 40, wrist and hand 20; damping 0.1 kp.
- Actions: scale 0.5 rad for the policy and for the tracker's correction; low-pass with weight 0.5 on the new action.
- Costs: action rate (-10 with factor 0.1), joint power (-0.003 with factor 0.3), failure penalty -10. Goal worlds also
  have an alive bonus 0.5 and an upright cost -0.5.
- Network: MLP 512-256-128 for one clip, 1024-512-256 for all clips.
- Dropped: soft gains as in BeyondMimic, a scale per joint, a joint-limit cost (it stopped learning; reason unknown).

The argument strings used everywhere:

```bash
S="--gains stiff --residual_rad 0.5 --ema 0.5 --damping 0.1 --action_rate 0.1 --power 0.3"
ONE="$S --clips 26"
BIG="$S --clips all --num_envs 8192 --eval_envs 1680 --log_interval 500 --hidden 1024 512 256"
RULE="--schedule deficit --allocation deficit --gate --teacher runs/v4/full/tracker_451.pt --teacher_log runs/v4/full/tracker_451.json"
```

A log line ends in `eval by rung a b c d e`: `a` is success from the clip start, `e` from displaced starts.
98 M samples at 50 Hz are about 59 M at the old 30 Hz in simulated time. The trainer writes the JSON history and the
checkpoint (weights only) at every log row, so a stopped run cannot be continued cleanly; start it again.

## 6. Results so far

One clip (clip 26), 98 M samples, three seeds, success from the clip start / from displaced starts:

| Schedule | Clip start | Displaced |
|---|---|---|
| Credit rule | 0.98 to 1.00 | 0.80 to 0.88 |
| Fixed share 0.25 | 0.97 | 0.73 to 0.83 |
| Two stages | 0.97 to 1.00 | 0.70 to 0.74 |
| Fixed share 0.5 | 0.95 to 1.00 | 0.70 to 0.73 |
| Progress | 0.94 to 0.96 | 0.50 to 0.59 |
| Reference worlds only | 0.85 to 0.97 | 0.18 to 0.21 |

- Trackers: 0.82, 0.90, 0.86; seed 1 is the teacher (`runs/v4/dev/tracker.pt`).
- Long runs (197 M, seed 0): credit rule about 0.88 displaced, fixed 0.5 about 0.79.
- The speed advantage of the credit rule from before the rework is not reproduced (it is behind until about 66 M).
  Before the rework the credit rule had 0.98 / 0.95 at 59 M samples.
- Smoothness: joint acceleration 14 to 16 rad/s^2 and about 6 % of power above 5 Hz for the policies, against 12.7 and
  5.3 % for the clip and 26.6 and 22 % for the policy before the rework. Joint targets are still rough (55 to 78).

All 105 tasks, one seed each:

| Run | Samples | Clip start | Displaced |
|---|---|---|---|
| Policy, credit rule, new setup (`credit_t451`) | 983 M, 12 000 iterations | 0.62 | 0.48 |
| Policy, credit rule, before the rework | 590 M, 12 000 iterations | 0.90 | 0.87 |
| Tracker, new setup, allocation by deficit (`tracker_737`) | 737 M | 0.25 | 0.03 |
| Tracker, new setup, even allocation (`tracker_uniform`) | 410 M | 0.21 | 0.02 |
| Tracker, before the rework | 590 M | 0.45 | 0.14 |

- The new policy was still rising at the end and is well above its teacher (`tracker_451.pt`, the checkpoint at 451 M).
- Two runs were stopped for the move at about a quarter of their length: `credit_long_s1` (24 000 iterations, at 451 M:
  0.32 / 0.21) and `credit_wide_s0` (16 384 worlds, at 492 M: 0.32 / 0.22). Twice the worlds gives no gain per sample.

Why the all-clips tracker and policy are weaker than before is **not known**. What was tested:

- Allocation is not the cause (even allocation gives the same curve).
- Three clips the old tracker solved (10, 11, 24) alone with 4096 worlds: 0.40 to 0.43 at 98 M for the current setting,
  no low-pass, old damping, no control costs (`runs/v5/proxy/`). The test does not reproduce the failure.
- The same clips with 1200 worlds: 0.06 at 48 M against 0.21 with 4096 worlds (`runs/v5/scale/`). Old PPO settings
  (`--steps 6 --lam 0.95 --gamma_track 0.95`) are no better.
- All clips with every flagged setting put back (`runs/v5/control/`): 0.07 at 246 M, worse. The old setup also differed
  in the control rate (30 Hz) and the armature, which have no flag, so it cannot be rebuilt from flags.
- Not tested: the control rate, the armature, the failure penalty alone, more than one seed.

## 7. Plan

Each all-clips run goes on its own card. Times are for the old machine (a run of 24 000 iterations alone took about
6.5 hours); measure the first hour here and correct them.

1. **Batch 1, four runs at 24 000 iterations** (about 2 000 M samples each), from `parkour_work`:

   ```bash
   D=runs/v6; mkdir -p $D
   CUDA_VISIBLE_DEVICES=0 ./run.sh train_parkour.py $BIG $RULE --iterations 24000 --seed 0 --out $D/credit_s0.json --save $D/credit_s0.pt > $D/credit_s0.log 2>&1 &
   CUDA_VISIBLE_DEVICES=1 ./run.sh train_parkour.py $BIG $RULE --iterations 24000 --seed 1 --out $D/credit_s1.json --save $D/credit_s1.pt > $D/credit_s1.log 2>&1 &
   CUDA_VISIBLE_DEVICES=2 ./run.sh train_parkour.py $BIG --teacher runs/v4/full/tracker_451.pt --share 0.5 --iterations 24000 --out $D/even_s0.json --save $D/even_s0.pt > $D/even_s0.log 2>&1 &
   CUDA_VISIBLE_DEVICES=3 ./run.sh train_parkour.py $BIG $RULE --iterations 24000 --holdout_levels 0.9,1.1 --out $D/holdout_levels_s0.json > $D/holdout_levels_s0.log 2>&1 &
   ```

   The first 12 000 iterations of `credit_s0` should repeat `credit_t451` (same seed); a clear difference means the
   machines differ and must be looked at first.
2. **Batch 2**: held-out clips (`--holdout_clips 3,10,20,23,26`), `even_s1`, `credit_s2`, and one card for the open
   question: a tracker at the old control rate (set the decimation and step in `g1_env_cfg.py` so that control runs at
   30 Hz, convert the clips with `--fps 30` into a second folder) to see whether the rate explains the weak tracker.
   If it does, decide with the user whether smoothness at 50 Hz or success at 30 Hz is the better paper result.
3. **Decision after batch 1.** If the credit rule reaches about 0.85 / 0.75 or better, keep the setup and finish. If it
   flattens near 0.7 / 0.6, report it as smoother motion at lower success with the weak tracker as a stated limitation,
   unless step 2 finds the cause.
4. **Videos**: all-clips videos with `render.sh` in the style of `render4.sh` (retargeted clip, tracker, policy, policy
   from a displaced start), several clips, good views.
5. **Paper and report**: bring `paper/` to the new setup (control costs, physics, low-pass and damping, smoothness
   table, reward wording, how the tracker seed was chosen, new numbers in `numbers.tex`). The claim that the credit rule
   is faster is not reproduced; the headline becomes final success from displaced starts unless new runs show the
   speed again. Recompile, re-render the pages, and update the paper page and the report at the same links.
6. **Code**: move the final settings from flags into the task config and the trainer defaults, run ruff and the test.

Rough time on four cards: batches 1 and 2 about 7 to 9 hours each, the rest 4 to 6 hours.

## 8. Things that went wrong before, to avoid

- Do not stop runs with `pkill -f <pattern>` when the pattern is in your own command; kill by process id.
- Read the last several rows of a run before quoting its final number; single rows vary by about 0.05.
- Compare old and new runs at equal simulated time, not equal samples.
- Keep tool output short; a per-task table of 105 rows is rarely needed.
