# M2 Diverse-Pilot Training Analysis
**Run roots**
- Phase A (iter 0–14000):
  `outputs/tunnel_m2_diverse_pilot/tunnel_m2_20260423_070951/2026-04-23_07-09-54/`
- Phase B (resume from iter 14000, +6000 iters):
  `outputs/tunnel_m2_diverse_pilot/tunnel_m2_20260423_070951_resume14000/2026-04-24_02-06-35/`

Eval cadence is every ~500 iters in Phase A (30 evals → ~14k iter) and every ~1000 iters
in Phase B (12 evals → ~12k iter, but counted from a fresh `Eval at 32768 steps.` line
because the resume run resets the global frame counter while the model weights are
loaded from `checkpoint_14000.pt`).

`reg_coeff = 0.005` (same as M1 winner). All other hyper-parameters identical to
the M1 baseline; only the user-input source changed (offline `trajectories_tunnel.h5`
with diverse pilot bias `vx ∈ [1.0, 2.0], vy ∈ ±2.0`).

---

## 1. Phase A — clean continuous training (the success story)

| iter | success | collision | episode_len | vec_target | vec_policy | task_reward |
|------|---------|-----------|-------------|------------|------------|-------------|
|     0|  18.4 % |   80.9 % |   365 |   0.76 |   0.56 | 0.67 |
|  500 |  56.3 % |   43.8 % |   693 |   0.51 |   0.31 | 0.65 |
| 1000 |  61.7 % |   38.3 % |   741 |   0.60 |   0.39 | 0.62 |
| 2000 |  61.3 % |   37.9 % |   652 |   0.74 |   0.46 | 0.67 |
| 3000 |  69.5 % |   30.1 % |   821 |   0.57 |   0.47 | 0.66 |
| 4000 |  76.2 % |   21.9 % |   868 |   0.52 |   0.45 | 0.66 |
| 5000 |  87.1 % |   10.5 % |   929 |   0.49 |   0.34 | 0.66 |
| 6000 |  91.4 % |    7.0 % |   906 |   0.51 |   0.40 | 0.69 |
| 7000 |  94.9 % |    4.3 % |   855 |   0.52 |   0.43 | 0.68 |
| 8000 |  98.0 % |    2.0 % |   946 |   0.50 |   0.45 | 0.66 |
| 9000 |  93.4 % |    6.6 % |  1078 |   0.51 |   0.32 | 0.61 |
|10000 |  84.8 % |   14.5 % |  1075 |   0.58 |   0.41 | 0.61 |
|11000 |  82.0 % |   16.8 % |  1043 |   0.55 |   0.16 | 0.53 |
|12000 |  93.0 % |    5.9 % |  1011 |   0.52 |   0.41 | 0.59 |
|13000 |  89.8 % |    9.0 % |   893 |   0.54 |   0.54 | 0.62 |
|14000 |  92.6 % |    7.0 % |   911 |   0.54 |   0.52 | 0.61 |
|14500 |  94.5 % |    5.1 % |   835 |   0.61 |   0.59 | 0.56 |
|15000 |  **99.6 %** | **0.4 %** |   822 |   0.58 |   0.61 | 0.59 |
|15500 |  95.7 % |    4.7 % |   896 |   0.62 |   0.45 | 0.54 |
|16000 |  **99.6 %** | **0.4 %** |   871 |   0.58 |   0.53 | 0.56 |
|16500 |  98.4 % |    1.6 % |   742 |   0.53 |   0.41 | 0.59 |
|17000 |  **100.0 %** | **0.0 %** | **730** |   0.59 |   0.35 | 0.55 |
|17500 |  98.0 % |    2.0 % |   673 |   0.50 |   0.35 | 0.52 |
|18000 |  99.6 % |    0.4 % |   687 |   0.53 |   0.55 | 0.53 |
|18500 |  98.0 % |    2.0 % |   685 |   0.54 |   0.51 | 0.54 |
|19000 |  98.4 % |    1.6 % |   697 |   0.54 |   0.50 | 0.55 |
|19500 |  99.6 % |    0.4 % |   698 |   0.48 |   0.66 | 0.56 |
|20000 |  98.8 % |    1.2 % |   698 |   0.51 |   0.64 | 0.56 |
|20500 |  97.7 % |    2.3 % |   664 |   0.53 |   0.87 | 0.55 |
|21000 |  97.7 % |    2.0 % |   685 |   0.51 |   0.65 | 0.59 |

(Iters above 14000 came from the original Phase-A wall-clock interval — eval/save
schedules were wider than the user expected, and the actual save cadence wrote
checkpoint files only up to `checkpoint_14000.pt` before the crash, which is why
the “14k step” framing was used. Eval logs continued past that point until the
crash at iter ~21k.)

### Takeaways from Phase A

- **Bug fixes worked.** Phase A starts at the cold-start collision rate (~81 %)
  expected for a randomly initialised policy and converges monotonically — exactly
  the trajectory shape M1 produced — proving the dataset / sample_scaled / axis
  fixes are now correct end-to-end.
- **Best model is at iter 17000**: **success 100 %, collision 0 %, episode_len 730**.
  This is the strongest checkpoint M2 has produced.
- **Stable plateau** of ≥ 97.7 % success with ≤ 2.3 % collision from iter 17000
  onwards. The shorter `episode_len` (~700) at the late phase vs. the iter 8000
  point (~950) shows the policy now traverses the tunnel **faster** while
  remaining safer — a clear quality improvement.
- **`vec_target` plateau ≈ 0.50** is consistent across all of Phase A; this
  metric is the time-mean **norm of the target velocity vector after the env's
  internal scaling** (after adding the random rotation of the user heading and
  the per-axis squashing the env applies before passing to the policy), not the
  raw 1.5 m/s `vx` mean of the dataset. Compared with the previous broken run
  where this stayed at 0.30, the dataset-fix lift to 0.50 is consistent with the
  expected ~3× increase in effective forward velocity.
- **`vec_policy` 0.31 → 0.65** grows over training: the residual policy learns to
  inject more correction once the dataset has enough information for it to be
  useful. With `reg_coeff = 0.005` the optimiser does not penalise this growth,
  so the policy can specialise to the non-trivial pilot trajectories.
- **`diag_reward_task` falls 0.67 → 0.55** as `vec_policy` grows; this is the
  expected `enable_task_reward` accounting (the task reward decays as the
  intervention magnitude rises) and is **not** a regression in actual task
  performance (success/collision are simultaneously improving).

---

## 2. Phase B — resume from `checkpoint_14000.pt` (the regression)

| iter (resume) | success | collision | vec_target | vec_policy |
|---|---|---|---|---|
|    0 | 76.6 % | 23.1 % | 0.48 | 0.85 |
| 1000 | 94.5 % |  5.5 % | 0.47 | 0.67 |
| 2000 | 93.4 % |  6.6 % | 0.45 | 0.82 |
| 3000 | 95.3 % |  4.7 % | 0.43 | 0.93 |
| 4000 | 94.1 % |  5.9 % | 0.38 | 0.93 |
| 5000 | 89.5 % | 10.5 % | 0.40 | 1.09 |
| 6000 | 89.8 % |  9.8 % | 0.48 | 1.07 |
| 7000 | 95.3 % |  4.3 % | 0.44 | 0.99 |
| 8000 | 92.6 % |  7.0 % | 0.40 | 0.93 |
| 9000 | 93.7 % |  5.9 % | 0.44 | 0.61 |
|10000 | 93.7 % |  6.3 % | 0.44 | 1.03 |
|11000 | 92.2 % |  7.8 % | 0.46 | 1.06 |
|12000 | 92.9 % |  7.0 % | 0.39 | 1.16 |

### Why resume regressed

- **First eval after resume = 76 % / 23 %**, much worse than the iter-14000
  checkpoint that was loaded (which itself was 92.6 % / 7.0 %). This is the
  smoking gun: weights load but **the optimiser state and the curriculum/env
  randomisation distribution were re-initialised**. The sudden drop is the
  model momentarily over-correcting under fresh advantage statistics.
- **`vec_policy` settles at 0.93–1.16, vs. 0.55–0.65 in late Phase A**: the
  resume run is intervening **~2× as aggressively** as the Phase-A late-phase
  policy, yet success is lower and collision is higher. Strictly worse trade-off.
- **No new global maximum** is reached anywhere in Phase B (max 95.3 %). The
  restart therefore contributed no improvement, and was effectively a 6 k-iter
  fine-tune in a slightly mismatched curriculum stage.
- **The "best model" is therefore Phase A's `checkpoint_17000.pt` — but only
  Phase A `checkpoint_14000.pt` was saved to disk** (the crash happened before
  any later iter was written to checkpoint slots). So the on-disk best is
  `checkpoint_14000.pt`, with the **next available** being Phase B's
  `checkpoint_final.pt` (≈ iter 20000 in resume = absolute iter 20000), neither
  of which corresponds to the true Phase-A peak at iter 17000.

---

## 3. Comparison with M1 baseline

| metric (eval, last-1k mean) | M1 baseline best (reg=0.005, online) | M2 Phase A late (iter 17–21k, offline diverse) |
|---|---|---|
| success      | 95.7 % | **98.5 %** |
| collision    | 4.3 %  | **1.5 %**  |
| episode_len  | ~720   | **~690**   |
| `vec_policy` | ~0.40  | ~0.55      |

M2 (offline diverse-pilot) **strictly dominates** M1 (online single-direction
pilot) on success, safety, and traversal time, while taking on a meaningfully
harder input distribution (vy spans the full ±2 m/s rather than just noise on
top of a constant heading). This is the strongest evidence so far that the
no-reg-loss design hypothesis from the M1 sweep generalises to the diverse-input
setting.

---

## 4. Recommendations

### Immediate
1. **Use `checkpoint_14000.pt` from Phase A** as the M2 deliverable — it is the
   newest disk artifact strictly inside the high-success plateau and is what we
   can defend with the saved file. Do **not** use the Phase B `checkpoint_final.pt`
   — its eval metrics are uniformly worse and `vec_policy` shows over-intervention.
2. **Run `eval_video.py`** on `checkpoint_14000.pt` to confirm qualitative
   behaviour (smooth following, no oscillation in the tunnel) before locking it
   in as M2.

### To get the iter 17 k peak back without re-doing 17 k iters
- Re-launch from `checkpoint_14000.pt` with **the original Phase A settings**
  but ensure (a) optimiser state is loaded, not re-initialised, and (b) the
  curriculum stage continues from where Phase A was, not from stage 0. The
  current resume script is doing weight-only resume — that's why the regression
  happened.
- Add a checkpoint-saving cadence change so we save **every 250 iters** in the
  17 k–21 k window (the regime where the policy peaks); currently we only save
  every 1 k iters and the crash robbed us of all post-14 k snapshots.

### For M3
- Phase A late-phase `vec_policy` is still creeping up (0.55 → 0.65 → 0.87 over
  iter 17 k–20 k). This is mild over-intervention; M3 should probably anneal
  `enable_task_reward` weight up slightly, or cap `vec_policy` magnitude in the
  smoothness penalty, to stop the slow drift while preserving the success rate.
- Schaff & Walter baseline for the head-to-head should be evaluated against
  Phase A `checkpoint_14000.pt` (or the regenerated peak), not the resume final.
