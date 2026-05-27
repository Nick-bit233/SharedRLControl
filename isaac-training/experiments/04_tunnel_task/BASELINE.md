# Tunnel Sprint Optimization — Frozen Baseline

This file pins the reference baseline against which every M1 / M2 / M3
sweep run is compared. Do **not** modify the values here without also
re-recording the source.

## Paper-current best policy (M0)

- Checkpoint (deployed):
  `SharedRLControl/ros1/navigation_runner/cfg/tunnel/checkpoint_best.pt`
- Trained with the historical staged tunnel curriculum; new staged runs should use
  `experiments/campaign.py campaign=tunnel_curriculum`
  (tunnel_stage1 → stage2 → stage3, last stage activates `reg_coeff`
  curriculum from 0.01 → 0.05).
- Distribution: Beta (`algo.distribution=beta`).
- Online tunnel pilot model: `src/core/user_model_tunnely.py`
  (forward-biased vx + Perlin vy, fixed-direction).

## Reference numbers (from `ana_docs/experiments/ros1_batch_20260417_082217_report.md`)

| metric          | RL (ours, current best) | IPC baseline |
|-----------------|------------------------:|-------------:|
| success rate    |                  54.0 % |       41.4 % |
| collision rate  |                  13.6 % |        0.0 % |
| timeout rate    |                  32.2 % |       58.6 % |

## Validation protocol used by every sprint milestone

1. **Training-side eval** (cheap, automatic, run by `train.py`):
   - Metric key: `eval/stats_success` from `evaluate()` in `train.py`.
   - Triggered every `eval_interval` iterations.
   - `checkpoint_best.pt` is auto-snapshotted on each new best.
   - Same env config as the run itself (no held-out distribution).

2. **External validation** (used to update headline numbers):
   - Run the ROS1 batch comparison pipeline that produced the
     `ros1_batch_20260417_082217_report.md` report.
   - Use the **same scenario set, seed list, and IPC baseline** as that
     report so numbers are directly comparable.

## Comparison rules

- All sprint sweeps **disable** `curriculum.enable` and `early_stopping`
  so reg_coeff or pilot distribution is the only varied factor.
- All sprint sweeps share the same `env` config from `tunnel.yaml`
  (50 obstacles, same map size).
- Ours-vs-Schaff/Walter framing: the **paper-current best** above plays
  the role of the Schaff/Walter-style "with reg_loss" mainline. Any M1
  no-reg / tiny-reg variant that meets the M1 decision threshold (see
  `plan.md`) is allowed to replace it as the new mainline.
