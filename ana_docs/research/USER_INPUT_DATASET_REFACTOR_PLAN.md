# User Input Dataset Refactor Plan

## Goal

Unify simulated pilot inputs around an offline dataset workflow:

1. Generate user input trajectories with an explicit generator backend.
2. Store all generator outputs in a compatible HDF5 schema.
3. Train from offline datasets instead of instantiating online user models in the training loop.
4. Compare the input distributions from supported generators with one visualization command.

The v1 scope covers three generator families:

- `legacy_perlin`: generic Perlin/APF/filter pilot input.
- `tunnel_perlin`: forward-biased tunnel pilot input, matching the current tunnel training dataset.
- `intent_pilot`: perception/intent/reactive pilot rollout generated offline.

The following are intentionally out of v1 scope and should not be used as future training-mainline input sources:

- `simple_constant`
- `safety_shield_diverse`

## Dataset Schema

Every dataset must keep the existing training-compatible fields:

```text
/velocities      float32 (N, T, 3)  body-frame human_action
/positions       float32 (N, T, 3)  integrated reference positions
/bboxes          float32 (N, 6)     per-trajectory position bounds
/styles/*        float32 (N,)       optional style metadata
/metadata attrs
```

The metadata group should include:

```text
schema_version = 2
generator_kind = legacy_perlin | tunnel_perlin | intent_pilot
action_frame = body
env_family = tunnel
dt
action_dim
reference_map_bounds
requires_env_geom
requires_assistant_action
```

Intent datasets may add optional diagnostics:

```text
/intent/intent_velocity      float32 (N, T, 3)
/intent/intent_mode          int64   (N, T)
/intent/react_mode           int64   (N, T)
/intent/threat               float32 (N, T)
/intent/perceived_dist       float32 (N, T)
/intent/critic_privileged    float32 (N, T, P) optional
```

Training code should depend only on `/velocities` for v1. Visualizers may read optional groups when present.

## Implementation Order

1. Extend the HDF5 writer to accept optional metadata attrs and optional extra groups while preserving the old schema.
2. Add `configs/user_input/*.yaml` for the three supported generator families.
3. Add generator dispatch to `trajectory_generator.py` through:

   ```yaml
   generator:
     kind: legacy_perlin | tunnel_perlin | intent_pilot
   ```

4. Keep legacy Perlin and tunnel Perlin backed by the existing GPU/CPU trajectory generation logic.
5. Generate `intent_pilot` through a lightweight tunnel rollout using `UserModelIntent` and fixed `assistant_policy=zero`.
6. Add `compare_user_input_datasets.py` to compare multiple HDF5 datasets with one command.
7. Validate with small CPU smoke datasets, schema checks, and Python compilation.

## Expected Commands

Small smoke generation:

```bash
python src/datasets/trajectory_generator.py --config-name user_input/legacy_perlin num_trajectories=16 trajectory_length=128 device=cpu backend=batched
python src/datasets/trajectory_generator.py --config-name user_input/tunnel_perlin num_trajectories=16 trajectory_length=128 device=cpu backend=batched
python src/datasets/trajectory_generator.py --config-name user_input/intent_pilot num_trajectories=16 trajectory_length=128 device=cpu backend=batched
```

Dataset comparison:

```bash
python src/datasets/compare_user_input_datasets.py \
  --dataset legacy=data/user_inputs/legacy_perlin_v1.h5 \
  --dataset tunnel=data/user_inputs/tunnel_perlin_v1.h5 \
  --dataset intent=data/user_inputs/intent_pilot_v1.h5 \
  --out-dir outputs/user_input_compare/tunnel_inputs_v1
```

## Acceptance Criteria

- All three generator configs produce valid HDF5 files.
- Each HDF5 file contains `/velocities`, `/positions`, `/bboxes`, and `/metadata`.
- `metadata.generator_kind` correctly identifies the generator.
- `TrajectoryDataset.sample_raw()` and `sample_scaled()` still work on generated datasets.
- The comparison script writes all expected figures plus `summary.json` and `summary.md`.
- `tunnel_perlin` has stronger forward progress than `legacy_perlin`.
- `intent_pilot` writes non-empty intent/reactive diagnostics.
