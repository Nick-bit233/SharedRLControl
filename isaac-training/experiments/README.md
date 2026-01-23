# Experiments Directory

Place your isolated experiment scripts here.

## Structure
- Each experiment gets its own folder (e.g., `01_simple_baseline`).
- Use `train.py` as the entry point for each experiment if they differ significantly in logic.
- If they share logic, refer to `src/` for shared components.


## run command example
```bash
cd SharedRLControl/isaac-training/
python experiments/02_residual_policy/train.py experiment=residual_policy wandb.mode=disabled
```
for wandb logging, set `wandb.mode=online` or `wandb.mode=offline`
