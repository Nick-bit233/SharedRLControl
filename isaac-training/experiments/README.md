# Experiments Directory

Unified experiment entrypoints live at the top of this directory. Historical
per-experiment scripts are compatibility shims only.

## Structure
- Use `experiments/train.py` for training.
- Use `experiments/eval.py` for checkpoint evaluation.
- Use `experiments/launch.py` for foreground/tmux/dry-run process launch.
- Use `experiments/campaign.py` for staged training orchestration.


## run command example
```bash
cd SharedRLControl/isaac-training/
python experiments/train.py experiment=tunnel wandb.mode=disabled
```
for wandb logging, set `wandb.mode=online` or `wandb.mode=offline`
