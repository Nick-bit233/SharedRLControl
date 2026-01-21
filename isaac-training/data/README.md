# This directory stores pre-generated trajectory datasets for offline training.

# To generate a trajectory dataset, run:
```
python training/scripts/trajectory_generator.py
```

# 1. Generate trajectory dataset (GPU, fast)
```
cd SharedRLControl/isaac-training/training/scripts
python trajectory_generator.py backend=batched num_trajectories=100000
```

# 2. Generate with CPU multiprocessing (fallback)
```
python trajectory_generator.py backend=library num_workers=8
```

# 3. Multi-GPU generation
```
torchrun --nproc_per_node=4 trajectory_generator.py backend=batched
```

# 4. Train with offline trajectories (scaled mode, boundary-aware)
```
python runner_simple.py user_model.offline_mode=true \
    user_model.dataset_path=./data/trajectories.h5 \
    user_model.sampling_mode=scaled
```

# 5. Train with raw sampling (no boundary-aware)
```
python runner_simple.py user_model.offline_mode=true \
    user_model.dataset_path=./data/trajectories.h5 \
    user_model.sampling_mode=raw
```