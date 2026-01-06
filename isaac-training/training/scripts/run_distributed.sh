#!/bin/bash
# Multi-GPU Distributed Training Launcher for Isaac Sim RL
#
# Usage:
#   ./run_distributed.sh                    # Use all available GPUs
#   ./run_distributed.sh 4                  # Use 4 GPUs
#   ./run_distributed.sh 4 "extra_args"     # Use 4 GPUs with extra arguments

set -e

# Get number of GPUs to use (default: all available)
NUM_GPUS=${1:-$(nvidia-smi --list-gpus | wc -l)}
EXTRA_ARGS=${2:-""}

echo "=========================================="
echo " Multi-GPU Distributed Training Launcher"
echo "=========================================="
echo " Number of GPUs: $NUM_GPUS"
echo " Extra args: $EXTRA_ARGS"
echo "=========================================="

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Launch distributed training with torchrun
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    runner_simple_distributed.py $EXTRA_ARGS
