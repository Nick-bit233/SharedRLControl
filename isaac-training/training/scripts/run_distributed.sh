#!/bin/bash
# =============================================================================
# Isaac Sim RL Training Launcher
# =============================================================================
#
# IMPORTANT: Isaac Sim does NOT support CUDA_VISIBLE_DEVICES for multi-GPU!
# Isaac Sim must see all GPUs and select one via internal configuration.
#
# For faster training, use these approaches:
#   1. Increase num_envs (more parallel environments on single GPU)
#   2. Use a specific GPU: ./run_single_gpu.sh <gpu_id>
#
# Usage:
#   ./run_distributed.sh              # Run on GPU 0 with default settings
#   ./run_distributed.sh 3            # Run on GPU 3
#   ./run_distributed.sh 0 1024       # Run on GPU 0 with 1024 environments
#
# =============================================================================

set -e

GPU_ID=${1:-0}
NUM_ENVS=${2:-256}
EXTRA_ARGS=${3:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
LOG_DIR="$SCRIPT_DIR/../outputs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"

echo "=========================================="
echo " Isaac Sim RL Training Launcher"
echo "=========================================="
echo " GPU ID: $GPU_ID"
echo " Num Envs: $NUM_ENVS"
echo " Log file: $LOG_FILE"
echo "=========================================="

# Run training with specified GPU (via Isaac Sim internal config, NOT CUDA_VISIBLE_DEVICES)
python runner_simple.py \
    sim.device="cuda:${GPU_ID}" \
    device="cuda:${GPU_ID}" \
    env.num_envs=$NUM_ENVS \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed. Log saved to: $LOG_FILE"


