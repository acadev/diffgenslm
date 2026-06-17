#!/bin/bash
# =============================================================================
# Aurora (ALCF) — Phase 2: train DiffGenSLM with oneCCL + Intel GPU
#
# Phase 1 uses DDP with --backend ccl (oneCCL).
# Full DeepSpeed ZeRO integration is Phase 2 (add --use_deepspeed flag
# once diffgenslm/deepspeed_config.json is written).
#
# Submit: qsub scripts/aurora_train.sh
# =============================================================================
#PBS -N diffgenslm_aurora
#PBS -l select=8:system=aurora
#PBS -l place=scatter
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -A <YOUR_PROJECT_HERE>

set -e
cd "${PBS_O_WORKDIR}"

module load frameworks/2024.2.1_u1
conda activate diffgenslm

# ── Intel oneCCL / XPU setup ─────────────────────────────────────────────────
source /opt/intel/oneapi/setvars.sh --force
export CCL_WORKER_COUNT=1
export CCL_LOG_LEVEL=info
export TORCH_LLM_ALLREDUCE=1

# ── Paths ─────────────────────────────────────────────────────────────────────
HDF5_DIR=/flare/projects/<project>/diffgenslm/hdf5
SAVE_DIR=/flare/projects/<project>/diffgenslm/checkpoints
CONFIG=diffgenslm/configs/medium.yaml

NNODES=$(wc -l < "${PBS_NODEFILE}")
NRANKS_PER_NODE=12         # 12 Intel GPUs per Aurora node
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

echo "Nodes: ${NNODES}  GPUs/node: ${NRANKS_PER_NODE}  Total ranks: ${NTOTRANKS}"

mpiexec -n "${NTOTRANKS}" \
    --ppn "${NRANKS_PER_NODE}" \
    --cpu-bind list:0-7:8-15:16-23:24-31:32-39:40-47:48-55:56-63:64-71:72-79:80-87:88-95 \
    python -m diffgenslm.train \
        --config    "${CONFIG}" \
        --hdf5_dir  "${HDF5_DIR}" \
        --save_dir  "${SAVE_DIR}" \
        --backend   ccl \
        --bf16 \
        --resume \
        --wandb_project diffgenslm \
        --wandb_run_name "aurora_medium_$(date +%Y%m%d)"

echo "=== Aurora training complete ==="
