#!/bin/bash
# =============================================================================
# Polaris (ALCF) — Phase 1/2: train DiffGenSLM
# 4 nodes × 4 A100 GPUs = 16 ranks total
#
# Submit: qsub scripts/polaris_train.sh
# =============================================================================
#PBS -N diffgenslm_train
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q prod
#PBS -A <YOUR_PROJECT_HERE>

set -e
cd "${PBS_O_WORKDIR}"

module load conda/2024-04-29
module load nccl
conda activate diffgenslm

# ── Paths (edit these) ───────────────────────────────────────────────────────
HDF5_DIR=/eagle/projects/<project>/diffgenslm/hdf5
SAVE_DIR=/eagle/projects/<project>/diffgenslm/checkpoints
CONFIG=diffgenslm/configs/small.yaml   # switch to medium.yaml for Phase 2

NNODES=$(wc -l < "${PBS_NODEFILE}")
NRANKS_PER_NODE=4          # 4 GPUs per Polaris node
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
NDEPTH=8

echo "Nodes: ${NNODES}  GPUs/node: ${NRANKS_PER_NODE}  Total ranks: ${NTOTRANKS}"

# NCCL tuning for Polaris InfiniBand
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=0

mpiexec -n "${NTOTRANKS}" \
    --ppn "${NRANKS_PER_NODE}" \
    --depth "${NDEPTH}" \
    --cpu-bind depth \
    -env OMP_NUM_THREADS="${NDEPTH}" \
    python -m diffgenslm.train \
        --config    "${CONFIG}" \
        --hdf5_dir  "${HDF5_DIR}" \
        --save_dir  "${SAVE_DIR}" \
        --backend   nccl \
        --bf16 \
        --resume \
        --wandb_project diffgenslm \
        --wandb_run_name "polaris_small_$(date +%Y%m%d)"

echo "=== Training complete ==="
