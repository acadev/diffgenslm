#!/bin/bash
# =============================================================================
# Polaris (ALCF) — Phase 0: tokenize genomes to HDF5
# PBS Professional job script
#
# Submit: qsub scripts/polaris_preprocess.sh
# =============================================================================
#PBS -N diffgenslm_preprocess
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q preemptable
#PBS -A <YOUR_PROJECT_HERE>

set -e
cd "${PBS_O_WORKDIR}"

# ── Modules ──────────────────────────────────────────────────────────────────
module load conda/2024-04-29
conda activate diffgenslm   # activate your environment

# ── Paths (edit these) ───────────────────────────────────────────────────────
GTO_DIR=/eagle/projects/<project>/streptomyces/gto_files
FASTA_DIR=/eagle/projects/<project>/streptomyces/Streptomyces_genomes
GFF_DIR=/eagle/projects/<project>/streptomyces/gff_files
TOKENIZER_DIR=/eagle/projects/<project>/diffgenslm/tokenizer
OUTPUT_DIR=/eagle/projects/<project>/diffgenslm/hdf5

NNODES=$(wc -l < "${PBS_NODEFILE}")
NRANKS_PER_NODE=4     # 4 CPUs per Polaris node used here (I/O bound task)
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

echo "Nodes: ${NNODES}  Ranks/node: ${NRANKS_PER_NODE}  Total: ${NTOTRANKS}"

# ── Step 1: Extract BPE training sequences (rank 0 only) ─────────────────────
if [ ! -f "${OUTPUT_DIR}/functional_seqs.txt" ]; then
    echo "=== Extracting sequences for BPE training ==="
    python -m diffgenslm.preprocessing.extract_for_bpe \
        --gto_dir    "${GTO_DIR}" \
        --fasta_dir  "${FASTA_DIR}" \
        --gff_dir    "${GFF_DIR}" \
        --out_functional "${OUTPUT_DIR}/functional_seqs.txt" \
        --out_noncoding  "${OUTPUT_DIR}/noncoding_seqs.txt"
fi

# ── Step 2: Create codon vocabulary (deterministic) ──────────────────────────
python -c "
from diffgenslm.tokenizer import create_codon_vocab
create_codon_vocab('${TOKENIZER_DIR}')
"

# ── Step 3: Train BPE tokenizers ─────────────────────────────────────────────
if [ ! -f "${TOKENIZER_DIR}/functional_bpe.model" ]; then
    echo "=== Training BPE tokenizers ==="
    python -m diffgenslm.preprocessing.train_bpe \
        --functional_input "${OUTPUT_DIR}/functional_seqs.txt" \
        --noncoding_input  "${OUTPUT_DIR}/noncoding_seqs.txt" \
        --output_dir       "${TOKENIZER_DIR}" \
        --functional_vocab 4096 \
        --noncoding_vocab  8192
fi

# ── Step 4: Tokenize genomes to HDF5 (MPI parallel) ─────────────────────────
echo "=== Building HDF5 datasets (${NTOTRANKS} MPI ranks) ==="
mpiexec -n "${NTOTRANKS}" \
    --ppn "${NRANKS_PER_NODE}" \
    --depth 8 \
    --cpu-bind depth \
    python -m diffgenslm.preprocessing.build_hdf5 \
        --gto_dir       "${GTO_DIR}" \
        --fasta_dir     "${FASTA_DIR}" \
        --gff_dir       "${GFF_DIR}" \
        --tokenizer_dir "${TOKENIZER_DIR}" \
        --output_dir    "${OUTPUT_DIR}" \
        --train_frac    0.8 \
        --val_frac      0.1 \
        --min_contig_len 500

echo "=== Preprocessing complete ==="
