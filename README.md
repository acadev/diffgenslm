# DiffGenSLM

A discrete absorbing-state diffusion language model for whole bacterial genomes. Inspired by LLaDA / Diffusion-LM and the Gemma architecture, adapted for nucleotide-level genomic sequences using biologically-aware tokenization from [GenSLM-2](https://github.com/StarNetLaboratory/GenSLM-2).

Targets ALCF Polaris (NVIDIA A100, NCCL) and Aurora (Intel GPU, oneCCL) via MPI-parallel preprocessing and DDP training.

---

## How it works

**Diffusion process** — At training time, each input sequence has a random fraction `t ~ U(0,1)` of its tokens replaced with `<mask>`. The model (a bidirectional transformer) sees the corrupted sequence and learns to predict every original token at masked positions. The loss is the LLaDA ELBO: `(1/t) × CE` at masked positions, which upweights heavily-masked examples to correct for reduced context.

**Sampling** — At inference, start from a fully-masked sequence. Run the model, commit the highest-confidence predictions, and repeat for `num_steps` iterations (iterative confidence-ranked unmasking). Supports conditional infilling: fix any subset of positions and let the model generate the rest.

**Tokenization** — Three-distribution composite vocabulary:

| Region | Tokenizer | Vocab size |
|---|---|---|
| CDS | CodonTokenizer (frame-aware triplets) | 88 tokens |
| Functional non-coding (tRNA, rRNA) | SentencePiece BPE | 4 096 tokens |
| Intergenic | SentencePiece BPE | 4 096 tokens |

Total small-model vocab: **8 280 tokens** (`88 + 4096 + 4096`).

---

## Repository layout

```
diffgenslm/
├── configs/
│   ├── small.yaml          # ~28M params  (4 096 ctx, Phase 1 dev)
│   └── medium.yaml         # ~220M params (8 192 ctx, Phase 2 scale)
├── preprocessing/
│   ├── genome_records.py   # FeatureRecord / ContigRecord / GenomeRecord dataclasses
│   ├── parse_gto.py        # Primary parser: PATRIC GTO JSON (self-contained)
│   ├── parse_gff_fasta.py  # Fallback parser: GFF3 + FASTA, handles NCBI↔PATRIC ID mismatch
│   ├── extract_for_bpe.py  # Extracts functional + noncoding sequences for BPE training
│   ├── train_bpe.py        # Trains SentencePiece BPE models
│   └── build_hdf5.py       # MPI-parallel genome → HDF5 tokenization pipeline
├── tokenizer/
│   └── __init__.py         # load_tokenizer(), create_codon_vocab()
├── models/
│   └── diffgenome.py       # DiffGenomeModel: bidirectional GQA transformer
├── diffusion/
│   ├── process.py          # forward_process(): absorbing-state masking
│   ├── loss.py             # diffusion_loss() (LLaDA ELBO), simple_loss() (val)
│   └── sample.py           # Iterative confidence-ranked unmasking sampler
├── data/
│   └── genome_dataset.py   # GenomeDiffusionDataset + build_dataloader()
├── eval/
│   └── biolab_model.py     # biolab LM adapter (NuclCharTokenizer + DiffGenSLM)
└── train.py                # DDP training loop (Polaris + Aurora)

tests/
├── conftest.py             # Shared fixtures (tiny model, checkpoint, tokenizer dir)
├── test_model.py           # Architecture tests (RMSNorm, RoPE, GQA, forward shapes)
├── test_diffusion.py       # Diffusion pipeline tests (process, loss, sampler)
├── test_biolab_adapter.py  # biolab LM protocol tests (tokeniser, embeddings, generation)
└── test_preprocessing.py  # Parser + BPE extraction tests (GTO, GFF3, rev-comp)

scripts/
├── polaris_preprocess.sh   # PBS job: Phase 0 data pipeline on Polaris
├── polaris_train.sh        # PBS job: Phase 1/2 training on Polaris (NCCL)
└── aurora_train.sh         # PBS job: Phase 1/2 training on Aurora (oneCCL)
```

---

## Installation

```bash
git clone https://github.com/acadev/diffgenslm.git
cd diffgenslm

# Core dependencies
pip install -e ".[dev]"

# GenSLM-2 composite tokenizer (required for preprocessing)
pip install git+https://github.com/StarNetLaboratory/GenSLM-2.git

# MPI support (install the version matching your cluster's MPI)
pip install mpi4py
```

**Python 3.9+ and PyTorch ≥ 2.1** are required. On Aurora, use the Intel oneAPI frameworks module rather than a pip-installed torch.

For the biolab evaluation harness:

```bash
pip install git+https://github.com/ramanathanlab/biolab.git
```

---

## Phase 0 — Data pipeline

Input data can be in either of two formats:

- **GTO** (preferred): PATRIC/BV-BRC JSON files. Self-contained — they include DNA sequences, annotated features, and the mapping from NCBI accession IDs to PATRIC internal contig IDs. A single `.gto` file per genome.
- **GFF3 + FASTA** (fallback): Standard annotation + sequence files. Note that GFF3 seqnames use NCBI accessions while PATRIC FASTA headers use internal IDs; the parser resolves this via length-based matching when a GTO is not available.

### Step 1 — Extract BPE training sequences

```bash
python -m diffgenslm.preprocessing.extract_for_bpe \
    --gto_dir   /path/to/gto_files \
    --fasta_dir /path/to/fasta_files \
    --gff_dir   /path/to/gff_files \
    --out_functional /tmp/functional_seqs.txt \
    --out_noncoding  /tmp/noncoding_seqs.txt
```

Writes one DNA sequence per line: functional non-coding (tRNA/rRNA) to one file, intergenic regions to another.

### Step 2 — Create the codon vocabulary

```bash
python -c "
from diffgenslm.tokenizer import create_codon_vocab
create_codon_vocab('/path/to/tokenizer_dir')
"
```

Deterministic: 21 special tokens + 4 single-base tokens + 64 codons = 89 entries. Written to `tokenizer_dir/codon/vocab.json`.

### Step 3 — Train BPE tokenizers

```bash
python -m diffgenslm.preprocessing.train_bpe \
    --functional_input /tmp/functional_seqs.txt \
    --noncoding_input  /tmp/noncoding_seqs.txt \
    --output_dir       /path/to/tokenizer_dir \
    --functional_vocab 4096 \
    --noncoding_vocab  8192
```

Outputs `functional_bpe.model` and `noncoding_bpe.model` in the tokenizer directory.

### Step 4 — Build HDF5 datasets (MPI-parallel)

```bash
# Single node
python -m diffgenslm.preprocessing.build_hdf5 \
    --gto_dir       /path/to/gto_files \
    --fasta_dir     /path/to/fasta_files \
    --gff_dir       /path/to/gff_files \
    --tokenizer_dir /path/to/tokenizer_dir \
    --output_dir    /path/to/hdf5_output \
    --train_frac 0.8 --val_frac 0.1

# Multi-node (Polaris example: 4 nodes × 4 ranks)
mpiexec -n 16 --ppn 4 python -m diffgenslm.preprocessing.build_hdf5 [same args]
```

Each rank writes its own shard: `train_rank0000.h5`, `val_rank0000.h5`, etc. The training dataloader picks up all shards automatically.

**HDF5 layout** (compatible with GenSLM-2):
```
input_ids       : int32[N_total_tokens]   # all sequences concatenated
sequence_starts : int64[N_sequences]      # byte offset of each sequence start
```

On Polaris, the full Phase 0 pipeline is wrapped in `scripts/polaris_preprocess.sh`.

---

## Phase 1 — Training

### Single node (development / debugging)

```bash
torchrun --nproc_per_node=4 -m diffgenslm.train \
    --config   diffgenslm/configs/small.yaml \
    --hdf5_dir /path/to/hdf5_output \
    --save_dir checkpoints/small \
    --bf16
```

### Polaris (4 nodes × 4 A100, NCCL)

```bash
qsub scripts/polaris_train.sh
```

Edit `HDF5_DIR`, `SAVE_DIR`, and `PBS -A` in the script first.

### Aurora (8 nodes × 12 Intel GPU, oneCCL)

```bash
qsub scripts/aurora_train.sh
```

Uses `--backend ccl` and sources the Intel oneAPI environment automatically.

### Key training arguments

| Flag | Default | Description |
|---|---|---|
| `--config` | required | Path to YAML config |
| `--hdf5_dir` | required | Directory containing HDF5 shards |
| `--save_dir` | `checkpoints/` | Where to write `checkpoint.pt` and `best.pt` |
| `--resume` | off | Resume from `checkpoint.pt` if it exists |
| `--bf16` | off | bfloat16 (recommended on A100/H100) |
| `--fp16` | off | float16 with GradScaler |
| `--backend` | `nccl` | DDP backend: `nccl` (NVIDIA), `ccl` (Intel), `gloo` (CPU) |
| `--wandb_project` | `diffgenslm` | W&B project name |
| `--wandb_log_freq` | `50` | Log per-step metrics every N steps |
| `--sample_every` | `5` | Generate sample sequences every N epochs (0 = off) |
| `--no_wandb` | off | Disable W&B entirely |

### Checkpoints

Two checkpoint files are maintained in `--save_dir`:
- `checkpoint.pt` — latest completed epoch (used for `--resume`)
- `best.pt` — epoch with lowest validation loss

Both contain `model`, `optimizer`, `scheduler`, `scaler`, `epoch`, `global_step`, and `model_config`.

---

## Model configs

### small (~28M params)

```yaml
model:
  vocab_size: 8280          # 88 codon + 4096 functional BPE + 4096 noncoding BPE
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  num_kv_heads: 4           # GQA: 2 Q heads per KV head
  ffn_intermediate_size: 1366
  max_seq_len: 4096

training:
  batch_size: 8             # per GPU
  lr: 3.0e-4
  epochs: 50
```

### medium (~220M params)

```yaml
model:
  vocab_size: 12376         # 88 + 4096 + 8192
  hidden_size: 1024
  num_layers: 16
  num_heads: 16
  num_kv_heads: 8
  max_seq_len: 8192

training:
  batch_size: 4
  lr: 1.0e-4
  epochs: 30
```

To define a new size, copy either YAML and adjust. The `ffn_intermediate_size` should be `round(8/3 * hidden_size)` (SwiGLU convention).

---

## W&B dashboard

The training loop logs the following metrics automatically (requires `wandb` to be installed and authenticated):

| Metric | Cadence | Description |
|---|---|---|
| `train/loss` | per step | LLaDA ELBO-weighted CE loss |
| `train/token_acc` | per step | Fraction of masked tokens correctly predicted (top-1) |
| `train/t_mean` | per step | Mean diffusion time in the batch |
| `train/grad_norm` | per step | L2 gradient norm (before clipping) |
| `train/lr` | per step | Current learning rate |
| `sys/seqs_per_sec` | per step | Throughput |
| `train/loss_epoch` | per epoch | Epoch-averaged training loss |
| `train/token_acc_epoch` | per epoch | Epoch-averaged token accuracy |
| `val/loss_epoch` | per epoch | Unweighted CE on validation set |
| `val/token_acc_epoch` | per epoch | Validation token accuracy |
| `val/best_loss` | per epoch | Best validation loss so far |
| `train/t_histogram` | per epoch | Histogram of diffusion `t` values (verify masking distribution) |
| `sys/epoch_sec` | per epoch | Wall-clock seconds per epoch |
| `sys/gpu_mem_alloc_gb` | per epoch | Peak GPU memory allocated |
| `samples/generated` | every N epochs | Table of short generated token sequences |

Gradient and parameter histograms are captured via `wandb.watch()` (sampled every `--wandb_log_freq` steps).

---

## Sampling / inference

```python
import torch
from diffgenslm.models.diffgenome import DiffGenomeConfig, DiffGenomeModel
from diffgenslm.diffusion.sample import sample, infill

# Load from checkpoint
ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
cfg  = DiffGenomeConfig(**ckpt["model_config"])
model = DiffGenomeModel(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

# Generate a new sequence (unconditional)
context = torch.full((1, 512), cfg.mask_token_id, dtype=torch.long)
generated = sample(
    model, context, cfg.mask_token_id, cfg.pad_token_id,
    num_steps=64,       # more steps = higher quality
    temperature=1.0,    # set <1 for less diversity, 0 for greedy
    schedule="cosine",  # "linear" or "cosine"
)

# Conditional infilling — fix some positions, generate the rest
context[0, :50] = known_prefix_tokens
filled = infill(model, context, cfg.mask_token_id, num_steps=64)
```

---

## biolab evaluation

DiffGenSLM integrates with the [biolab](https://github.com/ramanathanlab/biolab) benchmark suite via the adapter in `diffgenslm/eval/biolab_model.py`.

### How it works

biolab tasks supply raw DNA strings without GFF annotation. The adapter uses a character-level tokeniser (`NuclCharTokenizer`) that maps each nucleotide A/T/C/G to its single-base token ID in `codon/vocab.json`. This gives `model_encoding='char'`, which biolab routes to:
- sequence-level tasks → `average_pool` transform (mean over token embeddings)
- nucleotide-level tasks → `full_sequence` transform (per-nucleotide embeddings)

`generate_embeddings` extracts the final transformer hidden state (before the LM head); `generate_sequences` runs the iterative confidence-ranked unmasking sampler and decodes token IDs back to a DNA string.

### Running an evaluation

Create a YAML config and run biolab's evaluate entrypoint:

```yaml
# eval_config.yaml
lm_config:
  name: DiffGenSLM
  checkpoint_path: /path/to/checkpoints/best.pt
  tokenizer_dir:   /path/to/tokenizer
  max_length:      2048
  num_sample_steps: 64
  sample_schedule: cosine

task_configs:
  - name: GCContent
    dataset_name_or_path: /path/to/gc_content_dataset
    metrics: [mse, r2]
    task_type: regression

output_dir: results/diffgenslm_gcontent
```

```bash
python -m biolab.evaluate --config eval_config.yaml
```

Results (metrics + model outputs) are written to `output_dir`.

### Registering DiffGenSLM in biolab

To make DiffGenSLM discoverable by `biolab.modeling.get_model`, add two lines to `biolab/modeling/models/__init__.py`:

```python
from diffgenslm.eval.biolab_model import diffgenslm_models   # add this
model_registry = {
    ...
    **diffgenslm_models,     # add this
}
```

Alternatively, use the adapter directly without modifying biolab:

```python
from diffgenslm.eval.biolab_model import DiffGenSLM, DiffGenSLMConfig
from biolab.tasks import get_task
from biolab.tasks.gc_content import GCContentConfig

model = DiffGenSLM(DiffGenSLMConfig(
    checkpoint_path="checkpoints/best.pt",
    tokenizer_dir="tokenizer/",
))
task = get_task(GCContentConfig(dataset_name_or_path="data/gc_content",
                                output_dir="results/", metrics=["mse", "r2"],
                                task_type="regression"))
metrics = task.evaluate(model)
for m in metrics:
    print(m.report())
```

---

## Testing

```bash
# Run the full suite (116 tests, ~2 seconds on CPU)
pytest tests/ -q

# Run a specific module
pytest tests/test_model.py -v
pytest tests/test_diffusion.py -v
pytest tests/test_biolab_adapter.py -v
pytest tests/test_preprocessing.py -v
```

| File | Tests | What's covered |
|---|---|---|
| `test_model.py` | 14 | RMSNorm dtype, RoPE isometry, bidirectionality, weight tying, gradient flow, `return_hidden_states` |
| `test_diffusion.py` | 29 | Forward process masking statistics, ELBO vs. unweighted loss, unmasking schedules, sampler invariants (no residual masks, fixed positions preserved, seed reproducibility) |
| `test_biolab_adapter.py` | 38 | `NuclCharTokenizer` encode/decode/tokenize, config JSON+YAML roundtrip, embedding shapes and dtypes, generated sequences contain no mask tokens, fixed context preserved |
| `test_preprocessing.py` | 35 | GTO coordinate conversion (1-based→0-based, fwd/rev strand), feature type filtering, GFF3 parsing, `build_id_mapping_from_gtos`, reverse-complement extraction |

Fixtures live in `tests/conftest.py`: a session-scoped tiny model (`vocab=64`, `hidden=32`, `layers=2`) and matching checkpoint/tokenizer-dir pair let all tests run without disk I/O beyond a temporary directory.

---

## Architecture overview

`DiffGenomeModel` is a **bidirectional** transformer — there is no causal mask. Every position attends to every other position, which is required for denoising: the model must reason about the full partially-masked context to predict what belongs at each masked site.

Key design choices:

- **Grouped-Query Attention (GQA)** — reduces KV-cache memory by sharing key/value heads across query head groups (`num_kv_heads < num_heads`)
- **RoPE** positional embeddings (ported from [HiSAN](https://github.com/ramanathanlab/HiSAN)) applied to Q and K only
- **RMSNorm** pre-layer normalization (no bias)
- **SwiGLU** feed-forward network (`gate_proj` × `up_proj` → `down_proj`)
- **Tied input/output embeddings** — `lm_head.weight = embed.weight`
- **Weight init**: normal(0, 0.02) for all Linear and Embedding layers

---

## Adding a new genome data source

1. Implement a parser that returns a `GenomeRecord` (see `preprocessing/genome_records.py` for the dataclasses).
2. Add it as a branch in `preprocessing/build_hdf5.py` alongside the existing GTO and GFF/FASTA branches.
3. The rest of the pipeline (`_tokenize_contig`, HDF5 writing, train/val/test splitting) is format-agnostic.

## Adding a new tokenizer distribution

The composite vocabulary concatenates three sub-vocabularies with non-overlapping integer ranges. To add a fourth:

1. Train (or define) the new tokenizer.
2. Update `load_tokenizer()` in `diffgenslm/tokenizer/__init__.py` to load it.
3. Update `_tokenize_contig()` in `build_hdf5.py` to route the appropriate genomic region type to the new tokenizer.
4. Update `vocab_size` in your YAML config to reflect the new total.

---

## References

- **LLaDA**: *Large Language Diffusion with Masking* (2024) — absorbing-state diffusion for language
- **D3PM**: *Structured Denoising Diffusion Models in Discrete State-Spaces* (Austin et al., 2021)
- **GenSLM-2**: composite genome tokenizer and training framework — [StarNetLaboratory/GenSLM-2](https://github.com/StarNetLaboratory/GenSLM-2)
- **HiSAN**: DDP training infrastructure, RoPE implementation — [ramanathanlab/HiSAN](https://github.com/ramanathanlab/HiSAN)
