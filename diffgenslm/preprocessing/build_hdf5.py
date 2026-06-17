"""
Main preprocessing pipeline: genome files → tokenized HDF5 datasets.

Tokenization strategy (GenSLM-2 CompositeGenomeTokenizerV2):
  CDS regions        → CodonTokenizer  (codon triplets, frame-aware)
  misc/tRNA/rRNA     → functional BPE  (SentencePiece)
  Intergenic gaps    → noncoding BPE   (SentencePiece)

Output HDF5 layout:
  input_ids      : 1-D int32, all sequences concatenated
  strand_ids     : 1-D int8,  strand label per token (0=pad, 1=fwd, 2=rev, 3=intergenic)
  sequence_starts: 1-D int64, start offset of each sequence

MPI parallelism: each rank processes rank::world_size files.

Usage (single node):
    python -m diffgenslm.preprocessing.build_hdf5 \\
        --gto_dir /path/to/gto_files \\
        --fasta_dir /path/to/Streptomyces_genomes \\
        --gff_dir /path/to/gff_files \\
        --tokenizer_dir /path/to/tokenizer_models \\
        --output_dir /path/to/hdf5_output \\
        --train_frac 0.8 --val_frac 0.1

Usage (Polaris / mpiexec):
    mpiexec -n 32 python -m diffgenslm.preprocessing.build_hdf5 [same args]
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .genome_records import ContigRecord
from .parse_gto import build_id_mapping_from_gtos, parse_gto
from .parse_gff_fasta import parse_gff_fasta

# ---------------------------------------------------------------------------
# Strand ID constants (shared with genome_dataset.py and diffgenome.py)
# ---------------------------------------------------------------------------

STRAND_PAD        = 0   # BOS, EOS, pad tokens — no strand
STRAND_FWD        = 1   # forward / + strand
STRAND_REV        = 2   # reverse / − strand
STRAND_INTERGENIC = 3   # intergenic / non-coding (strand-ambiguous)


# ---------------------------------------------------------------------------
# MPI helpers
# ---------------------------------------------------------------------------

def _init_mpi() -> Tuple[int, int]:
    """Return (rank, world_size). Falls back to (0, 1) when MPI is absent."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except ImportError:
        return 0, 1


def _mpi_barrier():
    try:
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

class _HDF5Writer:
    """
    Write tokenized sequences to HDF5 in a contiguous 1-D token stream.

    Stores three datasets:
      input_ids       — int32, token IDs
      strand_ids      — int8,  per-token strand labels (only when provided)
      sequence_starts — int64, start index of each sequence in the flat arrays
    """

    def __init__(self, output_path: str, vocab_size: int, special_tokens: dict,
                 buffer_size: int = 100_000, compression: str = "gzip"):
        self.output_path = output_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.buffer_size = buffer_size
        self.compression = compression

        self._token_buf:  List[int] = []
        self._strand_buf: List[int] = []
        self._start_buf:  List[int] = []
        self._total_tokens = 0
        self._num_seqs = 0
        self._has_strands = False

        self._h5 = None
        self._ids_ds = None
        self._strands_ds = None
        self._starts_ds = None

    def add(self, token_ids: List[int], strand_ids: Optional[List[int]] = None):
        self._start_buf.append(self._total_tokens + len(self._token_buf))
        self._token_buf.extend(token_ids)
        if strand_ids is not None:
            self._strand_buf.extend(strand_ids)
            self._has_strands = True
        self._num_seqs += 1
        if len(self._token_buf) >= self.buffer_size:
            self._flush()

    def _flush(self):
        if not self._token_buf:
            return
        if self._h5 is None:
            self._open()
        n = len(self._token_buf)
        cur = self._ids_ds.shape[0]
        self._ids_ds.resize((cur + n,))
        self._ids_ds[cur : cur + n] = np.array(self._token_buf, dtype=np.int32)

        if self._has_strands and self._strands_ds is not None:
            self._strands_ds.resize((cur + n,))
            self._strands_ds[cur : cur + n] = np.array(self._strand_buf, dtype=np.int8)

        ns = len(self._start_buf)
        curs = self._starts_ds.shape[0]
        self._starts_ds.resize((curs + ns,))
        self._starts_ds[curs : curs + ns] = np.array(self._start_buf, dtype=np.int64)

        self._total_tokens += n
        self._token_buf = []
        self._strand_buf = []
        self._start_buf = []
        self._h5.flush()

    def _open(self):
        import h5py
        self._h5 = h5py.File(self.output_path, "w")
        chunk = 4096
        self._ids_ds = self._h5.create_dataset(
            "input_ids", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(chunk,), compression=self.compression,
        )
        if self._has_strands:
            self._strands_ds = self._h5.create_dataset(
                "strand_ids", shape=(0,), maxshape=(None,), dtype=np.int8,
                chunks=(chunk,), compression=self.compression,
            )
        self._starts_ds = self._h5.create_dataset(
            "sequence_starts", shape=(0,), maxshape=(None,), dtype=np.int64,
            chunks=(1024,), compression=self.compression,
        )

    def finalize(self, metadata: dict = None):
        self._flush()
        if self._h5 is None:
            self._open()
        self._h5.attrs["bos_token_id"]  = self.special_tokens.get("<bos>", 1)
        self._h5.attrs["eos_token_id"]  = self.special_tokens.get("<eos>", 2)
        self._h5.attrs["pad_token_id"]  = self.special_tokens.get("<pad>", 0)
        self._h5.attrs["mask_token_id"] = self.special_tokens.get("<mask>", 4)
        self._h5.attrs["num_sequences"] = self._num_seqs
        self._h5.attrs["total_tokens"]  = self._total_tokens
        self._h5.attrs["vocab_size"]    = self.vocab_size
        self._h5.attrs["has_strand_ids"] = self._has_strands
        if metadata:
            self._h5.attrs["metadata"] = json.dumps(metadata)
        self._h5.close()
        strand_note = " + strand_ids" if self._has_strands else ""
        print(f"[HDF5] {self._num_seqs} sequences, {self._total_tokens:,} tokens{strand_note} → {self.output_path}")


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _find_intergenic_regions(contig: ContigRecord) -> List[Tuple[int, int]]:
    """Return (start, end) pairs for unannotated gaps in the contig."""
    annotated = sorted((f.start, f.end) for f in contig.features)
    gaps = []
    pos = 0
    for start, end in annotated:
        if start > pos:
            gaps.append((pos, start))
        pos = max(pos, end)
    if pos < len(contig.sequence):
        gaps.append((pos, len(contig.sequence)))
    return gaps


def _build_region_list(contig: ContigRecord) -> List[dict]:
    """
    Build an ordered list of regions across the contig with their types.
    Each dict: {start, end, strand, region_type, frame}
    """
    regions = []
    for f in contig.features:
        regions.append({
            "start": f.start, "end": f.end,
            "strand": f.strand, "frame": f.frame,
            "region_type": f.feature_type,   # "CDS" or "misc_feature"
        })
    for start, end in _find_intergenic_regions(contig):
        regions.append({
            "start": start, "end": end,
            "strand": 1, "frame": 0,
            "region_type": "non_coding",
        })
    regions.sort(key=lambda r: r["start"])
    return regions


# ---------------------------------------------------------------------------
# Tokenize one contig — two variants
# ---------------------------------------------------------------------------

# Maps raw feature_type from GTO/GFF → CompositeGenomeTokenizerV2 region types.
# "misc_feature" covers tRNA, rRNA, and other functional non-coding elements;
# they use the functional BPE tokenizer, not the noncoding BPE.
_REGION_TYPE_MAP = {
    "CDS":          "CDS",
    "misc_feature": "functional_non_coding",
    "non_coding":   "non_coding",
}


def _tokenize_contig_with_strand(
    contig: ContigRecord,
    tokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a full contig and return aligned (token_ids, strand_ids).

    Strand encoding:
      STRAND_PAD        (0) — BOS / EOS / pad
      STRAND_FWD        (1) — forward (+) CDS or functional non-coding
      STRAND_REV        (2) — reverse (−) CDS or functional non-coding
      STRAND_INTERGENIC (3) — intergenic / non-coding

    Also fixes the misc_feature → functional_non_coding mapping (tRNA/rRNA
    were previously sent to the noncoding BPE instead of the functional BPE).
    """
    if not contig.features:
        result = tokenizer.encode(contig.sequence, add_special_tokens=True)
        ids = result["input_ids"]
        n = len(ids)
        strands = (
            [STRAND_PAD]
            + [STRAND_INTERGENIC] * max(0, n - 2)
            + ([STRAND_PAD] if n >= 2 else [])
        )
        return ids, strands[:n]

    token_ids  = [tokenizer.bos_token_id]
    strand_ids = [STRAND_PAD]

    for r in _build_region_list(contig):
        rtype = _REGION_TYPE_MAP.get(r["region_type"], "non_coding")
        region_seq = contig.sequence[r["start"]:r["end"]]
        if not region_seq:
            continue

        rev = r["strand"] == -1
        if rev:
            region_seq = tokenizer._reverse_complement(region_seq)

        if rtype == "CDS":
            frame = r["frame"] if r["frame"] > 0 else 0
            coding_seq = region_seq[frame:]
            tokens    = tokenizer.codon_tokenizer.tokenize(
                coding_seq, prefix_singles=r.get("prefix_singles", 0)
            )
            local_ids = tokenizer.codon_tokenizer.convert_tokens_to_ids(tokens)
            global_ids = [tokenizer._map_to_global_id("codon", lid) for lid in local_ids]
            sid = STRAND_REV if rev else STRAND_FWD

        elif rtype == "functional_non_coding":
            local_ids = tokenizer.tokenization_chunker.chunk_and_tokenize(
                region_seq, tokenizer.functional_tokenizer, rtype
            )
            global_ids = [tokenizer._map_to_global_id("functional", lid) for lid in local_ids]
            sid = STRAND_REV if rev else STRAND_FWD

        else:  # non_coding / intergenic
            local_ids = tokenizer.tokenization_chunker.chunk_and_tokenize(
                region_seq, tokenizer.noncoding_tokenizer, rtype
            )
            global_ids = [tokenizer._map_to_global_id("noncoding", lid) for lid in local_ids]
            sid = STRAND_INTERGENIC

        token_ids.extend(global_ids)
        strand_ids.extend([sid] * len(global_ids))

    token_ids.append(tokenizer.eos_token_id)
    strand_ids.append(STRAND_PAD)

    return token_ids, strand_ids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _genome_iterator(
    rank: int,
    world_size: int,
    gto_dir: Optional[Path],
    fasta_dir: Optional[Path],
    gff_dir: Optional[Path],
    file_list: Optional[List[Path]],
    all_maps: dict,
):
    """Yield GenomeRecords for files assigned to this rank."""
    sources: List[Tuple[str, Path, Optional[Path]]] = []

    if file_list:
        for p in file_list:
            if p.suffix == ".gto":
                sources.append(("gto", p, None))
            else:
                sources.append(("gff", p, None))
    else:
        if gto_dir and gto_dir.exists():
            for p in sorted(gto_dir.glob("*.gto")):
                sources.append(("gto", p, None))
        elif gff_dir and gff_dir.exists():
            for p in sorted(gff_dir.glob("*.gff")):
                sources.append(("gff", p, None))

    my_sources = sources[rank::world_size]

    for mode, primary, _ in my_sources:
        genome_id = primary.stem
        try:
            if mode == "gto":
                yield parse_gto(primary)
            else:
                fasta_path = None
                if fasta_dir:
                    for candidate in [
                        fasta_dir / f"Streptomyces_{genome_id}.fasta",
                        fasta_dir / f"{genome_id}.fasta",
                    ]:
                        if candidate.exists():
                            fasta_path = candidate
                            break
                if fasta_path is None:
                    print(f"[WARN] No FASTA for {genome_id}, skipping")
                    continue
                yield parse_gff_fasta(
                    fasta_path=fasta_path,
                    gff_path=primary,
                    genome_id=genome_id,
                    contig_id_map=all_maps.get(genome_id, {}),
                )
        except Exception as e:
            print(f"[WARN] Failed to parse {primary}: {e}")


def _split_files(
    gto_dir: Optional[Path],
    gff_dir: Optional[Path],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Return (train, val, test) lists of source file paths."""
    files: List[Path] = []
    if gto_dir and gto_dir.exists():
        files = sorted(gto_dir.glob("*.gto"))
    elif gff_dir and gff_dir.exists():
        files = sorted(gff_dir.glob("*.gff"))

    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def run(args):
    rank, world_size = _init_mpi()
    is_main = rank == 0

    if is_main:
        print(f"[build_hdf5] world_size={world_size}")
        print(f"  tokenizer_dir  = {args.tokenizer_dir}")
        print(f"  output_dir     = {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from genslm2.composite_tokenizer_v2 import CompositeGenomeTokenizerV2
    except ImportError:
        raise ImportError(
            "Install GenSLM-2: pip install git+https://github.com/StarNetLaboratory/GenSLM-2.git"
        )

    tdir = Path(args.tokenizer_dir)
    tokenizer = CompositeGenomeTokenizerV2(
        codon_tokenizer_path=str(tdir / "codon"),
        functional_tokenizer_path=str(tdir / "functional_bpe.model"),
        noncoding_tokenizer_path=str(tdir / "noncoding_bpe.model"),
    )
    vocab_size    = tokenizer.vocab_size
    special_tokens = tokenizer.special_token_ids

    if is_main:
        print(f"  vocab_size = {vocab_size}")

    all_maps: dict = {}
    if args.gto_dir and Path(args.gto_dir).exists():
        all_maps = build_id_mapping_from_gtos(Path(args.gto_dir))

    train_files, val_files, test_files = _split_files(
        Path(args.gto_dir) if args.gto_dir else None,
        Path(args.gff_dir) if args.gff_dir else None,
        args.train_frac, args.val_frac, args.seed,
    )
    if is_main:
        print(f"  splits: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if not split_files:
            continue

        out_path = str(
            args.output_dir / f"{split_name}_rank{rank:04d}.h5"
            if world_size > 1
            else args.output_dir / f"{split_name}.h5"
        )
        writer = _HDF5Writer(out_path, vocab_size, special_tokens, compression=args.compression)

        n_contigs = n_tokens = 0
        for genome in tqdm(
            _genome_iterator(rank, world_size, None, None, None, split_files, all_maps),
            desc=f"[rank {rank}] {split_name}",
            disable=not is_main,
        ):
            for contig in genome.contigs:
                if len(contig.sequence) < args.min_contig_len:
                    continue
                try:
                    token_ids, strand_ids = _tokenize_contig_with_strand(
                        contig, tokenizer
                    )
                except Exception as e:
                    print(f"[WARN] tokenize failed {genome.genome_id}/{contig.contig_id}: {e}")
                    continue
                if token_ids:
                    writer.add(token_ids, strand_ids)
                    n_contigs += 1
                    n_tokens  += len(token_ids)

        writer.finalize({"split": split_name, "rank": rank, "n_contigs": n_contigs})
        print(f"[rank {rank}] {split_name}: {n_contigs} contigs, {n_tokens:,} tokens")

    _mpi_barrier()
    if is_main:
        print("[build_hdf5] All ranks done.")


def main():
    ap = argparse.ArgumentParser(description="Tokenize genomes to HDF5")
    ap.add_argument("--gto_dir",       type=Path, default=None)
    ap.add_argument("--fasta_dir",     type=Path, default=None)
    ap.add_argument("--gff_dir",       type=Path, default=None)
    ap.add_argument("--tokenizer_dir", type=Path, required=True,
                    help="Directory with codon/, functional_bpe.model, noncoding_bpe.model")
    ap.add_argument("--output_dir",    type=Path, required=True)
    ap.add_argument("--train_frac",    type=float, default=0.8)
    ap.add_argument("--val_frac",      type=float, default=0.1)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--min_contig_len", type=int,  default=500)
    ap.add_argument("--compression",   default="gzip", choices=["gzip", "lzf", "none"])
    args = ap.parse_args()

    if args.compression == "none":
        args.compression = None

    run(args)


if __name__ == "__main__":
    main()
