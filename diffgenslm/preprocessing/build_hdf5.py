"""
Main preprocessing pipeline: genome files → tokenized HDF5 datasets.

Tokenization strategy (GenSLM-2 CompositeGenomeTokenizerV2):
  CDS regions        → CodonTokenizer  (codon triplets, frame-aware)
  misc/tRNA/rRNA     → functional BPE  (SentencePiece)
  Intergenic gaps    → noncoding BPE   (SentencePiece)

Output HDF5 layout (compatible with GenSLM-2's GenomeDataset):
  input_ids      : 1-D int32 array, all sequences concatenated
  sequence_starts: 1-D int64 array, start offset of each sequence

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
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .genome_records import ContigRecord, GenomeRecord
from .parse_gto import build_id_mapping_from_gtos, parse_gto
from .parse_gff_fasta import parse_gff_fasta

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
# HDF5 writer (mirrors GenSLM-2's HDF5Writer for compatibility)
# ---------------------------------------------------------------------------

class _HDF5Writer:
    """Write tokenized sequences to HDF5 in a contiguous 1-D token stream."""

    def __init__(self, output_path: str, vocab_size: int, special_tokens: dict,
                 buffer_size: int = 100_000, compression: str = "gzip"):
        self.output_path = output_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.buffer_size = buffer_size
        self.compression = compression

        self._token_buf: List[int] = []
        self._start_buf: List[int] = []
        self._total_tokens = 0
        self._num_seqs = 0
        self._h5 = None
        self._ids_ds = None
        self._starts_ds = None

    def add(self, token_ids: List[int]):
        self._start_buf.append(self._total_tokens + len(self._token_buf))
        self._token_buf.extend(token_ids)
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

        ns = len(self._start_buf)
        curs = self._starts_ds.shape[0]
        self._starts_ds.resize((curs + ns,))
        self._starts_ds[curs : curs + ns] = np.array(self._start_buf, dtype=np.int64)

        self._total_tokens += n
        self._token_buf = []
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
        self._starts_ds = self._h5.create_dataset(
            "sequence_starts", shape=(0,), maxshape=(None,), dtype=np.int64,
            chunks=(1024,), compression=self.compression,
        )

    def finalize(self, metadata: dict = None):
        self._flush()
        if self._h5 is None:
            self._open()
        self._h5.attrs["bos_token_id"] = self.special_tokens.get("<bos>", 1)
        self._h5.attrs["eos_token_id"] = self.special_tokens.get("<eos>", 2)
        self._h5.attrs["pad_token_id"] = self.special_tokens.get("<pad>", 0)
        self._h5.attrs["mask_token_id"] = self.special_tokens.get("<mask>", 4)
        self._h5.attrs["num_sequences"] = self._num_seqs
        self._h5.attrs["total_tokens"] = self._total_tokens
        self._h5.attrs["vocab_size"] = self.vocab_size
        if metadata:
            self._h5.attrs["metadata"] = json.dumps(metadata)
        self._h5.close()
        print(f"[HDF5] {self._num_seqs} sequences, {self._total_tokens:,} tokens → {self.output_path}")


# ---------------------------------------------------------------------------
# Region extraction (mirrors GenSLM-2 composite tokenizer region logic)
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
# Tokenize one contig using composite tokenizer
# ---------------------------------------------------------------------------

def _tokenize_contig(
    contig: ContigRecord,
    tokenizer,           # CompositeGenomeTokenizerV2 instance
    chunk_size: int = 10_000_000,
) -> List[int]:
    """
    Tokenize a full contig using the composite tokenizer.

    Rather than passing a BioPython SeqRecord (not always available), we
    reconstruct a GenomicRegionV2 list from our FeatureRecords and call
    the tokenizer's internal _tokenize_sequence directly.

    Falls back to tokenizer.encode(sequence) when annotations are absent.
    """
    try:
        from genslm2.composite_tokenizer_v2 import (
            CompositeGenomeTokenizerV2,
            GenomicRegionV2,
        )
    except ImportError:
        raise ImportError(
            "GenSLM-2 is required for tokenization.\n"
            "Install: pip install git+https://github.com/StarNetLaboratory/GenSLM-2.git"
        )

    if not contig.features:
        # No annotations: treat whole contig as non-coding
        result = tokenizer.encode(contig.sequence, add_special_tokens=True)
        return result["input_ids"]

    # Build GenomicRegionV2 objects from our FeatureRecords
    regions = []
    for r in _build_region_list(contig):
        regions.append(
            GenomicRegionV2(
                start=r["start"],
                end=r["end"],
                region_type=r["region_type"],
                strand=r["strand"],
                frame=r["frame"],
            )
        )

    return tokenizer._tokenize_sequence(
        contig.sequence,
        regions,
        add_bos=True,
        add_eos=True,
    )


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
    # Collect all genome source files
    sources: List[Tuple[str, Path, Optional[Path]]] = []  # (mode, primary, secondary)

    if file_list:
        # Explicit file list (train/val/test split already done)
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

    # Rank sharding
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
    n_val = int(n * val_frac)

    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def run(args):
    rank, world_size = _init_mpi()
    is_main = rank == 0

    if is_main:
        print(f"[build_hdf5] world_size={world_size}")
        print(f"  tokenizer_dir  = {args.tokenizer_dir}")
        print(f"  output_dir     = {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load composite tokenizer
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
    vocab_size = tokenizer.vocab_size
    special_tokens = tokenizer.special_token_ids

    if is_main:
        print(f"  vocab_size = {vocab_size}")

    # Build contig ID mapping from available GTOs
    all_maps: dict = {}
    if args.gto_dir and Path(args.gto_dir).exists():
        all_maps = build_id_mapping_from_gtos(Path(args.gto_dir))

    # Split files across train / val / test
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
                    token_ids = _tokenize_contig(contig, tokenizer, args.chunk_size)
                except Exception as e:
                    print(f"[WARN] tokenize failed {genome.genome_id}/{contig.contig_id}: {e}")
                    continue
                if token_ids:
                    writer.add(token_ids)
                    n_contigs += 1
                    n_tokens += len(token_ids)

        writer.finalize({"split": split_name, "rank": rank, "n_contigs": n_contigs})
        print(f"[rank {rank}] {split_name}: {n_contigs} contigs, {n_tokens:,} tokens")

    _mpi_barrier()
    if is_main:
        print("[build_hdf5] All ranks done.")


def main():
    ap = argparse.ArgumentParser(description="Tokenize genomes to HDF5")
    ap.add_argument("--gto_dir", type=Path, default=None, help="Directory of *.gto files")
    ap.add_argument("--fasta_dir", type=Path, default=None, help="Directory of *.fasta files")
    ap.add_argument("--gff_dir", type=Path, default=None, help="Directory of *.gff files")
    ap.add_argument("--tokenizer_dir", type=Path, required=True,
                    help="Directory with codon/, functional_bpe.model, noncoding_bpe.model")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_contig_len", type=int, default=500,
                    help="Skip contigs shorter than this (bp)")
    ap.add_argument("--chunk_size", type=int, default=10_000_000,
                    help="Max sequence length per BPE call (BPE performance)")
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    args = ap.parse_args()

    if args.compression == "none":
        args.compression = None

    run(args)


if __name__ == "__main__":
    main()
