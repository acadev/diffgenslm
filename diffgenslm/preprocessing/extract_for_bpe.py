"""
Extract functional-noncoding and intergenic sequences from GenomeRecords
for training the two SentencePiece BPE sub-tokenizers.

The composite tokenizer has three sub-tokenizers:
  1. CodonTokenizer  – fixed vocabulary, no training needed
  2. Functional BPE  – trained on tRNA/rRNA/misc_feature sequences
  3. Noncoding BPE   – trained on unannotated intergenic sequence

This script writes two plain-text files (one sequence per line, uppercase)
that SentencePiece will train on.

Usage:
    python -m diffgenslm.preprocessing.extract_for_bpe \
        --gto_dir /path/to/gto_files \
        --fasta_dir /path/to/Streptomyces_genomes \
        --gff_dir /path/to/gff_files \
        --out_functional functional_seqs.txt \
        --out_noncoding  noncoding_seqs.txt \
        --max_seq_len 50000
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Union

from tqdm import tqdm

from .genome_records import ContigRecord, GenomeRecord
from .parse_gto import build_id_mapping_from_gtos, parse_gto
from .parse_gff_fasta import parse_gff_fasta


def _reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    return seq.translate(comp)[::-1]


def _get_annotated_intervals(contig: ContigRecord) -> List[Tuple[int, int, str]]:
    """Return (start, end, region_type) for all annotated features, merged by type."""
    intervals = []
    for f in contig.features:
        rtype = "functional" if f.feature_type == "misc_feature" else "cds"
        intervals.append((f.start, f.end, rtype))
    return sorted(intervals, key=lambda x: x[0])


def _extract_intergenic(contig: ContigRecord, max_seq_len: int) -> List[str]:
    """Return unannotated (intergenic) sequences from a contig."""
    intervals = _get_annotated_intervals(contig)
    seqs = []
    pos = 0
    for start, end, _ in intervals:
        if start > pos:
            chunk = contig.sequence[pos:start].upper()
            if len(chunk) >= 20:  # skip tiny gaps
                for i in range(0, len(chunk), max_seq_len):
                    seqs.append(chunk[i : i + max_seq_len])
        pos = max(pos, end)
    # Tail
    if pos < len(contig.sequence):
        chunk = contig.sequence[pos:].upper()
        if len(chunk) >= 20:
            for i in range(0, len(chunk), max_seq_len):
                seqs.append(chunk[i : i + max_seq_len])
    return seqs


def _extract_functional(contig: ContigRecord) -> List[str]:
    """Return sequences for misc_feature / tRNA / rRNA regions."""
    seqs = []
    for f in contig.features:
        if f.feature_type != "misc_feature":
            continue
        seq = contig.sequence[f.start : f.end].upper()
        if f.strand == -1:
            seq = _reverse_complement(seq)
        if len(seq) >= 10:
            seqs.append(seq)
    return seqs


def extract_from_genome(
    genome: GenomeRecord, max_seq_len: int = 50_000
) -> Tuple[List[str], List[str]]:
    """
    Returns (functional_seqs, noncoding_seqs) for one GenomeRecord.
    """
    functional, noncoding = [], []
    for contig in genome.contigs:
        functional.extend(_extract_functional(contig))
        noncoding.extend(_extract_intergenic(contig, max_seq_len))
    return functional, noncoding


def _iter_genomes(
    gto_dir: Path,
    fasta_dir: Path,
    gff_dir: Path,
    all_maps: dict,
) -> GenomeRecord:
    """Yield GenomeRecords from GTO files first, then GFF+FASTA fallback."""
    processed_ids = set()

    # Primary: GTO files
    if gto_dir and gto_dir.exists():
        for gto_path in sorted(gto_dir.glob("*.gto")):
            genome_id = gto_path.stem
            try:
                yield parse_gto(gto_path)
                processed_ids.add(genome_id)
            except Exception as e:
                print(f"[WARN] GTO parse failed {gto_path}: {e}")

    # Fallback: GFF + FASTA
    if gff_dir and gff_dir.exists() and fasta_dir and fasta_dir.exists():
        for gff_path in sorted(gff_dir.glob("*.gff")):
            genome_id = gff_path.stem
            if genome_id in processed_ids:
                continue
            fasta_path = fasta_dir / f"Streptomyces_{genome_id}.fasta"
            if not fasta_path.exists():
                # Try without prefix
                fasta_path = fasta_dir / f"{genome_id}.fasta"
            if not fasta_path.exists():
                continue
            try:
                yield parse_gff_fasta(
                    fasta_path=fasta_path,
                    gff_path=gff_path,
                    genome_id=genome_id,
                    contig_id_map=all_maps.get(genome_id, {}),
                )
            except Exception as e:
                print(f"[WARN] GFF+FASTA parse failed {gff_path}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Extract sequences for BPE tokenizer training")
    ap.add_argument("--gto_dir", type=Path, default=None)
    ap.add_argument("--fasta_dir", type=Path, default=None)
    ap.add_argument("--gff_dir", type=Path, default=None)
    ap.add_argument("--out_functional", type=Path, required=True)
    ap.add_argument("--out_noncoding", type=Path, required=True)
    ap.add_argument("--max_seq_len", type=int, default=50_000,
                    help="Max chunk length for noncoding sequences (BPE perf)")
    args = ap.parse_args()

    # Build ID mapping from GTOs
    all_maps = {}
    if args.gto_dir and args.gto_dir.exists():
        print(f"Building contig ID mapping from {args.gto_dir} ...")
        all_maps = build_id_mapping_from_gtos(args.gto_dir)
        print(f"  Mapped {len(all_maps)} genomes")

    functional_out = open(args.out_functional, "w")
    noncoding_out = open(args.out_noncoding, "w")
    n_genomes = n_func = n_nc = 0

    genome_iter = _iter_genomes(args.gto_dir, args.fasta_dir, args.gff_dir, all_maps)

    for genome in tqdm(genome_iter, desc="Extracting sequences"):
        func, nc = extract_from_genome(genome, args.max_seq_len)
        for s in func:
            functional_out.write(s + "\n")
        for s in nc:
            noncoding_out.write(s + "\n")
        n_genomes += 1
        n_func += len(func)
        n_nc += len(nc)

    functional_out.close()
    noncoding_out.close()

    print(f"Done. {n_genomes} genomes | {n_func} functional seqs | {n_nc} noncoding seqs")
    print(f"  → {args.out_functional}")
    print(f"  → {args.out_noncoding}")


if __name__ == "__main__":
    main()
