"""
Parse PATRIC/BV-BRC GTO (Genome-Typed Object) JSON files.

GTO is self-contained: it carries both the DNA sequences (contigs[].dna)
and all feature annotations (features[].location) using consistent PATRIC
internal contig IDs, so no ID-mapping gymnastics are required.

Usage:
    from diffgenslm.preprocessing.parse_gto import parse_gto
    genome = parse_gto("/path/to/371608.5.gto")
"""

import json
from pathlib import Path
from typing import Union

from .genome_records import ContigRecord, FeatureRecord, GenomeRecord

# GTO feature types we want to keep and how to classify them for tokenization.
# CDS  → codon tokenizer
# rRNA / tRNA / ncRNA / misc_RNA → functional BPE tokenizer
# Everything else is dropped (repeat_region, mobile element, etc.)
_FEATURE_TYPE_MAP = {
    "CDS":          "CDS",
    "rRNA":         "misc_feature",
    "tRNA":         "misc_feature",
    "ncRNA":        "misc_feature",
    "misc_RNA":     "misc_feature",
    "regulatory":   "misc_feature",
}


def _parse_strand(strand_val) -> int:
    """GTO location strand: '+' or 1 → 1, '-' or -1 → -1."""
    if strand_val in ("+", 1):
        return 1
    if strand_val in ("-", -1):
        return -1
    return 1  # default forward


def parse_gto(path: Union[str, Path]) -> GenomeRecord:
    """
    Parse a single GTO file and return a GenomeRecord.

    GTO location format:  [[contig_id, start_str, strand, length], ...]
    start_str is 1-based (PATRIC convention).  We convert to 0-based half-open.
    """
    path = Path(path)
    with open(path) as fh:
        gto = json.load(fh)

    genome_id = gto.get("id", path.stem)
    sci_name = gto.get("scientific_name", "")

    # Build contig lookup: patric_id → ContigRecord (features filled in below)
    contig_map: dict[str, ContigRecord] = {}
    for c in gto.get("contigs", []):
        patric_id = c["id"]
        contig_map[patric_id] = ContigRecord(
            contig_id=patric_id,
            original_id=c.get("original_id", ""),
            sequence=c.get("dna", ""),
        )

    # Parse features
    for feat in gto.get("features", []):
        ftype_raw = feat.get("type", "")
        ftype = _FEATURE_TYPE_MAP.get(ftype_raw)
        if ftype is None:
            continue  # skip repeat regions, mobile elements, etc.

        feat_id = feat.get("id", "")
        function = feat.get("function", "")

        for loc in feat.get("location", []):
            # loc = [contig_id, start_1based_str, strand, length]
            if len(loc) < 4:
                continue
            contig_id, start_str, strand_raw, length = loc[0], loc[1], loc[2], loc[3]
            if contig_id not in contig_map:
                continue

            start_1based = int(start_str)
            strand = _parse_strand(strand_raw)
            length = int(length)

            # GTO uses 1-based start; convert to 0-based half-open [start, end)
            if strand == 1:
                start = start_1based - 1
                end = start + length
            else:
                # For reverse-strand features, GTO gives the leftmost coordinate
                # as start and the feature extends to the right.
                end = start_1based
                start = end - length

            start = max(0, start)
            end = min(len(contig_map[contig_id].sequence), end)
            if start >= end:
                continue

            contig_map[contig_id].features.append(
                FeatureRecord(
                    contig_id=contig_id,
                    start=start,
                    end=end,
                    strand=strand,
                    feature_type=ftype,
                    frame=0,  # prokaryotic CDS always start at codon boundary
                    feature_id=feat_id,
                    function=function,
                )
            )

    # Sort features by start within each contig
    for contig in contig_map.values():
        contig.features.sort(key=lambda f: f.start)

    return GenomeRecord(
        genome_id=genome_id,
        scientific_name=sci_name,
        contigs=list(contig_map.values()),
    )


def build_id_mapping_from_gtos(gto_dir: Union[str, Path]) -> dict[str, dict[str, str]]:
    """
    Scan all GTO files in gto_dir and return a mapping:
        { genome_id → { ncbi_accession → patric_contig_id } }

    Used by parse_gff_fasta.py when GTO files are not available per-genome
    but a bulk download has been done for a subset.
    """
    gto_dir = Path(gto_dir)
    mapping: dict[str, dict[str, str]] = {}
    for gto_path in sorted(gto_dir.glob("*.gto")):
        with open(gto_path) as fh:
            gto = json.load(fh)
        genome_id = gto.get("id", gto_path.stem)
        genome_map: dict[str, str] = {}
        for c in gto.get("contigs", []):
            orig = c.get("original_id", "")
            if orig:
                genome_map[orig] = c["id"]
        if genome_map:
            mapping[genome_id] = genome_map
    return mapping
