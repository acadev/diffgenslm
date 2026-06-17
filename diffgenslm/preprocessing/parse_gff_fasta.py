"""
Parse GFF3 + FASTA pairs where contig IDs do not match.

Context for this dataset:
  - FASTA headers use PATRIC internal IDs:  >103232.14.con.0086 contig
  - GFF seqnames use NCBI accessions:        SPBA01000178

The two are reconciled via a contig_id_map built from GTO files:
  { ncbi_accession → patric_contig_id }

If no map is available for a genome, we fall back to matching contigs by
length (sorted-order matching), which works when the FASTA and GFF contigs
are in the same order and have no duplicates.

Usage:
    from diffgenslm.preprocessing.parse_gff_fasta import parse_gff_fasta
    from diffgenslm.preprocessing.parse_gto import build_id_mapping_from_gtos

    # Build mapping once from available GTO files
    all_maps = build_id_mapping_from_gtos("/path/to/gto_files")

    genome = parse_gff_fasta(
        fasta_path="/path/to/Streptomyces_103232.14.fasta",
        gff_path="/path/to/103232.14.gff",
        genome_id="103232.14",
        contig_id_map=all_maps.get("103232.14", {}),
    )
"""

import re
from pathlib import Path
from typing import Dict, Optional, Union

from .genome_records import ContigRecord, FeatureRecord, GenomeRecord

# GFF3 col 3 feature types we retain
_GFF_FEATURE_KEEP = {"CDS", "misc_feature", "tRNA", "rRNA", "ncRNA", "misc_RNA"}

# How GFF feature types map to composite-tokenizer region types
_GFF_TO_REGION = {
    "CDS":       "CDS",
    "misc_feature": "misc_feature",
    "tRNA":      "misc_feature",
    "rRNA":      "misc_feature",
    "ncRNA":     "misc_feature",
    "misc_RNA":  "misc_feature",
}


def _parse_fasta(path: Path) -> Dict[str, str]:
    """Return {header_id: sequence} dict from a FASTA file."""
    sequences: Dict[str, str] = {}
    current_id: Optional[str] = None
    chunks: list[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(chunks)

    return sequences


def _parse_gff_features(path: Path) -> Dict[str, list]:
    """
    Parse a GFF3 file and return { seqname: [raw_feature_dict, ...] }.

    raw_feature_dict keys: start, end, strand, ftype, frame, feature_id, function
    start/end are 0-based half-open (converted from GFF3 1-based closed).
    """
    features_by_seq: Dict[str, list] = {}
    attr_re = re.compile(r'(?:ID|Name)=([^;]+)')

    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue

            seqname, _, ftype, start_s, end_s, _, strand_s, phase_s, attrs = cols
            if ftype not in _GFF_FEATURE_KEEP:
                continue

            # GFF3: start is 1-based inclusive, end is 1-based inclusive
            start = int(start_s) - 1   # → 0-based
            end = int(end_s)            # → 0-based exclusive

            strand = 1 if strand_s == "+" else -1
            frame = int(phase_s) if phase_s in ("0", "1", "2") else 0

            # Extract feature_id and product name from attributes
            matches = attr_re.findall(attrs)
            feature_id = matches[0] if matches else ""
            function = matches[1] if len(matches) > 1 else ""

            feat = {
                "start": start,
                "end": end,
                "strand": strand,
                "ftype": _GFF_TO_REGION.get(ftype, ftype),
                "frame": frame,
                "feature_id": feature_id,
                "function": function,
            }
            features_by_seq.setdefault(seqname, []).append(feat)

    return features_by_seq


def _match_by_length(
    fasta_ids: list[str],
    fasta_seqs: Dict[str, str],
    gff_seqnames: list[str],
    gff_lengths_approx: Dict[str, int],
) -> Dict[str, str]:
    """
    Last-resort: match GFF seqnames to FASTA IDs by sorting both by sequence length.

    Returns { ncbi_seqname → patric_fasta_id } mapping.
    Only reliable when the assembly has no duplicate-length contigs.
    """
    fasta_by_len = sorted(fasta_ids, key=lambda x: len(fasta_seqs[x]), reverse=True)
    gff_by_len = sorted(gff_seqnames, key=lambda x: gff_lengths_approx.get(x, 0), reverse=True)

    if len(fasta_by_len) != len(gff_by_len):
        return {}

    return {ncbi: patric for ncbi, patric in zip(gff_by_len, fasta_by_len)}


def parse_gff_fasta(
    fasta_path: Union[str, Path],
    gff_path: Union[str, Path],
    genome_id: str,
    scientific_name: str = "",
    contig_id_map: Optional[Dict[str, str]] = None,
) -> GenomeRecord:
    """
    Build a GenomeRecord from a FASTA + GFF3 pair.

    Args:
        fasta_path:     Path to FASTA file (headers use PATRIC IDs).
        gff_path:       Path to GFF3 file (seqnames use NCBI accessions).
        genome_id:      PATRIC genome ID (e.g. "103232.14").
        scientific_name: Species name.
        contig_id_map:  { ncbi_accession → patric_contig_id } from GTO.
                        If None or empty, length-based matching is attempted.

    Returns:
        GenomeRecord with fully populated ContigRecords and FeatureRecords.
    """
    fasta_path = Path(fasta_path)
    gff_path = Path(gff_path)

    fasta_seqs = _parse_fasta(fasta_path)
    gff_features = _parse_gff_features(gff_path)

    # Determine NCBI→PATRIC mapping
    if not contig_id_map:
        # Estimate lengths from max coordinate in GFF
        gff_max_coord: Dict[str, int] = {}
        for seqname, feats in gff_features.items():
            gff_max_coord[seqname] = max(f["end"] for f in feats)

        contig_id_map = _match_by_length(
            list(fasta_seqs.keys()), fasta_seqs,
            list(gff_features.keys()), gff_max_coord,
        )

    # Invert map for lookup: patric_id → ncbi_seqname
    patric_to_ncbi = {v: k for k, v in contig_id_map.items()}

    contigs: list[ContigRecord] = []
    for patric_id, seq in fasta_seqs.items():
        ncbi_id = patric_to_ncbi.get(patric_id, "")
        raw_feats = gff_features.get(ncbi_id, [])

        feature_records = []
        for rf in raw_feats:
            start = max(0, rf["start"])
            end = min(len(seq), rf["end"])
            if start >= end:
                continue
            feature_records.append(
                FeatureRecord(
                    contig_id=patric_id,
                    start=start,
                    end=end,
                    strand=rf["strand"],
                    feature_type=rf["ftype"],
                    frame=rf["frame"],
                    feature_id=rf["feature_id"],
                    function=rf["function"],
                )
            )

        feature_records.sort(key=lambda f: f.start)
        contigs.append(
            ContigRecord(
                contig_id=patric_id,
                original_id=ncbi_id,
                sequence=seq,
                features=feature_records,
            )
        )

    return GenomeRecord(
        genome_id=genome_id,
        scientific_name=scientific_name,
        contigs=contigs,
    )
