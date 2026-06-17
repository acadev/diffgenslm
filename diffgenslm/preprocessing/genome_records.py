"""Shared data structures for genome parsing."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureRecord:
    """A single annotated genomic feature (CDS, misc_feature, etc.)."""
    contig_id: str       # PATRIC internal ID (e.g. "371608.5.con.0067")
    start: int           # 0-based, inclusive
    end: int             # 0-based, exclusive
    strand: int          # 1 = forward (+), -1 = reverse (-)
    feature_type: str    # "CDS", "misc_feature", "tRNA", "rRNA", ...
    frame: int           # reading-frame offset (0, 1, or 2); always 0 for prokaryotic CDS
    feature_id: str      # e.g. "fig|103232.14.peg.9506"
    function: str        # human-readable product name


@dataclass
class ContigRecord:
    """One contig/chromosome within a genome."""
    contig_id: str        # PATRIC internal ID  (e.g. "103232.14.con.0086")
    original_id: str      # NCBI accession      (e.g. "SPBA01000178") — empty if unknown
    sequence: str         # full DNA sequence
    features: List[FeatureRecord] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class GenomeRecord:
    """All contigs and their annotations for one genome."""
    genome_id: str           # PATRIC genome ID  (e.g. "103232.14")
    scientific_name: str     # e.g. "Streptomyces coelicolor"
    contigs: List[ContigRecord] = field(default_factory=list)

    @property
    def total_length(self) -> int:
        return sum(c.length for c in self.contigs)

    @property
    def total_cds(self) -> int:
        return sum(
            1 for c in self.contigs for f in c.features if f.feature_type == "CDS"
        )
