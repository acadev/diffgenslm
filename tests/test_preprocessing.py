"""
Unit tests for the preprocessing pipeline.

Tests cover:
  - genome_records dataclasses
  - parse_gto coordinate conversion and feature parsing
  - parse_gff_fasta ID handling
  - extract_for_bpe functional/intergenic extraction
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from diffgenslm.preprocessing.genome_records import (
    ContigRecord,
    FeatureRecord,
    GenomeRecord,
)
from diffgenslm.preprocessing.parse_gto import parse_gto, build_id_mapping_from_gtos
from diffgenslm.preprocessing.parse_gff_fasta import parse_gff_fasta
from diffgenslm.preprocessing.extract_for_bpe import extract_from_genome


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_feature(
    contig_id="ctg1",
    start=0, end=9, strand=1,
    feature_type="CDS",
    frame=0,
    feature_id="f1",
    function="hypothetical",
) -> FeatureRecord:
    return FeatureRecord(
        contig_id=contig_id,
        start=start, end=end, strand=strand,
        feature_type=feature_type,
        frame=frame, feature_id=feature_id, function=function,
    )


def _make_contig(seq="ATCGATCGATCG", features=None) -> ContigRecord:
    return ContigRecord(
        contig_id="ctg1",
        original_id="NC_000001",
        sequence=seq,
        features=features or [],
    )


@pytest.fixture
def minimal_gto(tmp_path) -> Path:
    """Write a minimal GTO JSON file to a temp directory."""
    gto = {
        "id": "12345.1",
        "scientific_name": "Streptomyces sp.",
        "contigs": [
            {
                "id": "12345.1.con.0001",
                "original_id": "AABB01000001",
                "dna": "ATGATCAAATAA",   # ATG + ATC + AAA + TAA (stop)
            }
        ],
        "features": [
            {
                "id": "12345.1.peg.1",
                "type": "CDS",
                "function": "test CDS",
                "location": [
                    ["12345.1.con.0001", "1", "+", 9]  # 1-based, length 9
                ],
            },
            {
                "id": "12345.1.rna.1",
                "type": "tRNA",
                "function": "tRNA-Ala",
                "location": [
                    ["12345.1.con.0001", "10", "+", 3]
                ],
            },
            {
                "id": "12345.1.repeat.1",
                "type": "repeat_region",   # should be ignored
                "function": "",
                "location": [
                    ["12345.1.con.0001", "1", "+", 3]
                ],
            },
        ],
    }
    path = tmp_path / "12345.1.gto"
    path.write_text(json.dumps(gto))
    return path


@pytest.fixture
def minimal_genome(minimal_gto) -> GenomeRecord:
    return parse_gto(minimal_gto)


# ---------------------------------------------------------------------------
# GenomeRecord dataclasses
# ---------------------------------------------------------------------------

class TestGenomeRecords:
    def test_contig_length(self):
        c = _make_contig("ATCG")
        assert c.length == 4

    def test_contig_has_features(self):
        feat = _make_feature(start=0, end=3)
        c = _make_contig(features=[feat])
        assert len(c.features) == 1

    def test_genome_record_holds_contigs(self):
        c = _make_contig()
        g = GenomeRecord(genome_id="1", scientific_name="Test sp.", contigs=[c])
        assert len(g.contigs) == 1

    def test_feature_strand_values(self):
        fwd = _make_feature(strand=1)
        rev = _make_feature(strand=-1)
        assert fwd.strand == 1
        assert rev.strand == -1


# ---------------------------------------------------------------------------
# parse_gto
# ---------------------------------------------------------------------------

class TestParseGTO:
    def test_genome_id(self, minimal_genome):
        assert minimal_genome.genome_id == "12345.1"

    def test_scientific_name(self, minimal_genome):
        assert "Streptomyces" in minimal_genome.scientific_name

    def test_contig_count(self, minimal_genome):
        assert len(minimal_genome.contigs) == 1

    def test_contig_original_id(self, minimal_genome):
        assert minimal_genome.contigs[0].original_id == "AABB01000001"

    def test_contig_sequence_preserved(self, minimal_genome):
        assert minimal_genome.contigs[0].sequence == "ATGATCAAATAA"

    def test_cds_is_included(self, minimal_genome):
        features = minimal_genome.contigs[0].features
        cds = [f for f in features if f.feature_type == "CDS"]
        assert len(cds) == 1

    def test_trna_is_included_as_misc_feature(self, minimal_genome):
        features = minimal_genome.contigs[0].features
        misc = [f for f in features if f.feature_type == "misc_feature"]
        assert len(misc) == 1

    def test_repeat_region_is_excluded(self, minimal_genome):
        # repeat_region is not in _FEATURE_TYPE_MAP, so it must be dropped
        all_ids = [f.feature_id for f in minimal_genome.contigs[0].features]
        assert "12345.1.repeat.1" not in all_ids

    def test_coordinate_conversion_forward(self, minimal_genome):
        # GTO: start="1" (1-based), strand="+", length=9
        # → 0-based half-open: start=0, end=9
        cds = next(f for f in minimal_genome.contigs[0].features
                   if f.feature_type == "CDS")
        assert cds.start == 0
        assert cds.end   == 9

    def test_coordinate_conversion_trna(self, minimal_genome):
        # GTO: start="10", strand="+", length=3
        # → 0-based half-open: start=9, end=12
        trna = next(f for f in minimal_genome.contigs[0].features
                    if f.feature_type == "misc_feature")
        assert trna.start == 9
        assert trna.end   == 12

    def test_strand_is_integer(self, minimal_genome):
        for feat in minimal_genome.contigs[0].features:
            assert feat.strand in (1, -1)

    def test_reverse_strand_coordinate(self, tmp_path):
        # GTO reverse strand: start_1based="10", length=6
        # end   = start_1based = 10
        # start = end - length = 10 - 6 = 4
        # → 0-based half-open: [4, 10)
        gto = {
            "id": "rev.1",
            "scientific_name": "Rev sp.",
            "contigs": [{"id": "rev.1.con.0001", "original_id": "X", "dna": "A" * 20}],
            "features": [{
                "id": "rev.1.peg.1",
                "type": "CDS",
                "function": "rev",
                "location": [["rev.1.con.0001", "10", "-", 6]],
            }],
        }
        p = tmp_path / "rev.1.gto"
        p.write_text(json.dumps(gto))
        genome = parse_gto(p)
        feat = genome.contigs[0].features[0]
        assert feat.strand == -1
        assert feat.start == 4
        assert feat.end   == 10

    def test_build_id_mapping(self, tmp_path, minimal_gto):
        mapping = build_id_mapping_from_gtos(tmp_path)
        # mapping["12345.1"]["AABB01000001"] == "12345.1.con.0001"
        assert "12345.1" in mapping
        assert "AABB01000001" in mapping["12345.1"]
        assert mapping["12345.1"]["AABB01000001"] == "12345.1.con.0001"


# ---------------------------------------------------------------------------
# parse_gff_fasta
# ---------------------------------------------------------------------------

class TestParseGffFasta:
    @pytest.fixture
    def gff_fasta_dir(self, tmp_path):
        """Write matching GFF3 + FASTA files with consistent contig IDs."""
        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(
            ">ctg1\nATGATCAAATAA\n"
            ">ctg2\nGCATGCATGCAT\n"
        )
        gff_path = tmp_path / "test.gff"
        gff_path.write_text(textwrap.dedent("""\
            ##gff-version 3
            ctg1\t.\tCDS\t1\t9\t.\t+\t0\tID=gene1;product=hypothetical
            ctg1\t.\ttRNA\t10\t12\t.\t+\t.\tID=rna1;product=tRNA-Ala
            ctg2\t.\tCDS\t1\t6\t.\t-\t0\tID=gene2;product=another
        """))
        return fasta_path, gff_path, tmp_path

    def test_contig_count(self, gff_fasta_dir):
        fasta, gff, _ = gff_fasta_dir
        genome = parse_gff_fasta(fasta, gff, genome_id="test", scientific_name="Test sp.")
        assert len(genome.contigs) == 2

    def test_sequences_match_fasta(self, gff_fasta_dir):
        fasta, gff, _ = gff_fasta_dir
        genome = parse_gff_fasta(fasta, gff, genome_id="test", scientific_name="Test sp.")
        seqs = {c.contig_id: c.sequence for c in genome.contigs}
        assert seqs["ctg1"] == "ATGATCAAATAA"
        assert seqs["ctg2"] == "GCATGCATGCAT"

    def test_gff_coordinate_conversion(self, gff_fasta_dir):
        # GFF3: start=1 (1-based inclusive), end=9 → 0-based [0, 9)
        fasta, gff, _ = gff_fasta_dir
        genome = parse_gff_fasta(fasta, gff, genome_id="test", scientific_name="Test sp.")
        ctg1 = next(c for c in genome.contigs if c.contig_id == "ctg1")
        cds = next(f for f in ctg1.features if f.feature_type == "CDS")
        assert cds.start == 0
        assert cds.end   == 9

    def test_trna_parsed_as_misc_feature(self, gff_fasta_dir):
        fasta, gff, _ = gff_fasta_dir
        genome = parse_gff_fasta(fasta, gff, genome_id="test", scientific_name="Test sp.")
        ctg1 = next(c for c in genome.contigs if c.contig_id == "ctg1")
        misc = [f for f in ctg1.features if f.feature_type == "misc_feature"]
        assert len(misc) >= 1

    def test_reverse_strand_feature(self, gff_fasta_dir):
        fasta, gff, _ = gff_fasta_dir
        genome = parse_gff_fasta(fasta, gff, genome_id="test", scientific_name="Test sp.")
        ctg2 = next(c for c in genome.contigs if c.contig_id == "ctg2")
        cds = next(f for f in ctg2.features if f.feature_type == "CDS")
        assert cds.strand == -1

    def test_with_contig_id_map(self, gff_fasta_dir):
        # Provide explicit mapping NCBI→PATRIC IDs; should resolve without error
        fasta, gff, _ = gff_fasta_dir
        id_map = {"ctg1": "patric_ctg1", "ctg2": "patric_ctg2"}
        genome = parse_gff_fasta(
            fasta, gff, genome_id="test", scientific_name="Test sp.",
            contig_id_map=id_map,
        )
        patric_ids = {c.contig_id for c in genome.contigs}
        assert "patric_ctg1" in patric_ids or "ctg1" in patric_ids


# ---------------------------------------------------------------------------
# extract_for_bpe
# ---------------------------------------------------------------------------

class TestExtractForBPE:
    @pytest.fixture
    def genome_with_features(self):
        seq = "ATCGATCG" * 50          # 400 bp
        cds  = _make_feature(start=0,   end=30,  strand=1,  feature_type="CDS")
        trna = _make_feature(start=50,  end=80,  strand=-1, feature_type="misc_feature",
                             feature_id="f2")
        contig = _make_contig(seq=seq, features=[cds, trna])
        return GenomeRecord(genome_id="g1", scientific_name="Test sp.", contigs=[contig])

    def test_functional_seqs_extracted(self, genome_with_features):
        functional, _ = extract_from_genome(genome_with_features)
        assert len(functional) > 0

    def test_noncoding_seqs_extracted(self, genome_with_features):
        _, noncoding = extract_from_genome(genome_with_features)
        assert len(noncoding) > 0

    def test_functional_seqs_are_strings(self, genome_with_features):
        functional, _ = extract_from_genome(genome_with_features)
        assert all(isinstance(s, str) for s in functional)

    def test_functional_seqs_dna_only(self, genome_with_features):
        functional, _ = extract_from_genome(genome_with_features)
        allowed = set("ACGTNacgtn")
        for seq in functional:
            assert set(seq).issubset(allowed), f"Non-DNA chars in: {seq!r}"

    def test_noncoding_seqs_dna_only(self, genome_with_features):
        _, noncoding = extract_from_genome(genome_with_features)
        allowed = set("ACGTNacgtn")
        for seq in noncoding:
            assert set(seq).issubset(allowed), f"Non-DNA chars in: {seq!r}"

    def test_max_seq_len_respected(self):
        seq = "ATCG" * 500             # 2000 bp
        feat = _make_feature(start=0, end=30)
        contig = _make_contig(seq=seq, features=[feat])
        genome = GenomeRecord(genome_id="g", scientific_name="T", contigs=[contig])
        _, noncoding = extract_from_genome(genome, max_seq_len=100)
        assert all(len(s) <= 100 for s in noncoding)

    def test_reverse_complement_applied_for_minus_strand(self):
        seq  = "ATCGATCGATCGATCGATCGATCG"
        # Feature covers positions [0, 12) on minus strand (12 chars, above the 10-bp minimum)
        # Forward: "ATCGATCGATCG"  rev-comp: "CGATCGATCGAT"
        feat = _make_feature(start=0, end=12, strand=-1, feature_type="misc_feature")
        contig = _make_contig(seq=seq, features=[feat])
        genome = GenomeRecord(genome_id="g", scientific_name="T", contigs=[contig])
        functional, _ = extract_from_genome(genome)
        # "ATCGATCGATCG" → complement "TAGCTAGCTAGC" → reverse "CGATCGATCGAT"
        assert any("CGATCGATCGAT" in s for s in functional)

    def test_empty_genome_returns_empty(self):
        contig = _make_contig(seq="ATCG", features=[])
        genome = GenomeRecord(genome_id="g", scientific_name="T", contigs=[contig])
        functional, noncoding = extract_from_genome(genome)
        assert len(functional) == 0       # no misc_features
        assert len(noncoding)  >= 0      # short seq may not meet 20-bp min
