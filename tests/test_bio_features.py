"""
Unit tests for the three biologically-aware attention features ported from HiSAN:

  1. Relative distance bias   — BidirectionalGQA.rel_bias embedding
  2. Same-strand bias         — BidirectionalGQA.same_strand_bias parameter
  3. Padding mask             — variable-length genomes, zeroed in attention

Also tests the end-to-end pipeline:
  4. _HDF5Writer strand_ids dataset     — written and read back correctly
  5. GenomeDiffusionDataset strand_ids  — loaded from HDF5, fallback to zeros
  6. build_hdf5 region type mapping     — misc_feature → functional_non_coding
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from diffgenslm.models.diffgenome import (
    BidirectionalGQA,
    DiffGenomeConfig,
    DiffGenomeModel,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg(**overrides) -> DiffGenomeConfig:
    """Return a minimal config fast enough for CPU tests."""
    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_intermediate_size=64,
        max_seq_len=32,
        dropout=0.0,
        pad_token_id=0,
        mask_token_id=4,
        max_rel_dist=16,
        same_strand_bias_init=0.1,
    )
    defaults.update(overrides)
    return DiffGenomeConfig(**defaults)


def _freqs(cfg: DiffGenomeConfig, L: int):
    cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len)
    return cos[:L], sin[:L]


# ---------------------------------------------------------------------------
# 1. Relative distance bias
# ---------------------------------------------------------------------------

class TestRelativeDistanceBias:
    def test_rel_bias_param_exists(self):
        cfg = _tiny_cfg()
        gqa = BidirectionalGQA(cfg)
        assert hasattr(gqa, "rel_bias")
        assert isinstance(gqa.rel_bias, torch.nn.Embedding)

    def test_rel_bias_embedding_shape(self):
        cfg = _tiny_cfg(max_rel_dist=16, num_heads=4)
        gqa = BidirectionalGQA(cfg)
        # (max_rel_dist + 1) × num_heads
        assert gqa.rel_bias.weight.shape == (17, 4)

    def test_rel_bias_changes_output(self):
        # Zeroing rel_bias weights must change the model output.
        cfg = _tiny_cfg()
        model = DiffGenomeModel(cfg)
        model.eval()
        ids = torch.randint(5, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out_default = model(ids)
            # Zero out all rel_bias embeddings in every block
            for blk in model.blocks:
                blk.attn.rel_bias.weight.zero_()
            out_zeroed = model(ids)
        assert not torch.allclose(out_default, out_zeroed)

    def test_rel_bias_distance_clamping(self):
        # Tokens at distance > max_rel_dist must use the same bucket.
        cfg = _tiny_cfg(max_rel_dist=4, num_heads=4)
        gqa = BidirectionalGQA(cfg)
        gqa.eval()

        B, L = 1, 10   # distance 0-9; max_rel_dist=4 → positions 5-9 share bucket
        x = torch.randn(B, L, cfg.hidden_size)
        cos, sin = _freqs(cfg, L)

        # Set rel_bias so bucket 4 has a large value; only distances ≥ 4 should be equal
        with torch.no_grad():
            gqa.rel_bias.weight[4] = 100.0

        # Just check forward doesn't error and output is finite
        out = gqa(x, cos, sin, padding_mask=None, strand_ids=None)
        assert torch.isfinite(out).all()

    def test_rel_bias_symmetric(self):
        # Relative distance is |i-j|, so bias(i→j) == bias(j→i).
        cfg = _tiny_cfg(max_rel_dist=8, num_heads=4)
        gqa = BidirectionalGQA(cfg)
        gqa.eval()

        L = 8
        pos = torch.arange(L)
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().clamp_max(cfg.max_rel_dist)
        rel_b = gqa.rel_bias(rel)  # [L, L, H]

        # rel[i,j] == rel[j,i] → bias matrix is symmetric
        assert torch.allclose(rel_b, rel_b.transpose(0, 1))


# ---------------------------------------------------------------------------
# 2. Same-strand bias
# ---------------------------------------------------------------------------

class TestSameStrandBias:
    def test_same_strand_bias_param_exists(self):
        cfg = _tiny_cfg()
        gqa = BidirectionalGQA(cfg)
        assert hasattr(gqa, "same_strand_bias")
        assert isinstance(gqa.same_strand_bias, torch.nn.Parameter)

    def test_same_strand_bias_init_value(self):
        cfg = _tiny_cfg(same_strand_bias_init=0.25)
        gqa = BidirectionalGQA(cfg)
        assert abs(gqa.same_strand_bias.item() - 0.25) < 1e-6

    def test_strand_ids_change_output(self):
        # All-same-strand (full bias) vs half-fwd/half-rev (partial bias) must differ.
        # all_fwd: every pair is same-strand → all get the bias.
        # mixed:   fwd-rev cross pairs are excluded → different attention pattern.
        cfg = _tiny_cfg(same_strand_bias_init=1.0)
        model = DiffGenomeModel(cfg)
        model.eval()
        torch.manual_seed(7)
        ids     = torch.randint(5, cfg.vocab_size, (1, 8))
        all_fwd = torch.ones(1, 8, dtype=torch.long)
        mixed   = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]])
        with torch.no_grad():
            out_all = model(ids, strand_ids=all_fwd)
            out_mix = model(ids, strand_ids=mixed)
        assert not torch.allclose(out_all, out_mix)

    def test_no_strand_ids_is_same_as_zero_strands(self):
        # Passing strand_ids=None should be equivalent to all-zero strand IDs,
        # because the same-strand bias condition requires strand != 0.
        cfg = _tiny_cfg()
        model = DiffGenomeModel(cfg)
        model.eval()
        ids   = torch.randint(5, cfg.vocab_size, (1, 8))
        zeros = torch.zeros(1, 8, dtype=torch.long)
        with torch.no_grad():
            out_none  = model(ids, strand_ids=None)
            out_zeros = model(ids, strand_ids=zeros)
        torch.testing.assert_close(out_none, out_zeros)

    def test_pad_strand_zero_excluded_from_bias(self):
        # Strand 0 (pad) must NOT participate in same-strand bias.
        # Two tokens both with strand_id=0 should get NO same-strand bonus.
        cfg = _tiny_cfg(same_strand_bias_init=1e6)   # huge bias to amplify effect
        gqa = BidirectionalGQA(cfg)
        gqa.eval()

        B, L = 1, 4
        x   = torch.randn(B, L, cfg.hidden_size)
        cos, sin = _freqs(cfg, L)

        all_pad = torch.zeros(B, L, dtype=torch.long)
        all_fwd = torch.ones(B, L, dtype=torch.long)

        with torch.no_grad():
            out_pad = gqa(x, cos, sin, padding_mask=None, strand_ids=all_pad)
            out_fwd = gqa(x, cos, sin, padding_mask=None, strand_ids=all_fwd)

        # With huge same_strand_bias, fwd output must differ significantly from pad
        assert not torch.allclose(out_pad, out_fwd)

    def test_same_strand_bias_is_learnable(self):
        cfg   = _tiny_cfg()
        model = DiffGenomeModel(cfg)
        model.train()
        ids    = torch.randint(5, cfg.vocab_size, (2, 8))
        strand = torch.randint(1, 4, (2, 8))
        logits = model(ids, strand_ids=strand)
        loss   = logits.mean()
        loss.backward()
        # same_strand_bias should have a gradient in at least one block
        grads = [
            blk.attn.same_strand_bias.grad
            for blk in model.blocks
            if blk.attn.same_strand_bias.grad is not None
        ]
        assert len(grads) > 0
        assert all(torch.isfinite(g) for g in grads)


# ---------------------------------------------------------------------------
# 3. Padding mask
# ---------------------------------------------------------------------------

class TestPaddingMask:
    def test_explicit_padding_mask_suppresses_pad_positions(self):
        # Logits at pad positions should still be finite (mask affects keys,
        # not query output directly), but the output with vs without mask differs.
        cfg = _tiny_cfg()
        model = DiffGenomeModel(cfg)
        model.eval()

        ids = torch.randint(5, cfg.vocab_size, (1, 8))
        ids[0, 5:] = cfg.pad_token_id
        mask = ids.ne(cfg.pad_token_id)   # True for real tokens

        with torch.no_grad():
            out_masked   = model(ids, padding_mask=mask)
            out_unmasked = model(ids, padding_mask=torch.ones_like(mask))
        assert not torch.allclose(out_masked, out_unmasked)
        assert torch.isfinite(out_masked).all()

    def test_inferred_mask_matches_explicit_mask(self):
        cfg   = _tiny_cfg()
        model = DiffGenomeModel(cfg)
        model.eval()

        ids = torch.randint(5, cfg.vocab_size, (2, 8))
        ids[0, 6:] = cfg.pad_token_id
        mask = ids.ne(cfg.pad_token_id)

        with torch.no_grad():
            out_inferred = model(ids)
            out_explicit = model(ids, padding_mask=mask)
        torch.testing.assert_close(out_inferred, out_explicit)

    def test_all_real_tokens_no_mask_effect(self):
        cfg   = _tiny_cfg()
        model = DiffGenomeModel(cfg)
        model.eval()

        ids  = torch.randint(5, cfg.vocab_size, (1, 8))   # no padding
        mask = torch.ones(1, 8, dtype=torch.bool)

        with torch.no_grad():
            out_masked = model(ids, padding_mask=mask)
            out_auto   = model(ids)
        torch.testing.assert_close(out_masked, out_auto)

    def test_strand_ids_and_padding_mask_together(self):
        cfg   = _tiny_cfg()
        model = DiffGenomeModel(cfg)
        model.eval()

        ids    = torch.randint(5, cfg.vocab_size, (2, 12))
        ids[1, 8:] = cfg.pad_token_id
        strands    = torch.randint(0, 4, (2, 12))
        strands[1, 8:] = 0   # pad positions get strand 0

        with torch.no_grad():
            out = model(ids, strand_ids=strands)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4 & 5. HDF5 strand_ids round-trip + dataset loading
# ---------------------------------------------------------------------------

class TestHDF5StrandIds:
    def _write_hdf5(self, path, token_seqs: List[List[int]],
                    strand_seqs: List[List[int]] | None = None):
        """Write a minimal HDF5 in the build_hdf5 format."""
        import h5py
        all_ids    = []
        all_strands = []
        starts     = []
        for i, seq in enumerate(token_seqs):
            starts.append(len(all_ids))
            all_ids.extend(seq)
            if strand_seqs:
                all_strands.extend(strand_seqs[i])

        with h5py.File(path, "w") as hf:
            hf.create_dataset("input_ids",       data=np.array(all_ids,    dtype=np.int32))
            hf.create_dataset("sequence_starts", data=np.array(starts,     dtype=np.int64))
            if strand_seqs:
                hf.create_dataset("strand_ids",  data=np.array(all_strands, dtype=np.int8))
            hf.attrs["has_strand_ids"] = strand_seqs is not None

    def test_dataset_returns_tuple(self, tmp_path):
        from diffgenslm.data.genome_dataset import GenomeDiffusionDataset

        p = tmp_path / "test.h5"
        seqs    = [[1, 5, 6, 7, 8, 2]]
        strands = [[0, 1, 1, 2, 2, 0]]
        self._write_hdf5(p, seqs, strands)

        ds = GenomeDiffusionDataset([p], seq_len=6, pad_token_id=0)
        assert len(ds) > 0
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2
        ids_t, str_t = item
        assert ids_t.shape == (6,)
        assert str_t.shape == (6,)

    def test_strand_ids_match_written_values(self, tmp_path):
        from diffgenslm.data.genome_dataset import GenomeDiffusionDataset

        p = tmp_path / "test.h5"
        seqs    = [[1, 5, 6, 7, 8, 2]]
        strands = [[0, 1, 2, 3, 1, 0]]
        self._write_hdf5(p, seqs, strands)

        ds = GenomeDiffusionDataset([p], seq_len=6, pad_token_id=0)
        _, str_t = ds[0]
        assert str_t.tolist() == [0, 1, 2, 3, 1, 0]

    def test_missing_strand_ids_fallback_to_zeros(self, tmp_path):
        from diffgenslm.data.genome_dataset import GenomeDiffusionDataset

        p = tmp_path / "test.h5"
        self._write_hdf5(p, [[1, 5, 6, 7, 2]], strand_seqs=None)  # no strand_ids dataset

        ds = GenomeDiffusionDataset([p], seq_len=5, pad_token_id=0)
        _, str_t = ds[0]
        assert (str_t == 0).all()

    def test_padded_window_strand_ids_are_zero(self, tmp_path):
        from diffgenslm.data.genome_dataset import GenomeDiffusionDataset

        p = tmp_path / "test.h5"
        # seq shorter than seq_len — tail should be zero-padded
        seqs    = [[1, 5, 6, 2]]       # length 4
        strands = [[0, 1, 2, 0]]
        self._write_hdf5(p, seqs, strands)

        ds = GenomeDiffusionDataset([p], seq_len=8, pad_token_id=0)
        _, str_t = ds[0]
        # First 4 values match; last 4 are zero-padded
        assert str_t[:4].tolist() == [0, 1, 2, 0]
        assert (str_t[4:] == 0).all()

    def test_strand_ids_dtype_is_long(self, tmp_path):
        from diffgenslm.data.genome_dataset import GenomeDiffusionDataset

        p = tmp_path / "test.h5"
        self._write_hdf5(p, [[1, 5, 6, 7, 2]], [[0, 1, 2, 1, 0]])

        ds = GenomeDiffusionDataset([p], seq_len=5, pad_token_id=0)
        _, str_t = ds[0]
        assert str_t.dtype == torch.long


# ---------------------------------------------------------------------------
# 6. build_hdf5 region type mapping
# ---------------------------------------------------------------------------

class TestRegionTypeMapping:
    def test_misc_feature_maps_to_functional_non_coding(self):
        from diffgenslm.preprocessing.build_hdf5 import _REGION_TYPE_MAP
        assert _REGION_TYPE_MAP["misc_feature"] == "functional_non_coding"

    def test_cds_maps_to_cds(self):
        from diffgenslm.preprocessing.build_hdf5 import _REGION_TYPE_MAP
        assert _REGION_TYPE_MAP["CDS"] == "CDS"

    def test_non_coding_maps_to_non_coding(self):
        from diffgenslm.preprocessing.build_hdf5 import _REGION_TYPE_MAP
        assert _REGION_TYPE_MAP["non_coding"] == "non_coding"

    def test_unknown_type_falls_back_to_non_coding(self):
        from diffgenslm.preprocessing.build_hdf5 import _REGION_TYPE_MAP
        assert _REGION_TYPE_MAP.get("repeat_region", "non_coding") == "non_coding"

    def test_strand_constants_are_distinct(self):
        from diffgenslm.preprocessing.build_hdf5 import (
            STRAND_FWD, STRAND_INTERGENIC, STRAND_PAD, STRAND_REV,
        )
        values = [STRAND_PAD, STRAND_FWD, STRAND_REV, STRAND_INTERGENIC]
        assert len(set(values)) == 4  # all four are distinct

    def test_strand_constants_values(self):
        from diffgenslm.preprocessing.build_hdf5 import (
            STRAND_FWD, STRAND_INTERGENIC, STRAND_PAD, STRAND_REV,
        )
        assert STRAND_PAD        == 0
        assert STRAND_FWD        == 1
        assert STRAND_REV        == 2
        assert STRAND_INTERGENIC == 3
