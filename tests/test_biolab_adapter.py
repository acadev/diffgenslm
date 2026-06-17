"""
Unit tests for the biolab LM adapter (DiffGenSLM + NuclCharTokenizer).

These tests do NOT require a trained model or a real SentencePiece model.
They use the tiny fixtures from conftest.py (random weights, minimal vocab).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from diffgenslm.eval.biolab_model import (
    DiffGenSLMConfig,
    DiffGenSLM,
    NuclCharTokenizer,
)


# ---------------------------------------------------------------------------
# NuclCharTokenizer
# ---------------------------------------------------------------------------

class TestNuclCharTokenizer:
    @pytest.fixture
    def tokenizer(self, tokenizer_dir):
        vocab_path = tokenizer_dir / "codon" / "vocab.json"
        return NuclCharTokenizer(vocab_path)

    def test_encode_returns_ints(self, tokenizer):
        ids = tokenizer.encode("ATCG")
        assert all(isinstance(i, int) for i in ids)

    def test_encode_length_matches_sequence(self, tokenizer):
        seq = "ATCGATCG"
        assert len(tokenizer.encode(seq)) == len(seq)

    def test_known_bases_not_unk(self, tokenizer):
        unk = tokenizer.unk_id
        for base in "ATCG":
            ids = tokenizer.encode(base)
            assert ids[0] != unk, f"Base {base!r} should have its own token"

    def test_unknown_base_maps_to_unk(self, tokenizer):
        ids = tokenizer.encode("N")
        assert ids[0] == tokenizer.unk_id

    def test_case_insensitive(self, tokenizer):
        upper = tokenizer.encode("ATCG")
        lower = tokenizer.encode("atcg")
        assert upper == lower

    def test_max_length_truncates(self, tokenizer):
        ids = tokenizer.encode("ATCGATCG", max_length=4)
        assert len(ids) == 4

    def test_tokenize_returns_single_chars(self, tokenizer):
        pieces = tokenizer.tokenize("ATCG")
        assert pieces == ["A", "T", "C", "G"]

    def test_tokenize_len_equals_sequence_len(self, tokenizer):
        seq = "GCATGCAT"
        assert len(tokenizer.tokenize(seq)) == len(seq)

    def test_decode_roundtrip(self, tokenizer):
        seq = "ATCG"
        ids = tokenizer.encode(seq)
        decoded = tokenizer.decode(ids)
        assert decoded == seq

    def test_pad_id_is_zero(self, tokenizer):
        assert tokenizer.pad_id == 0

    def test_mask_id_is_four(self, tokenizer):
        assert tokenizer.mask_id == 4

    def test_len_reflects_vocab_size(self, tokenizer, tokenizer_dir):
        with open(tokenizer_dir / "codon" / "vocab.json") as fh:
            vocab = json.load(fh)
        assert len(tokenizer) == len(vocab)

    def test_missing_vocab_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DiffGenSLMConfig(
                checkpoint_path="fake.pt",
                tokenizer_dir=str(tmp_path),   # no codon/vocab.json here
            )
            # Instantiation of DiffGenSLM would raise; test config validation path
            # via the model constructor
            DiffGenSLM(DiffGenSLMConfig(
                checkpoint_path="fake.pt",
                tokenizer_dir=str(tmp_path),
            ))


# ---------------------------------------------------------------------------
# DiffGenSLMConfig
# ---------------------------------------------------------------------------

class TestDiffGenSLMConfig:
    def test_defaults(self, checkpoint_path, tokenizer_dir):
        cfg = DiffGenSLMConfig(
            checkpoint_path=str(checkpoint_path),
            tokenizer_dir=str(tokenizer_dir),
        )
        assert cfg.name == "DiffGenSLM"
        assert cfg.max_length == 2048
        assert cfg.half_precision is False
        assert cfg.num_sample_steps == 64
        assert cfg.sample_schedule == "cosine"

    def test_json_roundtrip(self, checkpoint_path, tokenizer_dir, tmp_path):
        cfg = DiffGenSLMConfig(
            checkpoint_path=str(checkpoint_path),
            tokenizer_dir=str(tokenizer_dir),
            max_length=512,
        )
        out = tmp_path / "cfg.json"
        cfg.write_json(out)
        loaded = DiffGenSLMConfig.from_json(out)
        assert loaded.max_length == 512
        assert loaded.name == "DiffGenSLM"

    def test_yaml_roundtrip(self, checkpoint_path, tokenizer_dir, tmp_path):
        cfg = DiffGenSLMConfig(
            checkpoint_path=str(checkpoint_path),
            tokenizer_dir=str(tokenizer_dir),
            sample_temperature=0.8,
        )
        out = tmp_path / "cfg.yaml"
        cfg.write_yaml(out)
        loaded = DiffGenSLMConfig.from_yaml(out)
        assert loaded.sample_temperature == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# DiffGenSLM adapter
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapter(checkpoint_path, tokenizer_dir):
    cfg = DiffGenSLMConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_dir=str(tokenizer_dir),
        max_length=16,
        num_sample_steps=4,     # keep tests fast
    )
    return DiffGenSLM(cfg)


class TestDiffGenSLMAdapter:
    # ── protocol attributes ─────────────────────────────────────────────
    def test_model_input(self, adapter):
        assert adapter.model_input == "dna"

    def test_model_encoding(self, adapter):
        assert adapter.model_encoding == "char"

    def test_tokenizer_property(self, adapter):
        assert isinstance(adapter.tokenizer, NuclCharTokenizer)

    def test_device_property(self, adapter):
        assert isinstance(adapter.device, torch.device)

    def test_tokenizer_config_is_dict(self, adapter):
        assert isinstance(adapter.tokenizer_config, dict)

    def test_dataloader_config_is_dict(self, adapter):
        assert isinstance(adapter.dataloader_config, dict)

    # ── generate_embeddings ─────────────────────────────────────────────
    def test_embeddings_returns_list(self, adapter):
        outputs = adapter.generate_embeddings(["ATCGATCG"])
        assert isinstance(outputs, list)
        assert len(outputs) == 1

    def test_embeddings_embedding_shape(self, adapter, tiny_cfg):
        seq = "ATCGATCG"
        outputs = adapter.generate_embeddings([seq])
        emb = outputs[0].embedding
        # [L, hidden_size] — L = len(seq) capped at max_length
        L = min(len(seq), adapter.config.max_length)
        assert emb.shape == (L, tiny_cfg.hidden_size)

    def test_embeddings_logits_shape(self, adapter, tiny_cfg):
        seq = "ATCG"
        outputs = adapter.generate_embeddings([seq])
        log = outputs[0].logits
        L = min(len(seq), adapter.config.max_length)
        assert log.shape == (L, tiny_cfg.vocab_size)

    def test_embeddings_dtype_is_float32(self, adapter):
        outputs = adapter.generate_embeddings(["ATCG"])
        assert outputs[0].embedding.dtype == np.float32

    def test_embeddings_batch(self, adapter, tiny_cfg):
        seqs = ["ATCG", "GCATGCAT", "A"]
        outputs = adapter.generate_embeddings(seqs)
        assert len(outputs) == len(seqs)
        for seq, out in zip(seqs, outputs):
            L = min(len(seq), adapter.config.max_length)
            assert out.embedding.shape == (L, tiny_cfg.hidden_size)

    def test_embeddings_are_finite(self, adapter):
        outputs = adapter.generate_embeddings(["ATCGATCGATCG"])
        assert np.isfinite(outputs[0].embedding).all()

    def test_embeddings_truncation(self, adapter, tiny_cfg):
        # Sequence longer than max_length should be truncated
        long_seq = "ATCG" * 20          # 80 chars > max_length=16
        outputs  = adapter.generate_embeddings([long_seq])
        assert outputs[0].embedding.shape[0] == adapter.config.max_length

    def test_embeddings_differ_across_sequences(self, adapter):
        # Different sequences should (almost certainly) get different embeddings
        out_a = adapter.generate_embeddings(["AAAAAAAAAA"])
        out_b = adapter.generate_embeddings(["TTTTTTTTTT"])
        # Mean-pooled vectors should differ
        assert not np.allclose(
            out_a[0].embedding.mean(0),
            out_b[0].embedding.mean(0),
        )

    # ── generate_sequences ──────────────────────────────────────────────
    def test_sequences_returns_list(self, adapter):
        outputs = adapter.generate_sequences([""])
        assert isinstance(outputs, list)
        assert len(outputs) == 1

    def test_sequences_has_sequence_field(self, adapter):
        outputs = adapter.generate_sequences([""])
        assert outputs[0].sequence is not None
        assert isinstance(outputs[0].sequence, str)

    def test_sequences_no_mask_tokens(self, adapter, tiny_cfg):
        # After generation, the decoded string must not contain any mask-token character
        outputs   = adapter.generate_sequences([""])
        seq_str   = outputs[0].sequence
        mask_char = adapter.tokenizer.decode([tiny_cfg.mask_token_id])
        assert mask_char not in seq_str

    def test_sequences_fixed_context_preserved(self, adapter):
        # N-free context characters must appear in the output at the same positions
        context = "ATCG" + "NNNN"        # first 4 are fixed, last 4 are free
        outputs = adapter.generate_sequences([context])
        seq_out = outputs[0].sequence
        for i, (c_in, c_out) in enumerate(zip(context, seq_out)):
            if c_in != "N":
                assert c_in == c_out, (
                    f"Fixed position {i} changed: {c_in!r} → {c_out!r}"
                )

    def test_sequences_logits_shape(self, adapter, tiny_cfg):
        outputs = adapter.generate_sequences([""])
        log = outputs[0].logits
        # logits shape: [L, vocab_size]
        assert log.ndim == 2
        assert log.shape[1] == tiny_cfg.vocab_size

    def test_sequences_batch(self, adapter):
        outputs = adapter.generate_sequences(["", "ATCG", "NNNN"])
        assert len(outputs) == 3
        for out in outputs:
            assert out.sequence is not None
