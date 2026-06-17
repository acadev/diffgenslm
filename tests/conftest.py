"""
Shared pytest fixtures for DiffGenSLM tests.

All fixtures that are needed by more than one test module live here so
they are available automatically via conftest discovery.
"""

from __future__ import annotations

import json

import pytest
import torch

from diffgenslm.models.diffgenome import DiffGenomeConfig, DiffGenomeModel


# ---------------------------------------------------------------------------
# Tiny model config — fast enough for CPU tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_cfg() -> DiffGenomeConfig:
    """Minimal DiffGenomeConfig that still exercises all code paths."""
    return DiffGenomeConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_intermediate_size=64,
        rope_theta=10_000.0,
        max_seq_len=32,
        dropout=0.0,       # deterministic during tests
        pad_token_id=0,
        mask_token_id=4,
        max_rel_dist=16,           # small for tests (seq_len=32, so 16 covers half)
        same_strand_bias_init=0.1,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg: DiffGenomeConfig) -> DiffGenomeModel:
    """Untrained tiny model in eval mode (shared across the session)."""
    model = DiffGenomeModel(tiny_cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Checkpoint + tokeniser directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory, tiny_cfg: DiffGenomeConfig):
    """Serialise a tiny model to a temporary checkpoint file."""
    model = DiffGenomeModel(tiny_cfg)
    tmp = tmp_path_factory.mktemp("ckpt")
    path = tmp / "checkpoint.pt"
    torch.save(
        {
            "epoch": 1,
            "global_step": 10,
            "model": model.state_dict(),
            "optimizer": {},
            "scaler": {},
            "scheduler": {},
            "best_val": 9.0,
            "model_config": tiny_cfg.__dict__,
        },
        path,
    )
    return path


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory, tiny_cfg: DiffGenomeConfig):
    """
    Minimal tokeniser directory with codon/vocab.json.

    Token IDs are kept within tiny_cfg.vocab_size so the embedding table
    won't throw an index-out-of-range error.
    """
    tmp = tmp_path_factory.mktemp("tokenizer")
    codon_dir = tmp / "codon"
    codon_dir.mkdir()

    vocab = {
        "<pad>":  0,
        "<bos>":  1,
        "<eos>":  2,
        "<unk>":  3,
        "<mask>": 4,
        "A":      5,
        "T":      6,
        "C":      7,
        "G":      8,
        # A few codon tokens to exercise decode paths
        "ATG":    9,
        "TAA":    10,
        "TAG":    11,
        "TGA":    12,
    }
    (codon_dir / "vocab.json").write_text(json.dumps(vocab))
    return tmp


# ---------------------------------------------------------------------------
# Convenience batch tensors
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_ids(tiny_cfg: DiffGenomeConfig) -> torch.Tensor:
    """Random token IDs [2, 16] within the tiny model's vocab."""
    torch.manual_seed(0)
    return torch.randint(5, tiny_cfg.vocab_size, (2, 16))
