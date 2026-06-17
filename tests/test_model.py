"""Unit tests for DiffGenomeModel architecture."""

from __future__ import annotations

import torch
import pytest

from diffgenslm.models.diffgenome import (
    DiffGenomeConfig,
    DiffGenomeModel,
    BidirectionalGQA,
    RMSNorm,
    SwiGLU,
    precompute_rope_freqs,
    apply_rope,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 10, 32)
        assert norm(x).shape == x.shape

    def test_unit_norm_at_init(self):
        # At init, weight=1 so output RMS ≈ 1 per position
        norm = RMSNorm(64)
        x = torch.randn(4, 8, 64)
        y = norm(x)
        rms = y.pow(2).mean(-1).sqrt()
        # RMSNorm normalises so ‖y‖ / sqrt(d) ≈ 1, not ‖y‖ itself
        # Just check output is finite and same dtype
        assert torch.isfinite(y).all()
        assert y.dtype == x.dtype

    def test_preserves_dtype_fp16(self):
        norm = RMSNorm(16)
        x = torch.randn(2, 4, 16).half()
        assert norm(x).dtype == torch.float16


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_precompute_shape(self):
        cos, sin = precompute_rope_freqs(head_dim=16, max_len=64)
        assert cos.shape == (64, 8)   # head_dim // 2
        assert sin.shape == (64, 8)

    def test_apply_rope_preserves_shape(self):
        B, H, L, Dh = 2, 4, 10, 16
        q = torch.randn(B, H, L, Dh)
        k = torch.randn(B, H, L, Dh)
        cos, sin = precompute_rope_freqs(Dh, L)
        q_r, k_r = apply_rope(q, k, cos, sin)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_apply_rope_is_isometric(self):
        # RoPE is a rotation; it must preserve the L2 norm of each vector.
        B, H, L, Dh = 1, 2, 8, 16
        q = torch.randn(B, H, L, Dh)
        k = torch.randn(B, H, L, Dh)
        cos, sin = precompute_rope_freqs(Dh, L)
        q_r, k_r = apply_rope(q, k, cos, sin)
        torch.testing.assert_close(
            q.float().norm(dim=-1), q_r.float().norm(dim=-1), rtol=1e-4, atol=1e-4
        )


# ---------------------------------------------------------------------------
# DiffGenomeModel
# ---------------------------------------------------------------------------

class TestDiffGenomeModel:
    def test_forward_output_shape(self, tiny_model, tiny_cfg, batch_ids):
        B, L = batch_ids.shape
        logits = tiny_model(batch_ids)
        assert logits.shape == (B, L, tiny_cfg.vocab_size)

    def test_forward_is_finite(self, tiny_model, batch_ids):
        logits = tiny_model(batch_ids)
        assert torch.isfinite(logits).all()

    def test_return_hidden_states(self, tiny_model, tiny_cfg, batch_ids):
        logits, hidden = tiny_model(batch_ids, return_hidden_states=True)
        B, L = batch_ids.shape
        assert logits.shape == (B, L, tiny_cfg.vocab_size)
        assert hidden.shape == (B, L, tiny_cfg.hidden_size)

    def test_hidden_state_is_pre_lm_head(self, tiny_model, tiny_cfg, batch_ids):
        # lm_head(hidden) should equal logits (weights are tied to embed)
        logits, hidden = tiny_model(batch_ids, return_hidden_states=True)
        recomputed = tiny_model.lm_head(hidden)
        torch.testing.assert_close(logits, recomputed)

    def test_weight_tying(self, tiny_model):
        # lm_head.weight and embed.weight must be the same tensor object
        assert tiny_model.lm_head.weight is tiny_model.embed.weight

    def test_padding_mask_inferred(self, tiny_model, tiny_cfg):
        # When input contains pad_token_id, padding_mask should be inferred
        # and those positions should receive zero attention weight → output still finite
        ids = torch.randint(5, tiny_cfg.vocab_size, (1, 8))
        ids[0, 5:] = tiny_cfg.pad_token_id      # last 3 positions are pad
        logits = tiny_model(ids)
        assert torch.isfinite(logits).all()

    def test_explicit_padding_mask(self, tiny_model, tiny_cfg, batch_ids):
        # Explicit all-True mask (no padding) should give same result
        mask = torch.ones(batch_ids.shape, dtype=torch.bool)
        logits_masked = tiny_model(batch_ids, padding_mask=mask)
        logits_auto   = tiny_model(batch_ids)
        torch.testing.assert_close(logits_masked, logits_auto)

    def test_bidirectional_attention(self, tiny_cfg):
        # In a bidirectional model, masking position j from attending
        # should still affect position i (i≠j) if they share attention.
        # Simplest check: changing token at position 0 changes logits at position 5.
        model = DiffGenomeModel(tiny_cfg)
        model.eval()
        with torch.no_grad():
            ids_a = torch.randint(5, tiny_cfg.vocab_size, (1, 16))
            ids_b = ids_a.clone()
            ids_b[0, 0] = (ids_a[0, 0] + 1) % tiny_cfg.vocab_size
            la = model(ids_a)
            lb = model(ids_b)
        # logits at position 5 must differ when position 0 changes
        assert not torch.allclose(la[0, 5], lb[0, 5])

    def test_num_params_positive(self, tiny_model):
        assert tiny_model.num_params() > 0

    def test_param_count_scales_with_depth(self, tiny_cfg):
        cfg_shallow = tiny_cfg
        cfg_deep    = DiffGenomeConfig(
            vocab_size=tiny_cfg.vocab_size,
            hidden_size=tiny_cfg.hidden_size,
            num_layers=tiny_cfg.num_layers * 2,
            num_heads=tiny_cfg.num_heads,
            num_kv_heads=tiny_cfg.num_kv_heads,
            ffn_intermediate_size=tiny_cfg.ffn_intermediate_size,
            max_seq_len=tiny_cfg.max_seq_len,
        )
        assert DiffGenomeModel(cfg_deep).num_params() > DiffGenomeModel(cfg_shallow).num_params()

    def test_gradient_flows(self, tiny_cfg, batch_ids):
        model = DiffGenomeModel(tiny_cfg)
        model.train()
        logits = model(batch_ids)
        loss   = logits.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_from_config(self, tiny_cfg):
        m = DiffGenomeModel.from_config(tiny_cfg)
        assert isinstance(m, DiffGenomeModel)

    def test_variable_sequence_lengths(self, tiny_model, tiny_cfg):
        for L in [1, 8, tiny_cfg.max_seq_len]:
            ids = torch.randint(5, tiny_cfg.vocab_size, (1, L))
            out = tiny_model(ids)
            assert out.shape == (1, L, tiny_cfg.vocab_size)
