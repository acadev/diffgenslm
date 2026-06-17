"""Unit tests for the discrete absorbing-state diffusion pipeline."""

from __future__ import annotations

import pytest
import torch

from diffgenslm.diffusion.loss import diffusion_loss, simple_loss
from diffgenslm.diffusion.process import forward_process
from diffgenslm.diffusion.sample import sample, infill, _unmask_fraction


# ---------------------------------------------------------------------------
# forward_process
# ---------------------------------------------------------------------------

class TestForwardProcess:
    def test_output_shapes(self, batch_ids, tiny_cfg):
        xt, mask, t = forward_process(batch_ids, tiny_cfg.mask_token_id,
                                      tiny_cfg.pad_token_id)
        assert xt.shape  == batch_ids.shape
        assert mask.shape == batch_ids.shape
        assert t.shape    == (batch_ids.shape[0],)

    def test_t_in_unit_interval(self, batch_ids, tiny_cfg):
        _, _, t = forward_process(batch_ids, tiny_cfg.mask_token_id,
                                  tiny_cfg.pad_token_id)
        assert (t >= 0).all() and (t <= 1).all()

    def test_masked_positions_get_mask_token(self, batch_ids, tiny_cfg):
        xt, mask, _ = forward_process(batch_ids, tiny_cfg.mask_token_id,
                                      tiny_cfg.pad_token_id)
        # Every position flagged as masked must hold mask_token_id in xt
        assert (xt[mask] == tiny_cfg.mask_token_id).all()

    def test_unmasked_positions_unchanged(self, batch_ids, tiny_cfg):
        xt, mask, _ = forward_process(batch_ids, tiny_cfg.mask_token_id,
                                      tiny_cfg.pad_token_id)
        assert (xt[~mask] == batch_ids[~mask]).all()

    def test_pad_positions_never_masked(self, tiny_cfg):
        ids = torch.randint(5, tiny_cfg.vocab_size, (2, 16))
        ids[:, 8:] = tiny_cfg.pad_token_id     # last 8 positions are padding
        _, mask, _ = forward_process(ids, tiny_cfg.mask_token_id,
                                     tiny_cfg.pad_token_id)
        pad_positions = ids.eq(tiny_cfg.pad_token_id)
        assert not mask[pad_positions].any(), "Pad positions must never be masked"

    def test_masking_fraction_close_to_t(self, tiny_cfg):
        # Over many samples, average mask fraction ≈ average t
        torch.manual_seed(42)
        ids = torch.randint(5, tiny_cfg.vocab_size, (256, 16))
        _, mask, t = forward_process(ids, tiny_cfg.mask_token_id,
                                     tiny_cfg.pad_token_id)
        observed_frac = mask.float().mean(dim=1)   # [B]
        # E[mask_frac | t] = t; average over batch
        assert abs(observed_frac.mean().item() - t.mean().item()) < 0.05

    def test_fixed_t_tensor(self, batch_ids, tiny_cfg):
        B = batch_ids.shape[0]
        t_fixed = torch.full((B,), 1.0)           # mask everything
        _, mask, t_out = forward_process(
            batch_ids, tiny_cfg.mask_token_id, tiny_cfg.pad_token_id,
            t_tensor=t_fixed,
        )
        assert (t_out == t_fixed).all()
        non_pad = batch_ids.ne(tiny_cfg.pad_token_id)
        assert mask[non_pad].all(), "t=1 should mask all non-pad tokens"


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestDiffusionLoss:
    def _make_batch(self, tiny_cfg):
        torch.manual_seed(7)
        x0 = torch.randint(5, tiny_cfg.vocab_size, (2, 16))
        _, mask, t = forward_process(x0, tiny_cfg.mask_token_id,
                                     tiny_cfg.pad_token_id)
        return x0, mask, t

    def test_loss_is_scalar(self, tiny_model, tiny_cfg):
        x0, mask, t = self._make_batch(tiny_cfg)
        logits = tiny_model(x0)
        loss = diffusion_loss(logits, x0, mask, t)
        assert loss.shape == ()

    def test_loss_is_positive(self, tiny_model, tiny_cfg):
        x0, mask, t = self._make_batch(tiny_cfg)
        logits = tiny_model(x0)
        assert diffusion_loss(logits, x0, mask, t).item() > 0

    def test_loss_finite(self, tiny_model, tiny_cfg):
        x0, mask, t = self._make_batch(tiny_cfg)
        logits = tiny_model(x0)
        assert torch.isfinite(diffusion_loss(logits, x0, mask, t))

    def test_weighted_exceeds_simple_at_high_t(self, tiny_model, tiny_cfg):
        # (1/t) weight at high t makes weighted > simple
        x0 = torch.randint(5, tiny_cfg.vocab_size, (2, 16))
        t_high = torch.full((2,), 0.95)
        xt, mask, _ = forward_process(
            x0, tiny_cfg.mask_token_id, tiny_cfg.pad_token_id, t_tensor=t_high
        )
        logits = tiny_model(xt)
        weighted = diffusion_loss(logits, x0, mask, t_high)
        simple   = simple_loss(logits, x0, mask)
        assert weighted.item() > simple.item()

    def test_perfect_predictions_give_low_loss(self, tiny_cfg):
        # Feed logits that put all probability on the correct token → loss ≈ 0
        x0 = torch.randint(5, tiny_cfg.vocab_size, (2, 8))
        t  = torch.full((2,), 0.5)
        mask = torch.ones_like(x0, dtype=torch.bool)

        # Build logits: large value at correct class, -inf elsewhere
        logits = torch.full((2, 8, tiny_cfg.vocab_size), -1e9)
        logits.scatter_(2, x0.unsqueeze(-1), 100.0)

        loss = diffusion_loss(logits, x0, mask, t)
        assert loss.item() < 0.01

    def test_reduction_none(self, tiny_model, tiny_cfg):
        x0, mask, t = self._make_batch(tiny_cfg)
        logits = tiny_model(x0)
        per_sample = diffusion_loss(logits, x0, mask, t, reduction="none")
        assert per_sample.shape == (x0.shape[0],)

    def test_empty_mask_simple_loss_zero(self, tiny_cfg):
        x0     = torch.randint(5, tiny_cfg.vocab_size, (2, 8))
        logits = torch.randn(2, 8, tiny_cfg.vocab_size)
        mask   = torch.zeros(2, 8, dtype=torch.bool)   # nothing masked
        loss   = simple_loss(logits, x0, mask)
        # clamp_min(1) prevents NaN; result is effectively 0/1 = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Unmasking schedule
# ---------------------------------------------------------------------------

class TestUnmaskSchedule:
    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_schedule_starts_at_zero(self, schedule):
        assert _unmask_fraction(0, 10, schedule) == pytest.approx(0.0)

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_schedule_ends_at_one(self, schedule):
        assert _unmask_fraction(10, 10, schedule) == pytest.approx(1.0)

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_schedule_is_monotone(self, schedule):
        fracs = [_unmask_fraction(s, 10, schedule) for s in range(11)]
        assert all(fracs[i] <= fracs[i + 1] for i in range(len(fracs) - 1))

    def test_cosine_slower_start_than_linear(self):
        # Cosine schedule: slow start, faster finish
        linear = _unmask_fraction(2, 10, "linear")
        cosine = _unmask_fraction(2, 10, "cosine")
        assert cosine < linear


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class TestSampler:
    def test_output_shape(self, tiny_model, tiny_cfg):
        context = torch.full((1, 16), tiny_cfg.mask_token_id, dtype=torch.long)
        out = sample(tiny_model, context, tiny_cfg.mask_token_id,
                     tiny_cfg.pad_token_id, num_steps=4)
        assert out.shape == context.shape

    def test_no_mask_tokens_remain(self, tiny_model, tiny_cfg):
        context = torch.full((2, 12), tiny_cfg.mask_token_id, dtype=torch.long)
        out = sample(tiny_model, context, tiny_cfg.mask_token_id,
                     tiny_cfg.pad_token_id, num_steps=4)
        assert not (out == tiny_cfg.mask_token_id).any()

    def test_tokens_within_vocab(self, tiny_model, tiny_cfg):
        context = torch.full((2, 12), tiny_cfg.mask_token_id, dtype=torch.long)
        out = sample(tiny_model, context, tiny_cfg.mask_token_id,
                     tiny_cfg.pad_token_id, num_steps=4)
        assert (out >= 0).all() and (out < tiny_cfg.vocab_size).all()

    def test_seed_reproducible(self, tiny_model, tiny_cfg):
        context = torch.full((1, 16), tiny_cfg.mask_token_id, dtype=torch.long)
        out_a = sample(tiny_model, context, tiny_cfg.mask_token_id,
                       tiny_cfg.pad_token_id, num_steps=4, seed=99)
        out_b = sample(tiny_model, context, tiny_cfg.mask_token_id,
                       tiny_cfg.pad_token_id, num_steps=4, seed=99)
        assert (out_a == out_b).all()

    def test_greedy_temperature_zero(self, tiny_model, tiny_cfg):
        context = torch.full((1, 8), tiny_cfg.mask_token_id, dtype=torch.long)
        out = sample(tiny_model, context, tiny_cfg.mask_token_id,
                     tiny_cfg.pad_token_id, num_steps=4, temperature=0.0)
        assert out.shape == context.shape
        assert not (out == tiny_cfg.mask_token_id).any()

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_schedule_variants_complete(self, tiny_model, tiny_cfg, schedule):
        context = torch.full((1, 12), tiny_cfg.mask_token_id, dtype=torch.long)
        out = sample(tiny_model, context, tiny_cfg.mask_token_id,
                     tiny_cfg.pad_token_id, num_steps=4, schedule=schedule)
        assert not (out == tiny_cfg.mask_token_id).any()

    def test_infill_fixed_positions_preserved(self, tiny_model, tiny_cfg):
        B, L = 1, 16
        context = torch.randint(5, tiny_cfg.vocab_size, (B, L))
        context[0, 6:10] = tiny_cfg.mask_token_id    # only fill positions 6-9
        fixed = ~context.eq(tiny_cfg.mask_token_id)

        out = infill(tiny_model, context, tiny_cfg.mask_token_id,
                     pad_token_id=tiny_cfg.pad_token_id, num_steps=4)

        # Fixed positions must be identical to input
        assert (out[fixed] == context[fixed]).all()
        # Masked positions must be filled (no mask tokens remain)
        assert not (out == tiny_cfg.mask_token_id).any()

    def test_partial_context_preserves_known(self, tiny_model, tiny_cfg):
        # Set the first half to known values, mask the second half
        known_tokens = torch.randint(5, tiny_cfg.vocab_size, (1, 8))
        context = torch.cat(
            [known_tokens,
             torch.full((1, 8), tiny_cfg.mask_token_id)], dim=1
        )
        fixed = torch.cat(
            [torch.ones(1, 8, dtype=torch.bool),
             torch.zeros(1, 8, dtype=torch.bool)], dim=1
        )
        out = sample(tiny_model, context, tiny_cfg.mask_token_id,
                     tiny_cfg.pad_token_id, num_steps=4, fixed_positions=fixed)
        assert (out[0, :8] == known_tokens[0]).all()
