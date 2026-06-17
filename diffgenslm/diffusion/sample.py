"""
Iterative unmasking sampler for discrete absorbing-state diffusion.

At each denoising step we:
  1. Run the model on the current (partially masked) sequence.
  2. Sample candidate tokens from the model's distribution at each masked position.
  3. Decide which fraction of masked positions to unmask this step.
  4. Commit the highest-confidence predictions; leave the rest masked.

Supports two unmasking schedules:
  - "linear":   unmask a constant fraction each step
  - "cosine":   unmask following a cosine schedule (tends to work slightly better)

Also supports conditional infilling: keep a subset of positions fixed.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def _unmask_fraction(step: int, total_steps: int, schedule: str = "linear") -> float:
    """Fraction of remaining masks to remove at this step."""
    progress = step / total_steps
    if schedule == "cosine":
        import math
        return 1.0 - math.cos(progress * math.pi / 2)
    return progress  # linear


@torch.no_grad()
def sample(
    model,
    input_ids: torch.Tensor,           # [B, L] — all-mask starting point or partial
    mask_token_id: int,
    pad_token_id: int = 0,
    num_steps: int = 64,
    temperature: float = 1.0,
    schedule: str = "linear",
    fixed_positions: Optional[torch.Tensor] = None,  # [B, L] bool — do not change
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Iteratively unmask a sequence of tokens.

    Args:
        model:            DiffGenomeModel (or any bidirectional LM with .forward returning logits).
        input_ids:        Starting sequence [B, L]. Positions to generate should be mask_token_id.
        mask_token_id:    ID of the absorbing <mask> state.
        pad_token_id:     Padding token — never replaced.
        num_steps:        Number of denoising steps (more = better quality, slower).
        temperature:      Sampling temperature (1.0 = multinomial, →0 = greedy).
        schedule:         Unmasking schedule: "linear" or "cosine".
        fixed_positions:  Bool mask [B, L]. True = keep this position's token as-is.
        top_p:            Nucleus sampling threshold (1.0 = disabled).
        seed:             Optional RNG seed for reproducibility.

    Returns:
        Completed token ids [B, L].
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = input_ids.device
    B, L = input_ids.shape
    x = input_ids.clone()

    if fixed_positions is None:
        fixed_positions = torch.zeros(B, L, dtype=torch.bool, device=device)

    # Track which positions are still masked
    is_masked = x.eq(mask_token_id) & ~fixed_positions & ~x.eq(pad_token_id)
    total_masked = is_masked.long().sum(dim=1).float()  # [B]

    for step in range(num_steps):
        if not is_masked.any():
            break

        # Forward pass
        logits = model(x)                                            # [B, L, V]

        # A denoising model must never predict the absorbing state (mask) or
        # padding as a clean token — zero out those logit entries.
        logits[:, :, mask_token_id] = float("-inf")
        logits[:, :, pad_token_id]  = float("-inf")

        # Sample or greedy-pick tokens at masked positions
        if temperature > 0:
            probs = _apply_temperature_and_top_p(logits, temperature, top_p)
            predicted = torch.multinomial(
                probs.view(B * L, -1), num_samples=1
            ).squeeze(-1).view(B, L)
        else:
            predicted = logits.argmax(dim=-1)

        # Compute model confidence (max prob) at each masked position
        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)                # [B, L, V]
            confidence = log_probs.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)  # [B, L]
            confidence = confidence.masked_fill(~is_masked, float("-inf"))

        # Determine how many to unmask this step (cumulative schedule)
        frac_now = _unmask_fraction(step + 1, num_steps, schedule)
        frac_prev = _unmask_fraction(step, num_steps, schedule)
        n_unmask = ((frac_now - frac_prev) * total_masked).long().clamp_min(1)  # [B]

        # Unmask the top-n confident positions
        for b in range(B):
            if n_unmask[b] <= 0 or not is_masked[b].any():
                continue
            top_k = min(int(n_unmask[b].item()), int(is_masked[b].sum().item()))
            conf_b = confidence[b]
            _, top_idxs = torch.topk(conf_b, k=top_k)
            x[b, top_idxs] = predicted[b, top_idxs]
            is_masked[b, top_idxs] = False

    # Final pass: replace any remaining masks — again excluding special tokens
    if is_masked.any():
        logits = model(x)
        logits[:, :, mask_token_id] = float("-inf")
        logits[:, :, pad_token_id]  = float("-inf")
        final_pred = logits.argmax(dim=-1)
        x[is_masked] = final_pred[is_masked]

    return x


def _apply_temperature_and_top_p(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Apply temperature scaling and optional nucleus filtering; return probabilities."""
    B, L, V = logits.shape
    logits = logits / max(temperature, 1e-6)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative prob above threshold
        remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    return logits.softmax(dim=-1)


@torch.no_grad()
def infill(
    model,
    context: torch.Tensor,             # [B, L] with mask_token_id at positions to fill
    mask_token_id: int,
    **kwargs,
) -> torch.Tensor:
    """
    Convenience wrapper for conditional infilling.

    Positions that are NOT mask_token_id are treated as fixed context.
    """
    fixed = ~context.eq(mask_token_id)
    return sample(model, context, mask_token_id, fixed_positions=fixed, **kwargs)
