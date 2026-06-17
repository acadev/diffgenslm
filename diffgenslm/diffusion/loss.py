"""
Diffusion training loss (LLaDA-style ELBO).

Loss = E_t [ (1/t) * CE(model(x_t), x_0)  at masked positions ]

The 1/t weighting is the continuous-time ELBO for absorbing-state diffusion.
It up-weights high-t (heavily masked) examples relative to low-t examples,
which corrects for the fact that the model sees fewer context tokens at high t.

Reference: LLaDA (Large Language Diffusion with mAsking), 2024.
"""

import torch
import torch.nn.functional as F


def diffusion_loss(
    logits: torch.Tensor,     # [B, L, V]  model output
    x0: torch.Tensor,         # [B, L]     original clean tokens
    mask: torch.Tensor,       # [B, L]     bool — True at diffused positions
    t: torch.Tensor,          # [B]        diffusion time / mask probability
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the weighted cross-entropy loss over masked positions.

    Args:
        logits:    Model output logits [B, L, V].
        x0:        Original (clean) token ids [B, L].
        mask:      Boolean mask [B, L] — True = position was masked.
        t:         Per-sample mask probability [B].
        reduction: "mean" (default) or "sum" or "none".

    Returns:
        Scalar loss (or per-sample tensor when reduction="none").
    """
    B, L, V = logits.shape

    # Cross-entropy per token at masked positions
    logits_flat = logits.view(B * L, V)
    targets_flat = x0.view(B * L)
    ce_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*L]
    ce = ce_flat.view(B, L)                                                  # [B, L]

    # Keep only masked positions
    ce = ce * mask.float()                                                   # [B, L]

    # Sum over sequence, normalize by number of masked tokens per sample
    num_masked = mask.float().sum(dim=1).clamp_min(1.0)                    # [B]
    ce_per_sample = ce.sum(dim=1) / num_masked                              # [B]

    # ELBO weighting: 1/t per sample
    weight = 1.0 / t.clamp_min(1e-4)                                        # [B]
    weighted = ce_per_sample * weight                                        # [B]

    if reduction == "mean":
        return weighted.mean()
    elif reduction == "sum":
        return weighted.sum()
    else:
        return weighted


def simple_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Unweighted cross-entropy at masked positions (useful for validation).
    """
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.view(B * L, V), x0.view(B * L), reduction="none").view(B, L)
    ce = ce * mask.float()
    num_masked = mask.float().sum(dim=1).clamp_min(1.0)
    return (ce.sum(dim=1) / num_masked).mean()
