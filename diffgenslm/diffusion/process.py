"""
Absorbing-state discrete diffusion forward process (LLaDA / D3PM style).

Forward process:
    x_t = Mask(x_0, t)
    where each token in x_0 is independently replaced with <mask>
    with probability t ~ Uniform(0, 1).

Reverse process (sampling) lives in sample.py.
"""

import torch


def forward_process(
    x0: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int = 0,
    t: float = None,
    t_tensor: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply absorbing-state diffusion: randomly mask tokens with probability t.

    Args:
        x0:            Clean token ids [B, L].
        mask_token_id: ID of the <mask> absorbing state.
        pad_token_id:  Padding positions are never masked.
        t:             Scalar mask probability in (0, 1].
                       If None, t_tensor must be provided.
        t_tensor:      Per-sample mask probabilities [B] in (0, 1].
                       If both t and t_tensor are None, t is sampled uniformly.

    Returns:
        xt:         Masked token ids [B, L].
        mask:       Bool tensor [B, L] — True where a token was masked.
        t_tensor:   The mask probabilities used, [B].
    """
    B, L = x0.shape
    device = x0.device

    if t_tensor is None:
        if t is not None:
            t_tensor = torch.full((B,), t, device=device)
        else:
            # Sample t ~ Uniform(0, 1) per sample
            t_tensor = torch.rand(B, device=device)

    # Expand t to [B, L] for per-token Bernoulli draw
    t_expanded = t_tensor.unsqueeze(1).expand(B, L)

    # Draw Bernoulli mask: True = mask this token
    rand = torch.rand_like(t_expanded)
    is_masked = rand < t_expanded

    # Never mask padding positions
    is_pad = x0.eq(pad_token_id)
    is_masked = is_masked & ~is_pad

    xt = x0.clone()
    xt[is_masked] = mask_token_id

    return xt, is_masked, t_tensor
