"""
DiffGenSLM training loop.

Supports both:
  - Polaris (NVIDIA A100):  torchrun or mpiexec + NCCL  (--backend nccl)
  - Aurora  (Intel GPU):    mpiexec + oneCCL             (--backend ccl)
                            Full DeepSpeed integration is Phase 2;
                            this script lays the ground work with the same
                            DDP skeleton and exposes --use_deepspeed for
                            opt-in ZeRO once ds config is ready.

Architecture closely mirrors HiSAN's train.py:
  - setup_distributed() handles torchrun + mpiexec + single-GPU fallback
  - DDP wrapping, AMP (fp16/bf16), gradient clipping, checkpoint/resume
  - W&B logging (optional, skip with --no_wandb)

Usage (single node, torchrun):
    torchrun --nproc_per_node=4 -m diffgenslm.train \\
        --hdf5_dir /path/to/hdf5 \\
        --config   /path/to/configs/small.yaml \\
        --save_dir /path/to/checkpoints

Usage (Polaris, mpiexec):
    mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE \\
        python -m diffgenslm.train [same args]
"""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data.genome_dataset import build_dataloader
from .diffusion.loss import diffusion_loss, simple_loss
from .diffusion.process import forward_process
from .diffusion.sample import sample as generate_sample
from .models.diffgenome import DiffGenomeConfig, DiffGenomeModel


# ---------------------------------------------------------------------------
# Distributed setup (mirrors HiSAN train.py)
# ---------------------------------------------------------------------------

def setup_distributed(backend: str = "nccl"):
    """Return (rank, world_size, device). Works with torchrun + mpiexec."""
    if dist.is_initialized():
        rank = dist.get_rank()
        ws = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        return rank, ws, device

    # mpiexec (Polaris PBS / Aurora PBS)
    if "PMI_RANK" in os.environ or "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
        ws   = int(os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
        local_rank = int(os.environ.get("PMI_LOCAL_RANK",
                                        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))

        if "MASTER_ADDR" not in os.environ:
            if "PBS_NODEFILE" in os.environ:
                with open(os.environ["PBS_NODEFILE"]) as f:
                    nodes = [line.strip() for line in f if line.strip()]
                os.environ["MASTER_ADDR"] = nodes[0] if nodes else "localhost"
            else:
                os.environ["MASTER_ADDR"] = "localhost"
        os.environ.setdefault("MASTER_PORT", "29500")

        dist.init_process_group(backend=backend, init_method="env://",
                                world_size=ws, rank=rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, ws, device

    # torchrun / torch.distributed.launch
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        ws   = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend=backend)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, ws, device

    # Single GPU / MPS / CPU fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return 0, 1, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model_config(cfg: dict) -> DiffGenomeConfig:
    m = cfg.get("model", {})
    return DiffGenomeConfig(
        vocab_size=m.get("vocab_size", 8280),
        hidden_size=m.get("hidden_size", 512),
        num_layers=m.get("num_layers", 8),
        num_heads=m.get("num_heads", 8),
        num_kv_heads=m.get("num_kv_heads", 4),
        ffn_intermediate_size=m.get("ffn_intermediate_size", 1366),
        rope_theta=m.get("rope_theta", 10_000.0),
        max_seq_len=m.get("max_seq_len", 4096),
        dropout=m.get("dropout", 0.1),
        pad_token_id=m.get("pad_token_id", 0),
        mask_token_id=m.get("mask_token_id", 4),
    )


# ---------------------------------------------------------------------------
# Metric / logging helpers
# ---------------------------------------------------------------------------

def _has_val_split(hdf5_dir: str) -> bool:
    """True if any val HDF5 file exists (single-file or rank-sharded)."""
    d = Path(hdf5_dir)
    return (d / "val.h5").exists() or bool(list(d.glob("val_rank*.h5")))


def _token_accuracy(
    logits: torch.Tensor,   # [B, L, V]
    x0:     torch.Tensor,   # [B, L]
    mask:   torch.Tensor,   # [B, L] bool — True = was masked
) -> float:
    """Fraction of masked tokens whose top-1 prediction equals the original."""
    with torch.no_grad():
        preds = logits.detach().argmax(-1)           # [B, L]
        correct = (preds == x0) & mask
        n_masked = mask.sum().float().clamp(min=1)
        return (correct.sum().float() / n_masked).item()


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5


def _gpu_mem_gb(device: torch.device):
    """Return (allocated_gb, reserved_gb) or (0, 0) for CPU."""
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated(device) / 1e9,
        torch.cuda.memory_reserved(device) / 1e9,
    )


def _build_sample_table(
    model_raw: DiffGenomeModel,
    mask_token_id: int,
    pad_token_id: int,
    device: torch.device,
    n_samples: int = 3,
    seq_len: int = 256,
):
    """Generate n_samples short sequences and return a wandb.Table."""
    import wandb  # imported here so the function stays importable without wandb

    rows = []
    model_raw.eval()
    try:
        with torch.no_grad():
            context = torch.full((1, seq_len), mask_token_id,
                                 dtype=torch.long, device=device)
            for i in range(n_samples):
                out = generate_sample(
                    model_raw, context, mask_token_id, pad_token_id,
                    num_steps=32, temperature=1.0, seed=1000 + i,
                )
                token_ids = out[0].tolist()
                preview = " ".join(str(t) for t in token_ids[:80])
                n_unique = len(set(token_ids))
                rows.append([i, preview, seq_len, n_unique])
    except Exception as exc:
        rows.append([-1, f"[error: {exc}]", 0, 0])
    model_raw.train()
    return wandb.Table(
        columns=["sample_idx", "token_ids_preview", "seq_len", "n_unique_tokens"],
        data=rows,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ── Distributed init ──────────────────────────────────────────────────
    rank, world_size, device = setup_distributed(args.backend)
    is_main = rank == 0
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    if is_main:
        print(f"[DDP] world_size={world_size}  device={device}  dtype={dtype}")

    cfg          = load_config(args.config)
    training_cfg = cfg.get("training", {})
    model_cfg    = build_model_config(cfg)

    is_cuda = device.type == "cuda"

    # ── Model ─────────────────────────────────────────────────────────────
    model = DiffGenomeModel(model_cfg).to(device)

    if is_main:
        n_params = model.num_params()
        print(f"[MODEL] DiffGenomeModel: {n_params / 1e6:.1f}M params")
        print(f"        vocab={model_cfg.vocab_size}  hidden={model_cfg.hidden_size}"
              f"  layers={model_cfg.num_layers}  heads={model_cfg.num_heads}"
              f"  kv_heads={model_cfg.num_kv_heads}  ffn={model_cfg.ffn_intermediate_size}")

    model_raw = model  # keep reference before DDP wrapping for checkpointing/sampling

    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    broadcast_buffers=False, find_unused_parameters=False)

    # ── Data ──────────────────────────────────────────────────────────────
    seq_len     = training_cfg.get("seq_len", model_cfg.max_seq_len)
    batch_size  = training_cfg.get("batch_size", 8)
    num_workers = training_cfg.get("num_workers", 4)

    train_loader = build_dataloader(
        args.hdf5_dir, "train", seq_len, batch_size,
        pad_token_id=model_cfg.pad_token_id,
        rank=rank, world_size=world_size, num_workers=num_workers,
        seed=training_cfg.get("seed", 42),
    )
    val_loader = build_dataloader(
        args.hdf5_dir, "val", seq_len, batch_size,
        pad_token_id=model_cfg.pad_token_id,
        rank=rank, world_size=world_size, num_workers=num_workers,
        shuffle=False, seed=training_cfg.get("seed", 42) + 1,
    ) if _has_val_split(args.hdf5_dir) else None

    # Sync batch counts across ranks to avoid NCCL deadlock on uneven shards
    if world_size > 1:
        tc = torch.tensor([len(train_loader)], device=device, dtype=torch.long)
        dist.all_reduce(tc, op=dist.ReduceOp.MIN)
        max_train_steps = int(tc.item())
    else:
        max_train_steps = len(train_loader)

    # Optional per-epoch step cap (useful for smoke tests on large datasets)
    _epoch_step_cap = training_cfg.get("max_steps_per_epoch", None)
    if _epoch_step_cap is not None:
        max_train_steps = min(max_train_steps, int(_epoch_step_cap))

    if is_main:
        print(f"[DATA] train batches/rank={len(train_loader)} (sync={max_train_steps})"
              + (f"  val={len(val_loader)}" if val_loader else "  val=none"))

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    lr = training_cfg.get("lr", 3e-4)
    wd = training_cfg.get("weight_decay", 0.01)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))

    epochs    = training_cfg.get("epochs", 20)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * max_train_steps, eta_min=lr / 10)
    scaler    = torch.amp.GradScaler("cuda", enabled=(args.fp16 and not args.bf16 and is_cuda))

    # ── Checkpoint & resume ───────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    ckpt_path = save_dir / "checkpoint.pt"

    best_val    = float("inf")
    start_epoch = 1
    global_step = 0

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model_raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        global_step = ckpt.get("global_step", (start_epoch - 1) * max_train_steps)
        if is_main:
            print(f"[RESUME] epoch={start_epoch}  step={global_step}  best_val={best_val:.4f}")

    # ── W&B init ─────────────────────────────────────────────────────────
    use_wandb  = False
    wandb_run  = None
    if is_main and not args.no_wandb:
        try:
            import wandb as _wandb
            wandb_run = _wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"diffgenslm_{Path(args.config).stem}",
                config={
                    **cfg,
                    "world_size":    world_size,
                    "device":        str(device),
                    "dtype":         str(dtype),
                    "num_params_M":  model_raw.num_params() / 1e6,
                    "max_train_steps_per_epoch": max_train_steps,
                },
                resume="allow",
            )
            # Gradient and parameter histograms — sampled every log_freq steps
            _wandb.watch(model_raw, log="gradients",
                         log_freq=args.wandb_log_freq, log_graph=False)
            use_wandb = True
            print(f"[WANDB] Run: {wandb_run.url}")
        except Exception as exc:
            print(f"[WANDB] Skipped: {exc}")

    mask_token_id = model_cfg.mask_token_id
    pad_token_id  = model_cfg.pad_token_id
    grad_clip     = training_cfg.get("grad_clip", 1.0)
    sample_every  = args.sample_every   # generate sample sequences every N epochs

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()

        # Accumulators for epoch-level aggregates
        epoch_loss   = epoch_acc    = 0.0
        epoch_t_vals: List[float]  = []   # diffusion t samples — for histogram
        n_steps      = 0

        pbar = tqdm(train_loader, total=max_train_steps,
                    desc=f"Epoch {epoch}/{epochs}", disable=not is_main)

        for step, x0 in enumerate(pbar):
            if step >= max_train_steps:
                break

            x0 = x0.to(device, non_blocking=True)
            batch_t0 = time.perf_counter()

            # Forward diffusion: sample t, mask tokens
            xt, mask, t_sample = forward_process(x0, mask_token_id, pad_token_id)
            t_mean = t_sample.mean().item()
            epoch_t_vals.extend(t_sample.cpu().tolist())

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(is_cuda and (args.fp16 or args.bf16))):
                logits = model(xt)
                loss   = diffusion_loss(logits, x0, mask, t_sample)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)

            gn = _grad_norm(model)   # compute after unscale, before clip

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_val = loss.item()
            acc_val  = _token_accuracy(logits, x0, mask)
            cur_lr   = optimizer.param_groups[0]["lr"]

            epoch_loss += loss_val
            epoch_acc  += acc_val
            n_steps    += 1
            global_step += 1

            steps_per_sec = 1.0 / (time.perf_counter() - batch_t0)
            seqs_per_sec  = steps_per_sec * x0.size(0)

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                acc=f"{acc_val:.3f}",
                t=f"{t_mean:.2f}",
                gnorm=f"{gn:.2f}",
            )

            # Per-step W&B logging (controlled by --wandb_log_freq)
            if use_wandb and (global_step % args.wandb_log_freq == 0):
                import wandb as _wandb
                _wandb.log({
                    "train/loss":      loss_val,
                    "train/token_acc": acc_val,
                    "train/t_mean":    t_mean,
                    "train/grad_norm": gn,
                    "train/lr":        cur_lr,
                    "sys/seqs_per_sec": seqs_per_sec,
                }, step=global_step)

        train_loss = epoch_loss / max(1, n_steps)
        train_acc  = epoch_acc  / max(1, n_steps)
        epoch_sec  = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────
        val_loss = val_acc = None
        max_val_steps = training_cfg.get("max_val_steps", None)
        if val_loader is not None:
            model.eval()
            vsum_loss = vsum_acc = vn = 0.0
            with torch.no_grad():
                for vstep, x0 in enumerate(val_loader):
                    if max_val_steps is not None and vstep >= max_val_steps:
                        break
                    x0   = x0.to(device, non_blocking=True)
                    xt, mask, _ = forward_process(x0, mask_token_id, pad_token_id)
                    with torch.amp.autocast("cuda", dtype=dtype,
                                           enabled=(is_cuda and (args.fp16 or args.bf16))):
                        logits = model(xt)
                        vl     = simple_loss(logits, x0, mask)
                    vsum_loss += vl.item()
                    vsum_acc  += _token_accuracy(logits, x0, mask)
                    vn        += 1.0
            val_loss = vsum_loss / max(1.0, vn)
            val_acc  = vsum_acc  / max(1.0, vn)

        if world_size > 1:
            dist.barrier()

        # ── Checkpoint ────────────────────────────────────────────────────
        if is_main:
            state = {
                "epoch":        epoch,
                "global_step":  global_step,
                "model":        model_raw.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scaler":       scaler.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_val":     best_val,
                "model_config": model_cfg.__dict__,
            }
            torch.save(state, ckpt_path)

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                torch.save(state, best_path)

            # ── Console output ─────────────────────────────────────────
            line = (f"[E{epoch:03d}] train={train_loss:.4f}  acc={train_acc:.3f}"
                    + (f"  val={val_loss:.4f}  val_acc={val_acc:.3f}" if val_loss is not None else "")
                    + f"  best={best_val:.4f}  {epoch_sec:.0f}s")
            print(line)

            # ── W&B epoch-level logging ────────────────────────────────
            if use_wandb and wandb_run:
                import wandb as _wandb

                mem_alloc, mem_reserved = _gpu_mem_gb(device)
                epoch_log = {
                    "epoch":                epoch,
                    "train/loss_epoch":     train_loss,
                    "train/token_acc_epoch": train_acc,
                    "train/lr":             optimizer.param_groups[0]["lr"],
                    "sys/epoch_sec":        epoch_sec,
                    "sys/gpu_mem_alloc_gb": mem_alloc,
                    "sys/gpu_mem_reserved_gb": mem_reserved,
                    "train/t_histogram":    _wandb.Histogram(epoch_t_vals),
                }
                if val_loss is not None:
                    epoch_log["val/loss_epoch"]      = val_loss
                    epoch_log["val/token_acc_epoch"] = val_acc
                if best_val < float("inf"):
                    epoch_log["val/best_loss"] = best_val

                # Periodic sample generation
                if sample_every > 0 and epoch % sample_every == 0:
                    sample_seq_len = min(256, model_cfg.max_seq_len)
                    epoch_log["samples/generated"] = _build_sample_table(
                        model_raw, mask_token_id, pad_token_id, device,
                        n_samples=3, seq_len=sample_seq_len,
                    )

                _wandb.log(epoch_log, step=global_step)

    if is_main and use_wandb:
        import wandb as _wandb
        _wandb.finish()

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DiffGenSLM")
    p.add_argument("--config",    type=str, required=True, help="Path to YAML config")
    p.add_argument("--hdf5_dir",  type=str, required=True, help="Directory with train/val HDF5")
    p.add_argument("--save_dir",  type=str, default="checkpoints/")
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--fp16",      action="store_true")
    p.add_argument("--bf16",      action="store_true", help="bfloat16 (preferred on A100/H100)")
    p.add_argument("--backend",   default="nccl", choices=["nccl", "ccl", "gloo"],
                   help="DDP backend: nccl (NVIDIA), ccl (Intel/Aurora), gloo (CPU)")
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--wandb_project",    default="diffgenslm")
    p.add_argument("--wandb_run_name",   default=None)
    p.add_argument("--wandb_log_freq",   type=int, default=50,
                   help="Log per-step metrics every N steps (default 50)")
    p.add_argument("--sample_every",     type=int, default=5,
                   help="Generate sample sequences every N epochs for W&B (0=off)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
