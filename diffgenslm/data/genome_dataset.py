"""
HDF5-backed dataset for diffusion model training.

Reads the contiguous token stream produced by build_hdf5.py and returns
fixed-length sliding windows suitable for transformer training.

Each item: token_ids LongTensor [seq_len]

Compatible with both:
  - torch.utils.data.DataLoader (single/multi-GPU, torchrun)
  - Manual rank-sharding used in HiSAN's DDP loop
"""

import glob
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class GenomeDiffusionDataset(Dataset):
    """
    Memory-mapped HDF5 dataset.

    Loads the full token stream for assigned HDF5 files into a memory-mapped
    numpy array, then serves fixed-length windows with random offsets.

    Args:
        hdf5_files:     List of HDF5 paths to load.
        seq_len:        Context window length (tokens).
        stride:         Step between windows (< seq_len gives overlap).
                        Set to seq_len for non-overlapping windows.
        pad_token_id:   ID used to pad the final incomplete window.
        rc_augment:     If True, randomly reverse-complement token sequences.
                        NOTE: only meaningful when tokenizer is nucleotide-level;
                        currently disabled (set False for composite tokenizer).
        rank:           DDP rank for pre-sharding.
        world_size:     DDP world size for pre-sharding.
        seed:           Random seed for shuffling.
    """

    def __init__(
        self,
        hdf5_files: List[Union[str, Path]],
        seq_len: int = 4096,
        stride: Optional[int] = None,
        pad_token_id: int = 0,
        rc_augment: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.pad_token_id = pad_token_id
        self.rc_augment = rc_augment

        # Shard files across DDP ranks
        all_files = sorted(str(f) for f in hdf5_files)
        my_files = all_files[rank::world_size]

        # Load all token streams from assigned files
        self._windows: List[np.ndarray] = []
        total_tokens = 0

        for fpath in my_files:
            try:
                import h5py
                with h5py.File(fpath, "r") as hf:
                    ids = hf["input_ids"][:]           # int32 array
                    starts = hf["sequence_starts"][:]  # int64 array
                total_tokens += len(ids)
                self._extract_windows(ids, starts)
            except Exception as e:
                print(f"[WARN] Failed to load {fpath}: {e}")

        # Shuffle windows
        rng = random.Random(seed + rank)
        rng.shuffle(self._windows)

        print(
            f"[GenomeDiffusionDataset] rank={rank}: "
            f"{len(my_files)} files, {total_tokens:,} tokens, "
            f"{len(self._windows):,} windows (seq_len={seq_len})"
        )

    def _extract_windows(self, ids: np.ndarray, starts: np.ndarray):
        """Slice per-sequence windows with the configured stride."""
        seq_starts = list(starts) + [len(ids)]
        for i in range(len(seq_starts) - 1):
            s = int(seq_starts[i])
            e = int(seq_starts[i + 1])
            seq = ids[s:e]
            # Slide window across sequence
            for offset in range(0, max(1, len(seq) - self.seq_len + 1), self.stride):
                chunk = seq[offset : offset + self.seq_len]
                if len(chunk) < self.seq_len:
                    # Pad last window
                    pad = np.full(self.seq_len - len(chunk), self.pad_token_id, dtype=np.int32)
                    chunk = np.concatenate([chunk, pad])
                self._windows.append(chunk.astype(np.int32))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tokens = torch.from_numpy(self._windows[idx]).long()
        return tokens


def build_dataloader(
    hdf5_dir: Union[str, Path],
    split: str,
    seq_len: int,
    batch_size: int,
    stride: Optional[int] = None,
    pad_token_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for a given split from the HDF5 output directory.

    Supports both single-file outputs (train.h5) and rank-sharded outputs
    (train_rank0000.h5, train_rank0001.h5, ...).
    """
    hdf5_dir = Path(hdf5_dir)

    # Collect files for this split
    files = sorted(hdf5_dir.glob(f"{split}_rank*.h5"))
    if not files:
        # Single-file output
        single = hdf5_dir / f"{split}.h5"
        if single.exists():
            files = [single]
        else:
            raise FileNotFoundError(f"No HDF5 files found for split '{split}' in {hdf5_dir}")

    dataset = GenomeDiffusionDataset(
        hdf5_files=files,
        seq_len=seq_len,
        stride=stride,
        pad_token_id=pad_token_id,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
