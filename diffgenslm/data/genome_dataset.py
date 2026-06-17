"""
HDF5-backed dataset for diffusion model training.

Reads the contiguous token stream produced by build_hdf5.py and returns
fixed-length sliding windows suitable for transformer training.

Each item: (input_ids, strand_ids) — both LongTensor [seq_len].

strand_ids encodes biological context per token:
  0 = pad / BOS / EOS
  1 = forward strand (+)
  2 = reverse strand (-)
  3 = intergenic / non-coding

strand_ids are loaded from the HDF5 ``strand_ids`` dataset when present.
Old HDF5 files without that dataset return a zero-filled tensor (safe
fallback: the same-strand bias in BidirectionalGQA is simply skipped when
all strand IDs are 0).

Compatible with both:
  - torch.utils.data.DataLoader (single/multi-GPU, torchrun)
  - Manual rank-sharding used in HiSAN's DDP loop
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class GenomeDiffusionDataset(Dataset):
    """
    Memory-mapped HDF5 dataset.

    Loads the full token stream (and optionally strand IDs) for assigned HDF5
    files, then serves fixed-length windows with sliding stride.

    Args:
        hdf5_files:     List of HDF5 paths to load.
        seq_len:        Context window length (tokens).
        stride:         Step between windows (< seq_len gives overlap).
                        Defaults to seq_len (non-overlapping).
        pad_token_id:   ID used to pad the final incomplete window.
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
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.seq_len      = seq_len
        self.stride       = stride or seq_len
        self.pad_token_id = pad_token_id

        all_files = sorted(str(f) for f in hdf5_files)
        my_files  = all_files[rank::world_size]

        # Parallel lists: _windows[i] = (input_ids_chunk, strand_ids_chunk)
        self._windows: List[Tuple[np.ndarray, np.ndarray]] = []
        total_tokens = 0

        for fpath in my_files:
            try:
                import h5py
                with h5py.File(fpath, "r") as hf:
                    ids    = hf["input_ids"][:]           # int32
                    starts = hf["sequence_starts"][:]     # int64
                    # strand_ids: present in files built with the bio-features pipeline
                    if "strand_ids" in hf:
                        strands = hf["strand_ids"][:]     # int8
                    else:
                        strands = None
                total_tokens += len(ids)
                self._extract_windows(ids, starts, strands)
            except Exception as e:
                print(f"[WARN] Failed to load {fpath}: {e}")

        rng = random.Random(seed + rank)
        rng.shuffle(self._windows)

        print(
            f"[GenomeDiffusionDataset] rank={rank}: "
            f"{len(my_files)} files, {total_tokens:,} tokens, "
            f"{len(self._windows):,} windows (seq_len={seq_len})"
        )

    def _extract_windows(
        self,
        ids:     np.ndarray,
        starts:  np.ndarray,
        strands: Optional[np.ndarray],
    ):
        pad_strand = np.zeros(self.seq_len, dtype=np.int8)
        seq_starts = list(starts) + [len(ids)]

        for i in range(len(seq_starts) - 1):
            s = int(seq_starts[i])
            e = int(seq_starts[i + 1])
            seq_ids = ids[s:e]
            seq_str = strands[s:e] if strands is not None else None

            for offset in range(0, max(1, len(seq_ids) - self.seq_len + 1), self.stride):
                id_chunk  = seq_ids[offset : offset + self.seq_len]
                str_chunk = seq_str[offset : offset + self.seq_len] if seq_str is not None else None

                if len(id_chunk) < self.seq_len:
                    pad_n    = self.seq_len - len(id_chunk)
                    id_chunk = np.concatenate(
                        [id_chunk, np.full(pad_n, self.pad_token_id, dtype=np.int32)]
                    )
                    if str_chunk is not None:
                        str_chunk = np.concatenate(
                            [str_chunk, np.zeros(pad_n, dtype=np.int8)]
                        )

                if str_chunk is None:
                    str_chunk = pad_strand.copy()

                self._windows.append((
                    id_chunk.astype(np.int32),
                    str_chunk.astype(np.int8),
                ))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids, strands = self._windows[idx]
        return torch.from_numpy(ids).long(), torch.from_numpy(strands).long()


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

    Each batch is a tuple (input_ids, strand_ids) of shape [B, seq_len].
    """
    hdf5_dir = Path(hdf5_dir)

    files = sorted(hdf5_dir.glob(f"{split}_rank*.h5"))
    if not files:
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
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
