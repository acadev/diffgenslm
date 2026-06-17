"""
DiffGenSLM adapter for the biolab evaluation harness.

Implements the biolab ``LM`` protocol so DiffGenSLM can be evaluated on
any biolab task (GC-content regression, GUE classification, PATRIC secondary
structure, etc.) without modification to the biolab codebase.

Usage in a biolab eval config (YAML):

    lm_config:
      name: DiffGenSLM
      checkpoint_path: /path/to/checkpoints/best.pt
      tokenizer_dir:   /path/to/tokenizer

    task_configs:
      - name: GCContent
        dataset_name_or_path: /path/to/gc_content_dataset

Then run:
    python -m biolab.evaluate --config eval_config.yaml

Tokenisation strategy
---------------------
biolab tasks supply raw DNA strings without GFF/GTO annotation, so we
cannot use the composite tokeniser's CDS path.  Instead we use a
character-level tokeniser that maps each nucleotide A/T/C/G to the
corresponding single-base token in the codon vocabulary (vocab.json).
This gives ``model_encoding = 'char'``, which biolab routes to:
  - sequence-level tasks  → ``average_pool`` transform
  - nucleotide-level tasks → ``full_sequence`` transform
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from biolab.api.modeling import HDF5CachedList, LMConfig, SequenceModelOutput

from ..models.diffgenome import DiffGenomeConfig, DiffGenomeModel
from ..diffusion.sample import sample as _diffusion_sample


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class NuclCharTokenizer:
    """
    Character-level DNA tokeniser backed by the codon vocabulary.

    Maps each nucleotide to its single-base token ID in
    ``tokenizer_dir/codon/vocab.json``.  Provides ``tokenize()`` for the
    biolab ``SuperResolution`` transform (returns list[str]) and
    ``encode()`` for internal use (returns list[int]).
    """

    def __init__(self, codon_vocab_path: str | Path):
        with open(codon_vocab_path) as fh:
            vocab: dict[str, int] = json.load(fh)

        self._vocab = vocab
        self._rev_vocab = {v: k for k, v in vocab.items()}

        unk = vocab.get("<unk>", 3)
        self._base_ids: dict[str, int] = {
            "A": vocab.get("A", unk),
            "T": vocab.get("T", unk),
            "C": vocab.get("C", unk),
            "G": vocab.get("G", unk),
        }
        self.pad_id  = vocab.get("<pad>",  0)
        self.mask_id = vocab.get("<mask>", 4)
        self.unk_id  = unk

    # ------------------------------------------------------------------
    # biolab-facing API

    def tokenize(self, sequence: str) -> list[str]:
        """Return a list of single-character strings (one per nucleotide).

        Called by the biolab ``SuperResolution`` transform; ``len(piece)``
        must equal the number of nucleotides the piece covers, so we return
        single characters.
        """
        return list(sequence.upper())

    # ------------------------------------------------------------------
    # Internal API

    def encode(self, sequence: str, max_length: int | None = None) -> list[int]:
        ids = [self._base_ids.get(b, self.unk_id) for b in sequence.upper()]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._rev_vocab.get(i, "N") for i in ids)

    def __len__(self) -> int:
        return len(self._vocab)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DiffGenSLMConfig(LMConfig):
    """Configuration for the DiffGenSLM biolab adapter."""

    name: Literal["DiffGenSLM"] = "DiffGenSLM"

    # Path to a DiffGenSLM checkpoint (best.pt or checkpoint.pt)
    checkpoint_path: str
    # Directory that contains codon/vocab.json (output of create_codon_vocab)
    tokenizer_dir: str

    # Embedding / generation settings
    max_length: int = 2048
    half_precision: bool = False
    # Sampler settings for generate_sequences
    num_sample_steps: int = 64
    sample_temperature: float = 1.0
    sample_schedule: str = "cosine"


# ---------------------------------------------------------------------------
# LM adapter
# ---------------------------------------------------------------------------

class DiffGenSLM:
    """
    biolab ``LM`` adapter for DiffGenSLM.

    Implements ``generate_embeddings`` (returns per-nucleotide hidden states)
    and ``generate_sequences`` (iterative confidence-ranked unmasking).
    """

    model_input: str    = "dna"
    model_encoding: str = "char"

    def __init__(self, config: DiffGenSLMConfig) -> None:
        self.config = config

        # ── Load tokeniser ────────────────────────────────────────────
        vocab_path = Path(config.tokenizer_dir) / "codon" / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Codon vocab not found at {vocab_path}. "
                "Run `diffgenslm.tokenizer.create_codon_vocab(tokenizer_dir)` first."
            )
        self._tokenizer = NuclCharTokenizer(vocab_path)

        # ── Load model from checkpoint ────────────────────────────────
        ckpt = torch.load(config.checkpoint_path, map_location="cpu",
                          weights_only=False)
        model_cfg = DiffGenomeConfig(**ckpt["model_config"])
        model = DiffGenomeModel(model_cfg)
        model.load_state_dict(ckpt["model"])

        if config.half_precision:
            model = model.half()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self._device)
        model.eval()

        self._model     = model
        self._model_cfg = model_cfg

    # ------------------------------------------------------------------
    # biolab LM protocol properties

    @property
    def tokenizer(self) -> NuclCharTokenizer:
        return self._tokenizer

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        # We handle tokenisation internally; biolab tasks don't call this
        # for char-level models, but the protocol requires the property.
        return {}

    @property
    def dataloader_config(self) -> dict[str, Any]:
        return self.config.dataloader_config.model_dump()

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Embedding generation

    def generate_embeddings(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
    ) -> list[SequenceModelOutput]:
        """
        Encode DNA strings and return per-nucleotide hidden states.

        Each output has:
          ``embedding``: np.ndarray [L, hidden_size]  (pre-lm_head hidden state)
          ``logits``:    np.ndarray [L, vocab_size]

        The biolab ``average_pool`` transform will mean-pool ``embedding``
        to [hidden_size] for sequence-level tasks; ``full_sequence`` passes
        it through unchanged for nucleotide-level tasks.
        """
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []

        batch_size = self.config.dataloader_config.batch_size

        # Tokenise all sequences and record their true lengths
        all_ids:    list[list[int]] = []
        seq_lens:   list[int]       = []
        for seq in sequences:
            ids = self._tokenizer.encode(seq, max_length=self.config.max_length)
            all_ids.append(ids)
            seq_lens.append(len(ids))

        max_len = max(seq_lens) if seq_lens else 1

        # Build padded tensor [N, max_len]
        pad_id = self._tokenizer.pad_id
        input_tensor = torch.full(
            (len(sequences), max_len), pad_id, dtype=torch.long
        )
        for i, ids in enumerate(all_ids):
            input_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        dataset    = TensorDataset(input_tensor, torch.tensor(seq_lens))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            for batch_ids, batch_lens in dataloader:
                batch_ids = batch_ids.to(self._device)
                logits, hidden = self._model(
                    batch_ids, return_hidden_states=True
                )  # [B, L, V], [B, L, D]

                logits = logits.float().cpu().numpy()
                hidden = hidden.float().cpu().numpy()

                for i, seq_len in enumerate(batch_lens.tolist()):
                    output = SequenceModelOutput(
                        embedding=hidden[i, :seq_len, :],
                        logits=logits[i,   :seq_len, :],
                    )
                    model_outputs.append(output)

        return model_outputs

    # ------------------------------------------------------------------
    # Sequence generation

    def generate_sequences(
        self,
        input: list[str],
        model_outputs: HDF5CachedList | None = None,
    ) -> list[SequenceModelOutput]:
        """
        Generate DNA sequences via iterative confidence-ranked unmasking.

        Each string in ``input`` is treated as an optional prefix/context:
          - Non-N characters are treated as fixed (context to infill around).
          - Empty strings or all-N strings trigger unconditional generation
            of length ``max_length``.

        Returns ``SequenceModelOutput`` with:
          ``sequence``: decoded DNA string
          ``logits``:   np.ndarray of final-pass logits [L, V]
        """
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []

        mask_id = self._model_cfg.mask_token_id
        pad_id  = self._model_cfg.pad_token_id

        for seq_str in input:
            # Build context: encode what we have, fill the rest with mask
            target_len = min(
                max(len(seq_str), 1),
                self.config.max_length,
            ) if seq_str.strip("N") else self.config.max_length

            context = torch.full((1, target_len), mask_id, dtype=torch.long)

            fixed_positions: torch.Tensor | None = None
            if seq_str:
                ids = self._tokenizer.encode(seq_str, max_length=target_len)
                is_fixed = torch.zeros(1, target_len, dtype=torch.bool)
                for pos, (char, token_id) in enumerate(
                    zip(seq_str[:target_len], ids)
                ):
                    if char.upper() != "N":
                        context[0, pos] = token_id
                        is_fixed[0, pos] = True
                fixed_positions = is_fixed

            context = context.to(self._device)
            if fixed_positions is not None:
                fixed_positions = fixed_positions.to(self._device)

            with torch.no_grad():
                out_ids = _diffusion_sample(
                    self._model,
                    context,
                    mask_id,
                    pad_id,
                    num_steps=self.config.num_sample_steps,
                    temperature=self.config.sample_temperature,
                    schedule=self.config.sample_schedule,
                    fixed_positions=fixed_positions,
                )
                # Final logit pass for the completed sequence
                final_logits = self._model(out_ids).float().cpu().numpy()

            decoded = self._tokenizer.decode(out_ids[0].tolist())
            model_outputs.append(
                SequenceModelOutput(
                    sequence=decoded,
                    logits=final_logits[0],
                )
            )

        return model_outputs


# ---------------------------------------------------------------------------
# Registry entry (mirrors the pattern in biolab/modeling/models/)
# ---------------------------------------------------------------------------

diffgenslm_models = {
    DiffGenSLMConfig: DiffGenSLM,
}
