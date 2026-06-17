"""
Tokenizer integration for DiffGenSLM.

Wraps GenSLM-2's CompositeGenomeTokenizerV2, which applies three
biologically-motivated sub-tokenizers:

  CDS regions         → CodonTokenizer  (64 triplets + 4 singles + specials)
  Functional non-CDS  → SentencePiece BPE (trained on tRNA/rRNA/regulatory)
  Intergenic          → SentencePiece BPE (trained on non-annotated sequence)

Quick start:
    from diffgenslm.tokenizer import load_tokenizer
    tok = load_tokenizer("/path/to/tokenizer_dir")
    ids = tok.encode("ATGCCCGTT...", add_special_tokens=True)["input_ids"]
"""

from pathlib import Path
from typing import Union


def load_tokenizer(tokenizer_dir: Union[str, Path]):
    """
    Load the CompositeGenomeTokenizerV2 from a directory that contains:
        codon/                  – CodonTokenizer vocab (vocab.json)
        functional_bpe.model    – SentencePiece model for functional regions
        noncoding_bpe.model     – SentencePiece model for intergenic regions

    Args:
        tokenizer_dir: Path to the directory produced by train_bpe.py +
                       create_codon_vocab.py.
    Returns:
        CompositeGenomeTokenizerV2 instance.
    """
    try:
        from genslm2.composite_tokenizer_v2 import CompositeGenomeTokenizerV2
    except ImportError:
        raise ImportError(
            "GenSLM-2 must be installed.\n"
            "  pip install git+https://github.com/StarNetLaboratory/GenSLM-2.git"
        )

    d = Path(tokenizer_dir)
    return CompositeGenomeTokenizerV2(
        codon_tokenizer_path=str(d / "codon"),
        functional_tokenizer_path=str(d / "functional_bpe.model"),
        noncoding_tokenizer_path=str(d / "noncoding_bpe.model"),
    )


def create_codon_vocab(output_dir: Union[str, Path]) -> Path:
    """
    Create the CodonTokenizer vocabulary file (vocab.json) in output_dir/codon/.

    The codon vocabulary is deterministic:
        special tokens  (20)  +  64 codons (ACGT triplets)  +  4 singles (A C G T)
    Total = 88 tokens.

    Returns path to the created vocab.json.
    """
    import itertools
    import json

    special = [
        "<unk>", "<s>", "</s>", "<mask>", "<sep>", "<cls>", "<pad>",
        "<thinking>", "</thinking>", "<solution>", "</solution>",
        "<task_01>", "<task_02>", "<task_03>", "<task_04>", "<task_05>",
        "<task_06>", "<task_07>", "<task_08>", "<task_09>", "<task_10>",
    ]
    singles = ["A", "C", "G", "T"]
    codons = ["".join(c) for c in itertools.product("ACGT", repeat=3)]

    vocab = {}
    for i, tok in enumerate(special + singles + codons):
        vocab[tok] = i

    out_dir = Path(output_dir) / "codon"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w") as fh:
        json.dump(vocab, fh, indent=2)

    print(f"Created codon vocab ({len(vocab)} tokens) → {vocab_path}")
    return vocab_path
