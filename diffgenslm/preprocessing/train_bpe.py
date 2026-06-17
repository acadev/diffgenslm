"""
Train SentencePiece BPE tokenizers for functional and noncoding genomic regions.

Two separate models are trained:
  - functional_bpe.model  : for tRNA/rRNA/regulatory sequences
  - noncoding_bpe.model   : for intergenic/non-annotated sequences

Vocab sizes are tunable; defaults target a ~4k functional + ~8k noncoding split.

Usage:
    python -m diffgenslm.preprocessing.train_bpe \
        --functional_input functional_seqs.txt \
        --noncoding_input  noncoding_seqs.txt \
        --output_dir       /path/to/tokenizer_models \
        --functional_vocab 4096 \
        --noncoding_vocab  8192
"""

import argparse
import os
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]


def _train_spm(
    input_file: str,
    output_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
):
    """Train a SentencePiece model. Requires sentencepiece installed."""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("sentencepiece is required: pip install sentencepiece")

    # Format special tokens list for spm
    user_defined = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=user_defined,
        # DNA-specific settings: treat each uppercase letter as a unit
        byte_fallback=False,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        # Performance
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 8,
    )
    print(f"Trained {model_type} vocab={vocab_size} → {output_prefix}.model")


def _train_hf_bpe(
    input_file: str,
    output_path: str,
    vocab_size: int,
):
    """Train a HuggingFace BPE tokenizer (alternative to SentencePiece)."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError:
        raise ImportError("tokenizers is required: pip install tokenizers")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train(files=[input_file], trainer=trainer)
    tokenizer.save(output_path)
    print(f"Trained HF BPE vocab={vocab_size} → {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Train BPE tokenizers for genomic regions")
    ap.add_argument("--functional_input", type=Path, required=True,
                    help="Text file with functional non-coding sequences (one per line)")
    ap.add_argument("--noncoding_input", type=Path, required=True,
                    help="Text file with intergenic sequences (one per line)")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Directory to write trained tokenizer models")
    ap.add_argument("--functional_vocab", type=int, default=4096)
    ap.add_argument("--noncoding_vocab", type=int, default=8192)
    ap.add_argument("--backend", choices=["sentencepiece", "hf"], default="sentencepiece",
                    help="Tokenizer training backend")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "sentencepiece":
        func_prefix = str(args.output_dir / "functional_bpe")
        nc_prefix = str(args.output_dir / "noncoding_bpe")

        print(f"Training functional BPE (vocab={args.functional_vocab}) ...")
        _train_spm(str(args.functional_input), func_prefix, args.functional_vocab)

        print(f"Training noncoding BPE (vocab={args.noncoding_vocab}) ...")
        _train_spm(str(args.noncoding_input), nc_prefix, args.noncoding_vocab)

        print(f"\nTokenizer models written to {args.output_dir}/")
        print(f"  functional_bpe.model  (vocab={args.functional_vocab})")
        print(f"  noncoding_bpe.model   (vocab={args.noncoding_vocab})")

    else:  # hf
        func_path = str(args.output_dir / "functional_bpe_tokenizer.json")
        nc_path = str(args.output_dir / "noncoding_bpe_tokenizer.json")

        print(f"Training functional HF BPE (vocab={args.functional_vocab}) ...")
        _train_hf_bpe(str(args.functional_input), func_path, args.functional_vocab)

        print(f"Training noncoding HF BPE (vocab={args.noncoding_vocab}) ...")
        _train_hf_bpe(str(args.noncoding_input), nc_path, args.noncoding_vocab)


if __name__ == "__main__":
    main()
