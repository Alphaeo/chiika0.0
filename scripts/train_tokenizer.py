"""Entraine le tokenizer BPE de domaine sur data/train_v4.txt (le TRAIN
uniquement : jamais l'eval, sinon le vocabulaire fuit des stats de l'eval),
le sauvegarde, et le compare a GPT-2 sur data/eval_v3.txt.

Usage :
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 8192 --dim 128

Colonnes du rapport :
- octets/token : plus c'est haut, plus le tokenizer compresse (fenetre de
  contexte plus utile, moins de calcul pour le meme texte)
- vocab vu au train : part des tokens du vocabulaire qui apparaissent au
  moins une fois dans le train. Un token jamais vu a une ligne d'embedding
  jamais entrainee.
- tokens eval inconnus : part des tokens de l'eval jamais vus au train
- embedding : parametres de la table d'embedding (vocab x dim), a comparer
  au reste du modele
"""

import argparse
import sys
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiikamini.tokenizer import ChiikaTokenizer, train_bpe_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"


def report_row(name: str, encode, vocab_size: int, train_text: str, eval_text: str, dim: int) -> str:
    train_ids, eval_ids = encode(train_text), encode(eval_text)
    seen = set(train_ids)
    unknown = sum(t not in seen for t in eval_ids) / len(eval_ids)
    bytes_per_token = len(eval_text.encode("utf-8")) / len(eval_ids)
    return (f"{name:12s} vocab={vocab_size:6d}  tokens train={len(train_ids):7d}  eval={len(eval_ids):6d}  "
            f"octets/token={bytes_per_token:.2f}  vocab vu au train={len(seen) / vocab_size:4.0%}  "
            f"eval inconnus={unknown:5.1%}  embedding={vocab_size * dim / 1e6:.2f}M")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraine un BPE de domaine et le compare a GPT-2")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--train-file", type=Path, default=DATA / "train_v4.txt",
                        help="corpus d'entrainement du tokenizer (JAMAIS l'eval ni le test)")
    parser.add_argument("--dim", type=int, default=128, help="dimension d'embedding, pour chiffrer la table")
    parser.add_argument("--out", type=Path, default=None, help="defaut : data/tokenizer_bpe<vocab>.json")
    args = parser.parse_args()

    train_text = args.train_file.read_text(encoding="utf-8")
    eval_text = (DATA / "eval_v3.txt").read_text(encoding="utf-8")

    tok = train_bpe_tokenizer([train_text], vocab_size=args.vocab_size, special_tokens=["<image>"])
    out = args.out or DATA / f"tokenizer_bpe{args.vocab_size}.json"
    tok.save(str(out))
    print(f"Tokenizer sauvegarde : {out} (vocabulaire reel : {tok.get_vocab_size()})\n")

    gpt2 = ChiikaTokenizer()
    ours = ChiikaTokenizer(tokenizer_path=out)
    print(report_row("gpt2", gpt2.encode, gpt2.vocab_size, train_text, eval_text, args.dim))
    print(report_row(f"bpe{args.vocab_size}", ours.encode, ours.vocab_size, train_text, eval_text, args.dim))


if __name__ == "__main__":
    main()
