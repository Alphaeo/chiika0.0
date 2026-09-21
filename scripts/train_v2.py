"""Entraine chiikamini sur data/train_v4.txt (ou --train-file) avec le tokenizer BPE de domaine
(data/tokenizer_bpe8192.json, voir scripts/train_tokenizer.py), avec suivi
train/eval a chaque epoch, arret anticipe et restauration du meilleur epoch.

Sauvegarde en LOCAL uniquement (pas de push HF) dans --out : poids, config,
tokenizer.json (lu ensuite par scripts/chat.py) et history.json.

Compare le resultat en bits/octet a la reference du modele publie
(GPT-2, 27k tokens). Le jeu de TEST (data/test_v3.txt) n'est mesure qu'avec
--report-test : a n'utiliser qu'une fois par modele final, jamais pour
choisir entre variantes (utiliser eval_v3 pour ca).

Usage :
    python scripts/train_v2.py
    python scripts/train_v2.py --lr 1e-3 --epochs 30 --patience 3
"""

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiikamini.checkpoint import save_checkpoint
from chiikamini.config import TextConfig
from chiikamini.data import RandomWindowTextDataset, ToyTextDataset
from chiikamini.evals import bits_per_byte, eval_fixed_prompts, perplexity
from chiikamini.model import ChiikaMiniForCausalLM
from chiikamini.tokenizer import ChiikaTokenizer
from chiikamini.train import train_loop

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

# Modele publie (GPT-2, entraine sur 27k tokens), mesure en bits/octet sur le dataset v3.
REFERENCE_BPB = {"eval": 5.579, "test": 5.224}

PROMPTS = ["def forward(self", "class Matrix", "A binary search tree", "for (std::size_t i = 0;"]


def check_dataset_fingerprint(train_file: Path) -> None:
    """Le dataset doit etre celui du manifest : sinon deux runs ne sont pas comparables."""
    manifest_file = DATA / f"manifest_{train_file.stem.split('_')[-1]}.json"   # train_v5.txt -> manifest_v5.json
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for split, path in (("train", train_file), ("eval", DATA / "eval_v3.txt"), ("test", DATA / "test_v3.txt")):
        text = path.read_text(encoding="utf-8")
        if hashlib.sha1(text.encode()).hexdigest() != manifest["sha1"][split]:
            print(f"!!! ATTENTION : {path.name} ne correspond plus a {manifest_file.name} (dataset modifie) -- "
                  "les resultats ne sont PAS comparables aux runs precedents.")
            return
    print(f"dataset conforme a {manifest_file.name} (sha1 identiques)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reentraine chiikamini (tokenizer BPE de domaine)")
    parser.add_argument("--train-file", type=Path, default=DATA / "train_v4.txt",
                        help="train_v4.txt (defaut), train_v5.txt (plus de code du Hub) ou train_v3.txt (ton code seul)")
    parser.add_argument("--tokenizer", type=Path, default=DATA / "tokenizer_bpe8192.json")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="epochs partiels de k pas (gros corpus) ; defaut : epoch complet")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=64, help="longueur de contexte (fenetres d'entrainement et max_seq_len)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation", default="xielu", choices=["xielu", "swiglu"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=176)
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout residuel (TODO 4b)")
    parser.add_argument("--random-windows", action="store_true",
                        help="fenetres a offset aleatoire a chaque epoch (TODO 4a) au lieu de fenetres fixes")
    parser.add_argument("--report-test", action="store_true", help="mesure aussi test_v3 (une seule fois !)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "chiikamini-bpe4096-xielu")
    args = parser.parse_args()

    check_dataset_fingerprint(args.train_file)
    tokenizer = ChiikaTokenizer(tokenizer_path=args.tokenizer)
    train_ds = ToyTextDataset(args.train_file, tokenizer, seq_len=args.seq_len)
    if args.random_windows:   # meme nombre de fenetres par epoch que la baseline : meme budget de calcul
        train_ds = RandomWindowTextDataset(args.train_file, tokenizer, seq_len=args.seq_len,
                                           windows_per_epoch=len(train_ds), seed=0)
    eval_ds = ToyTextDataset(DATA / "eval_v3.txt", tokenizer, seq_len=args.seq_len)
    print(f"tokenizer vocab={tokenizer.vocab_size} | fenetres train={len(train_ds)} eval={len(eval_ds)}")

    cfg = TextConfig(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        max_seq_len=args.seq_len,
        ffn_activation=args.activation,
        dropout=args.dropout,
    )
    model = ChiikaMiniForCausalLM(cfg)

    started = time.time()
    history = train_loop(
        model, train_ds, n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        warmup_steps=args.warmup_steps, eval_dataset=eval_ds,
        patience=args.patience, restore_best=True, steps_per_epoch=args.steps_per_epoch, seed=args.seed,
    )

    train_seconds = time.time() - started
    eval_path = DATA / "eval_v3.txt"
    bpb = bits_per_byte(model, tokenizer, eval_path, seq_len=args.seq_len, batch_size=args.batch_size)
    ppl = perplexity(model, eval_ds, batch_size=args.batch_size)
    results = {"eval": bpb}
    print(f"\nRESULTAT  eval : {bpb:.3f} bits/octet | perplexite par token {ppl:.1f}")
    if args.report_test:
        results["test"] = bits_per_byte(model, tokenizer, DATA / "test_v3.txt", seq_len=args.seq_len, batch_size=args.batch_size)
    for split, value in results.items():
        ref = REFERENCE_BPB[split]
        print(f"  {split:4s}: {value:.3f} bits/octet  (modele publie GPT-2/27k : {ref:.3f}  ->  {value - ref:+.3f}, {100 * (value / ref - 1):+.1f} %)")

    print("\nCompletions (greedy) :")
    for prompt, text in zip(PROMPTS, eval_fixed_prompts(model, tokenizer, PROMPTS, max_new_tokens=30)):
        print(f"  {prompt!r} -> {text!r}")

    save_checkpoint(model, cfg, args.out)
    tokenizer.tokenizer.save(str(args.out / "tokenizer.json"))
    (args.out / "history.json").write_text(json.dumps({
        "config": dataclasses.asdict(cfg),
        "lr": args.lr,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "train_file": args.train_file.name,
        "tokenizer": args.tokenizer.name,
        "steps_per_epoch": args.steps_per_epoch,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(train_seconds),
        "bits_per_byte": results,
        "reference_bits_per_byte": REFERENCE_BPB,
        "epochs": history,
    }, indent=1), encoding="utf-8")
    print(f"\nCheckpoint sauvegarde dans {args.out}")


if __name__ == "__main__":
    main()
