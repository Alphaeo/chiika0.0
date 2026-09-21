"""Ablation des TODO 4a (fenetres a offset aleatoire) et 4b (dropout) :
entraine chaque variante dans des conditions IDENTIQUES (dataset v3 gele,
seed, LR, patience) et compare la loss en bits/octet sur eval_v3 (le jeu de
VALIDATION). Ne touche jamais au jeu de test.

Variantes : baseline | random-windows | dropout | both

Usage :
    python scripts/ablate.py
    python scripts/ablate.py --dropout 0.1 --only baseline dropout

Chaque variante prend quelques minutes sur CPU. Necessite que 4a et/ou 4b
soient implementes (sinon la variante correspondante echoue avec
NotImplementedError et est marquee comme telle).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation fenetres aleatoires / dropout")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--only", nargs="+", default=None, help="sous-ensemble de variantes")
    args = parser.parse_args()

    dropout = ["--dropout", str(args.dropout)]
    variants = {
        "baseline": [],
        "random-windows": ["--random-windows"],
        "dropout": dropout,
        "both": ["--random-windows", *dropout],
    }
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}

    rows = []
    for name, extra in variants.items():
        out = REPO_ROOT / "checkpoints" / f"ablation-{name}"
        cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts" / "train_v2.py"), "--lr", str(args.lr),
               "--epochs", str(args.epochs), "--patience", str(args.patience), "--out", str(out), *extra]
        print(f"\n=== {name} : {' '.join(extra) or '(aucune option)'} ===", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        history_file = out / "history.json"
        if proc.returncode != 0 or not history_file.exists():
            reason = "NotImplementedError (TODO pas encore implemente)" if "NotImplementedError" in proc.stderr else proc.stderr.strip().splitlines()[-1:]
            print(f"  ECHEC : {reason}")
            rows.append((name, None, None, None))
            continue
        history = json.loads(history_file.read_text(encoding="utf-8"))
        epochs = [e for e in history["epochs"] if "eval_ppl" in e]
        best = min(epochs, key=lambda e: e["eval_ppl"])
        last = epochs[-1]
        gap = last["eval_ppl"] / last["train_ppl"]
        rows.append((name, history["bits_per_byte"]["eval"], best["epoch"], gap))
        print(f"  eval = {history['bits_per_byte']['eval']:.3f} bits/octet (meilleur epoch {best['epoch']}, "
              f"ecart eval/train au dernier epoch x{gap:.1f})", flush=True)

    print("\n=== Resume (bits/octet sur eval_v3, plus bas = mieux) ===")
    base = next((r[1] for r in rows if r[0] == "baseline" and r[1] is not None), None)
    for name, bpb, best_epoch, gap in rows:
        if bpb is None:
            print(f"  {name:15s} -")
            continue
        delta = f"  ({bpb - base:+.3f} vs baseline)" if base is not None and name != "baseline" else ""
        print(f"  {name:15s} {bpb:.3f}  meilleur epoch {best_epoch}  ecart x{gap:.1f}{delta}")


if __name__ == "__main__":
    main()
