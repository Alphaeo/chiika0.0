"""Echelle de tailles : la capacite du modele est-elle un facteur limitant avec
NOS donnees ? Entraine plusieurs tailles dans des conditions identiques (dataset
gele, tokenizer, patience ; train_v4 + tokenizer 8192 par defaut) et compare la loss en bits/octet sur eval_v3
(validation ; jamais le test).

Le LR baisse avec la largeur (usage courant : les grands modeles supportent moins
un LR eleve). Ce choix reste une confusion possible : a garder en tete a la lecture.

Usage : python scripts/scale.py [--only S M L]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SIZES = {
    # nom: (dim, couches, tetes, tetes KV, ffn, lr)
    "S": (128, 4, 4, 2, 512, 3e-3),
    "M": (192, 6, 6, 2, 768, 2e-3),
    "L": (256, 8, 8, 2, 1024, 1.5e-3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Echelle de tailles de modele")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="epochs partiels (gros corpus)")
    parser.add_argument("--tag", default="v4", help="suffixe des dossiers de sortie (evite d'ecraser d'anciens runs)")
    args = parser.parse_args()

    rows = []
    for name, (dim, layers, heads, kv, ffn, lr) in SIZES.items():
        if args.only and name not in args.only:
            continue
        out = REPO_ROOT / "checkpoints" / f"scale-{args.tag}-{name}"
        cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts" / "train_v2.py"),
               "--dim", str(dim), "--n-layers", str(layers), "--n-heads", str(heads), "--n-kv-heads", str(kv),
               "--ffn-hidden-dim", str(ffn), "--lr", str(lr), "--epochs", str(args.epochs),
               "--patience", str(args.patience), "--out", str(out)]
        if args.steps_per_epoch:
            cmd += ["--steps-per-epoch", str(args.steps_per_epoch)]
        print(f"=== {name} : dim={dim} couches={layers} lr={lr:g} ===", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        history_file = out / "history.json"
        if proc.returncode != 0 or not history_file.exists():
            print("  ECHEC :", proc.stderr.strip().splitlines()[-1:], flush=True)
            continue
        h = json.loads(history_file.read_text(encoding="utf-8"))
        epochs = [e for e in h["epochs"] if "eval_ppl" in e]
        best = min(epochs, key=lambda e: e["eval_ppl"])
        gap = best["eval_ppl"] / best["train_ppl"]
        rows.append((name, h["n_parameters"], best["epoch"], best["train_ppl"], best["eval_ppl"],
                     h["bits_per_byte"]["eval"], gap, h["train_seconds"]))
        print(f"  {h['n_parameters'] / 1e6:.2f}M params | meilleur epoch {best['epoch']} | "
              f"eval {h['bits_per_byte']['eval']:.3f} bits/octet | ecart train/eval x{gap:.1f} | {h['train_seconds']} s", flush=True)

    print("\n=== Resume (eval_v3, plus bas = mieux) ===")
    print(f"{'taille':6s} {'params':>8s} {'meilleur ep.':>12s} {'train ppl':>10s} {'eval ppl':>9s} {'bits/octet':>11s} {'ecart':>6s} {'temps':>7s}")
    for name, n, ep, tp, ep_ppl, bpb, gap, sec in rows:
        print(f"{name:6s} {n / 1e6:7.2f}M {ep:12d} {tp:10.1f} {ep_ppl:9.1f} {bpb:11.3f} {'x%.1f' % gap:>6s} {sec:6d}s")


if __name__ == "__main__":
    main()
