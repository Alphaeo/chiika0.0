"""REPL pour tester chiikamini a la main : charge un checkpoint (local
ou telecharge depuis le Hugging Face Hub) et genere une completion pour
chaque prompt tape au clavier.

Usage :
    python scripts/chat.py
    python scripts/chat.py --checkpoint checkpoints/chiikamini-xielu
    python scripts/chat.py --checkpoint Crocolil/Chiikamini --from-hub
    python scripts/chat.py --temperature 0 --max-new-tokens 60

Tape 'exit' (ou Ctrl+C / Ctrl+D) pour quitter.
"""

import argparse
import sys
from pathlib import Path

# `python scripts/chat.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chiikamini.checkpoint import load_checkpoint
from chiikamini.generate import generate
from chiikamini.tokenizer import ChiikaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="REPL pour tester chiikamini a la main")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/chiikamini-xielu",
        help="dossier local (defaut) ou repo_id du Hugging Face Hub avec --from-hub",
    )
    parser.add_argument("--from-hub", action="store_true", help="telecharge --checkpoint depuis le Hub")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy (deterministe)")
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint
    if args.from_hub:
        from huggingface_hub import snapshot_download

        print(f"Telechargement de {args.checkpoint} depuis le Hub...")
        checkpoint_dir = snapshot_download(args.checkpoint)

    print(f"Chargement du checkpoint depuis {checkpoint_dir}...")
    model = load_checkpoint(checkpoint_dir)
    model.eval()

    tokenizer_file = Path(checkpoint_dir) / "tokenizer.json"
    tokenizer = ChiikaTokenizer(tokenizer_path=tokenizer_file if tokenizer_file.exists() else None)

    print("Modele charge. Tape un prompt (ou 'exit' pour quitter).\n")

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.strip().lower() in ("exit", "quit"):
            break
        if not prompt.strip():
            continue

        ids = torch.tensor([tokenizer.encode(prompt)])
        out = generate(
            model,
            ids,
            images=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.temperature > 0 else None,
        )
        print(tokenizer.decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
