"""Assemble data/eval_corpus.txt : un jeu d'evaluation que le modele n'a
jamais vu a l'entrainement.

Contenu :
- data/eval_notes.txt : notes CS sur des sujets ABSENTS de data/cs_notes.txt
  (OS, concurrence, reseau, bases de donnees, compilation, cache...)
- des fichiers de code de chiikaml (C++) et chiikamini (Python) qui ne sont
  PAS dans scripts/build_corpus.py.

Verifie ensuite qu'aucune fenetre de 32 tokens du jeu d'eval n'apparait
dans le corpus d'entrainement (contamination). Si c'est le cas, le script
l'affiche : dans ce cas la perplexite mesuree serait trop optimiste.

Usage : python scripts/build_eval_set.py
"""

import sys
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiikamini.tokenizer import ChiikaTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
CHIIKAML_ROOT = REPO_ROOT.parent / "mlc++"

NOTES_PATH = REPO_ROOT / "data" / "eval_notes.txt"
TRAIN_PATH = REPO_ROOT / "data" / "cs_code_corpus.txt"
OUT_PATH = REPO_ROOT / "data" / "eval_corpus.txt"

CHIIKAML_FILES = [
    "include/chiikaml/linear_regression.hpp",
    "src/linear_regression.cpp",
    "include/chiikaml/ridge_regression.hpp",
    "include/chiikaml/random_forest.hpp",
    "src/random_forest.cpp",
    "include/chiikaml/kmeans.hpp",
]

CHIIKAMINI_FILES = [
    "chiikamini/generate.py",
    "chiikamini/train.py",
    "chiikamini/checkpoint.py",
    "chiikamini/utils.py",
]

WINDOW = 32


def section(rel_path: str, path: Path, lang: str) -> str:
    code = path.read_text(encoding="utf-8")
    return f"\n\n# --- {rel_path} ({lang}) ---\n\n```{lang}\n{code}\n```\n"


def contamination(train_ids: list[int], eval_ids: list[int]) -> float:
    """Fraction des fenetres de WINDOW tokens de l'eval presentes dans le train."""
    train_windows = {tuple(train_ids[i : i + WINDOW]) for i in range(len(train_ids) - WINDOW + 1)}
    eval_windows = [tuple(eval_ids[i : i + WINDOW]) for i in range(len(eval_ids) - WINDOW + 1)]
    seen = sum(w in train_windows for w in eval_windows)
    return seen / len(eval_windows)


def main() -> None:
    parts = [NOTES_PATH.read_text(encoding="utf-8")]

    for rel in CHIIKAML_FILES:
        path = CHIIKAML_ROOT / rel
        if path.exists():
            parts.append(section(rel, path, "cpp"))
        else:
            print(f"skip (introuvable) : {path}")

    for rel in CHIIKAMINI_FILES:
        path = REPO_ROOT / rel
        if path.exists():
            parts.append(section(rel, path, "python"))
        else:
            print(f"skip (introuvable) : {path}")

    OUT_PATH.write_text("".join(parts), encoding="utf-8")

    tokenizer = ChiikaTokenizer()
    eval_ids = tokenizer.encode(OUT_PATH.read_text(encoding="utf-8"))
    train_ids = tokenizer.encode(TRAIN_PATH.read_text(encoding="utf-8"))
    rate = contamination(train_ids, eval_ids)

    print(f"Jeu d'eval ecrit : {OUT_PATH} ({OUT_PATH.stat().st_size} octets, {len(eval_ids)} tokens)")
    print(f"Contamination (fenetres de {WINDOW} tokens presentes dans le train) : {rate:.2%}")


if __name__ == "__main__":
    main()
