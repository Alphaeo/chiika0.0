"""Assemble data/cs_code_corpus.txt a partir de :
- data/cs_notes.txt (explications CS ecrites a la main)
- un extrait de code reel de chiikaml (C++, projet voisin) et de
  chiikamini (Python, ce projet) -- zero telechargement, zero question
  de licence (ce sont tes propres projets).

Usage : python scripts/build_corpus.py
(suppose que chiikaml vit a ../mlc++ relativement a ce repo -- ajuste
CHIIKAML_ROOT si ton arborescence est differente)
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHIIKAML_ROOT = REPO_ROOT.parent / "mlc++"

OUT_PATH = REPO_ROOT / "data" / "cs_code_corpus.txt"
NOTES_PATH = REPO_ROOT / "data" / "cs_notes.txt"

CHIIKAML_FILES = [
    "include/chiikaml/matrix.hpp",
    "include/chiikaml/knn.hpp",
    "src/knn.cpp",
    "include/chiikaml/kdtree.hpp",
    "src/kdtree.cpp",
    "src/kmeans.cpp",
    "include/chiikaml/decision_tree.hpp",
]

CHIIKAMINI_FILES = [
    "chiikamini/layers/norm.py",
    "chiikamini/layers/rope.py",
    "chiikamini/layers/attention.py",
    "chiikamini/layers/activations.py",
    "chiikamini/layers/mlp.py",
    "chiikamini/layers/block.py",
    "chiikamini/model.py",
]


def section(rel_path: str, path: Path, lang: str) -> str:
    code = path.read_text(encoding="utf-8")
    return f"\n\n# --- {rel_path} ({lang}) ---\n\n```{lang}\n{code}\n```\n"


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
    print(f"Corpus ecrit : {OUT_PATH} ({OUT_PATH.stat().st_size} octets)")


if __name__ == "__main__":
    main()
