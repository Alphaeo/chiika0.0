"""Construit le dataset v3 : data/train_v3.txt (~200k tokens), data/eval_v3.txt
(VALIDATION : choix du LR, de l'epoch, des variantes), data/test_v3.txt (TEST :
chiffre final, utilise une seule fois) et data/manifest_v3.json (tracabilite +
empreintes sha1, pour detecter toute derive du dataset).

Sources : tes repos publics (chiikaml, insta-trend-ai, diamant) et les notes CS (data/cs_notes.txt -> train,
data/eval_notes.txt -> eval).

Pipeline :
1. Collecte + filtres : pas de dossiers vendored (node_modules, Library,
   build...), pas de fichiers generes/minifies (lignes > 400 caracteres),
   pas de fichier contenant un motif de secret.
2. Split PAR FICHIER, deterministe (hash du chemin) : ~10 % des fichiers
   vont a l'eval.
3. Deduplication : fichiers identiques, puis fichiers dont > 60 % des
   lignes (>= 20 caracteres) existent deja dans un fichier garde.
4. Anti-contamination : un fichier d'eval est retire s'il partage > 50 %
   de ses lignes avec le train, ou > 10 % de ses fenetres de 32 tokens
   avec le train RETENU (le controle par lignes rate les longues chaines
   className quasi repetees des pages TSX ; le controle par fenetres non).
5. Equilibrage : quotas par categorie (cpp/python/typescript/javascript/
   prose) avec redistribution si une categorie manque de matiere, pour
   ne pas laisser une seule langue dominer.
6. Rapport : composition, elements retires, taux de contamination
   (fenetres de 32 tokens).

Les .txt generes sont ignores par git (regenerables, deterministes) ;
seul le manifest est versionne.

Usage : python scripts/build_dataset.py
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chiikamini.tokenizer import ChiikaTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS = REPO_ROOT.parent
DATA = REPO_ROOT / "data"

SOURCES = {
    "chiikaml": PROJECTS / "mlc++",
    "insta-trend-ai": PROJECTS / "insta-trend-ai",
    "diamant": PROJECTS / "diamant",
    # PAS "chiikamini" : ce repo est modifie en permanence, l'inclure ferait deriver le
    # dataset a chaque edition et rendrait les comparaisons entre runs invalides.
}

CATEGORY_BY_EXT = {
    ".py": "python",
    ".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".c": "cpp",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".md": "prose",
}

SKIP_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__", "build", "dist",
    ".next", ".cache", "out", ".idea", ".vscode", ".pytest_cache", "site-packages",
    ".qodo", "vendor", "target", "bin", "obj", "coverage", ".turbo", "_deps",
    "data", "checkpoints", "Library",
}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

MAX_FILE_BYTES = 100_000
MIN_FILE_BYTES = 200
MAX_LINE_LEN = 400

SECRET = re.compile(
    r"(api[_-]?key|secret|passwd|password|token)\s*[=:]\s*['\"][^'\"\s]{12,}"
    r"|BEGIN (RSA |EC )?PRIVATE KEY|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
    r"|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}",
    re.I,
)

TRAIN_BUDGET = 200_000
EVAL_BUDGET = 40_000       # eval_v3 = jeu de VALIDATION : sert a choisir LR, epoch d'arret, variantes
TEST_BUDGET = 30_000       # test_v3  = jeu de TEST : utilise UNE fois, pour rapporter le chiffre final
TRAIN_FRACTIONS = {"python": 0.20, "cpp": 0.25, "typescript": 0.20, "javascript": 0.15, "prose": 0.20}
EVAL_EVERY_N = 6           # ~1/6 des fichiers candidats a l'eval (une partie est retiree ensuite)
NEAR_DUP_TRAIN = 0.60
NEAR_DUP_EVAL_VS_TRAIN = 0.50
CONTAMINATION_WINDOW = 32
WINDOW_DROP_THRESHOLD = 0.10   # un fichier d'eval dont > 10 % des fenetres sont dans le train est retire


def order_key(project: str, rel: str) -> str:
    return hashlib.sha1(f"{project}/{rel}".encode()).hexdigest()


def collect(stats: Counter) -> list[dict]:
    files = []
    for project, root in SOURCES.items():
        if not root.exists():
            print(f"skip source introuvable : {root}")
            continue
        for d, dirs, names in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for name in sorted(names):
                ext = os.path.splitext(name)[1].lower()
                if ext not in CATEGORY_BY_EXT or name in SKIP_FILES or name.endswith(".min.js"):
                    continue
                path = Path(d) / name
                size = path.stat().st_size
                if size > MAX_FILE_BYTES or size < MIN_FILE_BYTES:
                    stats["ignore_taille"] += 1
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    stats["ignore_illisible"] += 1
                    continue
                if max((len(l) for l in text.splitlines()), default=0) > MAX_LINE_LEN:
                    stats["ignore_genere_ou_minifie"] += 1
                    continue
                if SECRET.search(text):
                    stats["ignore_secret_suspect"] += 1
                    print(f"  ! secret suspect, fichier exclu : {path}")
                    continue
                rel = path.relative_to(root).as_posix()
                files.append({
                    "project": project, "rel": rel, "category": CATEGORY_BY_EXT[ext],
                    "text": text, "order": order_key(project, rel),
                })
    return files


def norm_lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if len(l.strip()) >= 20]


def dup_ratio(lines: list[str], seen: set[str]) -> float:
    return sum(l in seen for l in lines) / len(lines) if lines else 0.0


def dedupe(files: list[dict], stats: Counter, label: str, reference_lines: set[str] | None = None,
           threshold: float = NEAR_DUP_TRAIN) -> tuple[list[dict], set[str]]:
    seen_hashes: set[str] = set()
    seen_lines: set[str] = set()
    kept = []
    for f in sorted(files, key=lambda f: (f["project"], f["rel"])):
        h = hashlib.sha1(f["text"].encode()).hexdigest()
        lines = norm_lines(f["text"])
        if h in seen_hashes:
            stats[f"{label}_retire_identique"] += 1
            continue
        against = seen_lines | reference_lines if reference_lines is not None else seen_lines
        if dup_ratio(lines, against) > threshold:
            stats[f"{label}_retire_quasi_doublon"] += 1
            continue
        seen_hashes.add(h)
        seen_lines.update(lines)
        kept.append(f)
    return kept, seen_lines


def select(files: list[dict], budget: int, fractions: dict[str, float]) -> list[dict]:
    """Quotas par categorie avec redistribution du budget inutilise."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(files, key=lambda f: f["order"]):
        by_cat[f["category"]].append(f)
    avail = {c: sum(f["tokens"] for f in fs) for c, fs in by_cat.items()}

    quota = {c: 0.0 for c in fractions}
    active = {c for c in fractions if avail.get(c, 0) > 0}
    remaining = float(budget)
    while remaining >= 1 and active:
        wsum = sum(fractions[c] for c in active)
        alloc = {c: remaining * fractions[c] / wsum for c in active}
        remaining = 0.0
        for c in list(active):
            give = min(alloc[c], avail[c] - quota[c])
            quota[c] += give
            remaining += alloc[c] - give
            if quota[c] >= avail[c] - 1e-9:
                active.discard(c)

    chosen = []
    for c, fs in by_cat.items():
        taken = 0
        for f in fs:
            if taken + f["tokens"] <= quota.get(c, 0) * 1.05:
                chosen.append(f)
                taken += f["tokens"]
    return chosen


def render(files: list[dict]) -> str:
    parts = []
    for f in sorted(files, key=lambda f: f["order"]):
        title = f"{f['project']}/{f['rel']}"
        if f["category"] == "prose":
            parts.append(f"\n\n# --- {title} ---\n\n{f['text']}\n")
        else:
            parts.append(f"\n\n# --- {title} ({f['category']}) ---\n\n```{f['category']}\n{f['text']}\n```\n")
    return "".join(parts)


def contamination(train_ids: list[int], eval_ids: list[int]) -> float:
    w = CONTAMINATION_WINDOW
    train_windows = {tuple(train_ids[i:i + w]) for i in range(len(train_ids) - w + 1)}
    eval_windows = [tuple(eval_ids[i:i + w]) for i in range(len(eval_ids) - w + 1)]
    return sum(x in train_windows for x in eval_windows) / len(eval_windows)


def composition(files: list[dict]) -> str:
    tok = Counter()
    for f in files:
        tok[f["category"]] += f["tokens"]
    total = sum(tok.values())
    return ", ".join(f"{c} {100 * t / total:.0f}%" for c, t in tok.most_common())


def main() -> None:
    tokenizer = ChiikaTokenizer()
    stats: Counter = Counter()

    files = collect(stats)
    for f in files:
        f["tokens"] = len(tokenizer.encode(f["text"]))

    train_pool, eval_pool = [], []
    for f in files:
        bucket = int(f["order"], 16) % EVAL_EVERY_N
        (eval_pool if bucket == 0 else train_pool).append(f)

    for path, project, pool in [(DATA / "cs_notes.txt", "notes", train_pool), (DATA / "eval_notes.txt", "notes", eval_pool)]:
        text = path.read_text(encoding="utf-8")
        pool.append({
            "project": project, "rel": path.name, "category": "prose", "text": text,
            "order": order_key(project, path.name), "tokens": len(tokenizer.encode(text)),
        })

    train_pool, train_lines = dedupe(train_pool, stats, "train")
    eval_pool, _ = dedupe(eval_pool, stats, "eval", reference_lines=train_lines, threshold=NEAR_DUP_EVAL_VS_TRAIN)

    train_files = select(train_pool, TRAIN_BUDGET, TRAIN_FRACTIONS)

    w = CONTAMINATION_WINDOW
    train_windows: set[tuple[int, ...]] = set()
    for f in train_files:
        ids = tokenizer.encode(f["text"])
        train_windows.update(tuple(ids[i:i + w]) for i in range(len(ids) - w + 1))
    eval_kept = []
    for f in eval_pool:
        ids = tokenizer.encode(f["text"])
        windows = [tuple(ids[i:i + w]) for i in range(len(ids) - w + 1)]
        if windows and sum(x in train_windows for x in windows) / len(windows) > WINDOW_DROP_THRESHOLD:
            stats["eval_retire_contamination_fenetres"] += 1
            continue
        eval_kept.append(f)
    eval_files = select(eval_kept, EVAL_BUDGET, TRAIN_FRACTIONS)

    # Jeu de TEST : fichiers ecartes de l'entrainement par les quotas de categories
    # (donc aucune donnee d'entrainement perdue), jamais utilises pour choisir un LR,
    # un epoch ou une architecture. Meme filtre anti-contamination, contre le train
    # ET contre l'eval.
    selected_ids = {id(f) for f in train_files}
    eval_windows: set[tuple[int, ...]] = set()
    for f in eval_files:
        ids = tokenizer.encode(f["text"])
        eval_windows.update(tuple(ids[i:i + w]) for i in range(len(ids) - w + 1))
    test_kept = []
    for f in train_pool:
        if id(f) in selected_ids:
            continue
        ids = tokenizer.encode(f["text"])
        windows = [tuple(ids[i:i + w]) for i in range(len(ids) - w + 1)]
        worst = max(sum(x in ref for x in windows) / len(windows) for ref in (train_windows, eval_windows)) if windows else 0.0
        if worst > WINDOW_DROP_THRESHOLD:
            stats["test_retire_contamination_fenetres"] += 1
            continue
        test_kept.append(f)
    test_files = select(test_kept, TEST_BUDGET, TRAIN_FRACTIONS)

    train_text, eval_text, test_text = render(train_files), render(eval_files), render(test_files)
    (DATA / "train_v3.txt").write_text(train_text, encoding="utf-8")
    (DATA / "eval_v3.txt").write_text(eval_text, encoding="utf-8")
    (DATA / "test_v3.txt").write_text(test_text, encoding="utf-8")

    train_ids, eval_ids, test_ids = tokenizer.encode(train_text), tokenizer.encode(eval_text), tokenizer.encode(test_text)
    rate = contamination(train_ids, eval_ids)
    rate_test_train = contamination(train_ids, test_ids)
    rate_test_eval = contamination(eval_ids, test_ids)

    manifest = {
        "sha1": {
            "train": hashlib.sha1(train_text.encode()).hexdigest(),
            "eval": hashlib.sha1(eval_text.encode()).hexdigest(),
            "test": hashlib.sha1(test_text.encode()).hexdigest(),
        },
        "tokenizer": "gpt2 (+ <image>)",
        "train_tokens": len(train_ids),
        "eval_tokens": len(eval_ids),
        "contamination_32_token_windows": round(rate, 4),
        "test_tokens": len(test_ids),
        "contamination_test_vs_train": round(rate_test_train, 4),
        "contamination_test_vs_eval": round(rate_test_eval, 4),
        "filters_and_dedup": dict(stats),
        "train_files": [{"file": f"{f['project']}/{f['rel']}", "category": f["category"], "tokens": f["tokens"]}
                        for f in sorted(train_files, key=lambda f: f["order"])],
        "eval_files": [{"file": f"{f['project']}/{f['rel']}", "category": f["category"], "tokens": f["tokens"]}
                       for f in sorted(eval_files, key=lambda f: f["order"])],
        "test_files": [{"file": f"{f['project']}/{f['rel']}", "category": f["category"], "tokens": f["tokens"]}
                       for f in sorted(test_files, key=lambda f: f["order"])],
    }
    (DATA / "manifest_v3.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"\nFichiers retenus apres filtres : {len(files)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\nTRAIN : {len(train_ids)} tokens, {len(train_files)} fichiers  [{composition(train_files)}]")
    print(f"EVAL  : {len(eval_ids)} tokens, {len(eval_files)} fichiers  [{composition(eval_files)}]")
    print(f"TEST  : {len(test_ids)} tokens, {len(test_files)} fichiers  [{composition(test_files)}]")
    print(f"Contamination (fenetres de {CONTAMINATION_WINDOW} tokens de l'eval presentes dans le train) : {rate:.2%}")
    print(f"Contamination test vs train : {rate_test_train:.2%} | test vs eval : {rate_test_eval:.2%}")


if __name__ == "__main__":
    main()
