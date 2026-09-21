"""Ajoute du code public du Hub au corpus d'entrainement -> data/train_v4.txt.

Source : codeparrot/github-code-clean (non verrouille, une licence par fichier).
train_v4 = train_v3 (ton propre code + notes) + un echantillon de code du Hub.
Les jeux eval_v3 et test_v3 ne changent PAS : les resultats restent comparables.

Garde-fous :
- licences PERMISSIVES uniquement (mit, apache-2.0, bsd-2/3-clause, isc, cc0-1.0,
  unlicense). Les licences copyleft (gpl, lgpl, agpl, mpl, epl...) sont exclues.
  Ces licences exigent de conserver l'attribution : data/manifest_v4.json liste, pour
  chaque fichier retenu, depot, chemin et licence.
- pas de motif de secret, pas de fichier quasi binaire (> 20 % de caracteres non ASCII)
- au plus MAX_PER_REPO fichiers par depot (diversite)
- deduplication : identiques + quasi-doublons (lignes) contre le train existant
- ANTI-CONTAMINATION : un fichier dont > 5 % des fenetres de 32 tokens existent dans
  eval_v3 ou test_v3 est retire (le code du Hub contient des copies de composants
  tres repandus, ex. shadcn/ui, qui sont aussi dans tes repos).

Telecharge un shard de ~356 Mo (mis en cache par huggingface_hub).

Usage : python scripts/fetch_hub_code.py [--shards 440] [--budget-bytes 5000000]
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from build_dataset import MAX_LINE_LEN, SECRET, dup_ratio, norm_lines, order_key, render
from chiikamini.tokenizer import ChiikaTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

DATASET = "codeparrot/github-code-clean"
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"}
CATEGORY_BY_LANGUAGE = {"C++": "cpp", "C": "cpp", "Python": "python", "JavaScript": "javascript", "TypeScript": "typescript"}
FRACTIONS = {"cpp": 0.30, "python": 0.30, "javascript": 0.20, "typescript": 0.20}
MIN_BYTES, MAX_BYTES = 300, 40_000
MAX_PER_REPO = 3
NON_ASCII_MAX = 0.20
WINDOW = 32
WINDOW_DROP = 0.05
NEAR_DUP = 0.60


def windows(ids: list[int]) -> list[tuple[int, ...]]:
    return [tuple(ids[i:i + WINDOW]) for i in range(len(ids) - WINDOW + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajoute du code public du Hub au corpus (train_v4)")
    parser.add_argument("--shards", type=int, nargs="+", default=[440])
    parser.add_argument("--budget-bytes", type=int, default=5_000_000)
    args = parser.parse_args()

    tokenizer = ChiikaTokenizer()
    eval_text = (DATA / "eval_v3.txt").read_text(encoding="utf-8")
    test_text = (DATA / "test_v3.txt").read_text(encoding="utf-8")
    train_v3_text = (DATA / "train_v3.txt").read_text(encoding="utf-8")

    reference_windows = set(windows(tokenizer.encode(eval_text))) | set(windows(tokenizer.encode(test_text)))
    seen_lines = set(norm_lines(train_v3_text))
    seen_hashes: set[str] = set()

    quota = {c: args.budget_bytes * f for c, f in FRACTIONS.items()}
    taken = Counter()
    per_repo = Counter()
    stats = Counter()
    kept: list[dict] = []

    for shard in args.shards:
        name = f"data/train-{shard:05d}-of-00880.parquet"
        print(f"telechargement/lecture : {name}", flush=True)
        path = hf_hub_download(DATASET, name, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
            rows = batch.to_pydict()
            for code, repo, fpath, lang, lic in zip(rows["code"], rows["repo_name"], rows["path"], rows["language"], rows["license"]):
                category = CATEGORY_BY_LANGUAGE.get(lang)
                if category is None:
                    continue
                # Beaucoup de fichiers GitHub ont des fins de ligne Windows (CRLF) : sans normalisation,
                # l'ecriture sous Windows donne CR CR LF, puis une ligne vide de trop apres CHAQUE ligne.
                code = code.replace("\r\n", "\n").replace("\r", "\n")
                stats["langage_retenu"] += 1
                if lic not in PERMISSIVE:
                    stats["ecarte_licence"] += 1
                    continue
                size = len(code.encode("utf-8"))
                if not MIN_BYTES <= size <= MAX_BYTES or taken[category] >= quota[category]:
                    stats["ecarte_taille_ou_quota"] += 1
                    continue
                if per_repo[repo] >= MAX_PER_REPO:
                    stats["ecarte_depot_sature"] += 1
                    continue
                if max((len(l) for l in code.splitlines()), default=0) > MAX_LINE_LEN:
                    stats["ecarte_ligne_longue"] += 1
                    continue
                if sum(ord(ch) > 127 for ch in code) / len(code) > NON_ASCII_MAX:
                    stats["ecarte_non_ascii"] += 1
                    continue
                if SECRET.search(code):
                    stats["ecarte_secret"] += 1
                    continue
                digest = hashlib.sha1(code.encode()).hexdigest()
                lines = norm_lines(code)
                if digest in seen_hashes or dup_ratio(lines, seen_lines) > NEAR_DUP:
                    stats["ecarte_doublon"] += 1
                    continue
                ids = tokenizer.encode(code)
                ws = windows(ids)
                if ws and sum(w in reference_windows for w in ws) / len(ws) > WINDOW_DROP:
                    stats["ecarte_contamination_eval_test"] += 1
                    continue
                seen_hashes.add(digest)
                seen_lines.update(lines)
                per_repo[repo] += 1
                taken[category] += size
                rel = f"{repo}/{fpath}"
                kept.append({"project": "hub", "rel": rel, "category": category, "text": code,
                             "order": order_key("hub", rel), "license": lic, "language": lang,
                             "repo": repo, "path": fpath, "bytes": size, "sha1": digest})
            if all(taken[c] >= quota[c] for c in quota):
                break
        if all(taken[c] >= quota[c] for c in quota):
            break

    hub_text = render(kept)
    (DATA / "hub_code_v1.txt").write_text(hub_text, encoding="utf-8", newline="\n")
    train_v4 = train_v3_text + hub_text
    (DATA / "train_v4.txt").write_text(train_v4, encoding="utf-8", newline="\n")

    manifest = {
        "note": "train_v4 = train_v3 + code du Hub ; eval_v3 et test_v3 inchanges",
        "source": DATASET,
        "shards": args.shards,
        "permissive_licenses": sorted(PERMISSIVE),
        "sha1": {
            "train": hashlib.sha1(train_v4.encode()).hexdigest(),
            "eval": hashlib.sha1(eval_text.encode()).hexdigest(),
            "test": hashlib.sha1(test_text.encode()).hexdigest(),
        },
        "filters": dict(stats),
        "hub_bytes_par_categorie": dict(taken),
        "hub_licences": dict(Counter(f["license"] for f in kept)),
        "hub_fichiers": [{"repo": f["repo"], "path": f["path"], "license": f["license"], "language": f["language"],
                          "bytes": f["bytes"], "sha1": f["sha1"]} for f in sorted(kept, key=lambda f: f["order"])],
    }
    (DATA / "manifest_v4.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    hub_tokens = len(tokenizer.encode(hub_text))
    print(f"\nFiltres : {dict(stats)}")
    print(f"HUB : {len(kept)} fichiers, {len({f['repo'] for f in kept})} depots, {sum(taken.values()) / 1e6:.2f} Mo, {hub_tokens} tokens GPT-2")
    print(f"  par categorie (Mo) : { {c: round(v / 1e6, 2) for c, v in taken.items()} }  (quotas : { {c: round(v / 1e6, 2) for c, v in quota.items()} })")
    print(f"  licences : {manifest['hub_licences']}")
    print(f"TRAIN_v4 : {len(tokenizer.encode(train_v4))} tokens GPT-2 (train_v3 : {len(tokenizer.encode(train_v3_text))})")


if __name__ == "__main__":
    main()
