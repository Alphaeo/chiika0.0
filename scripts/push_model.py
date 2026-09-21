"""Publie un checkpoint entraine par scripts/train_v2.py sur le Hugging Face Hub.

- Genere README.md (model card) a partir de history.json : chiffres reels du run,
  jamais saisis a la main.
- Joint la provenance des donnees (data/manifest_<version>.json -> training_data_manifest.json),
  qui liste depot, chemin et licence de chaque fichier de code public utilise.
- Visibilite demandee a la creation : PRIVE par defaut (--public pour changer) ; la visibilite
  d'un depot deja existant n'est jamais modifiee, la valeur reelle est affichee a la fin. Pas de token en argument : utilise la
  session deja connectee (`hf auth login` ou HF_TOKEN).

Usage :
    python scripts/push_model.py --checkpoint checkpoints/chiikamini-v4-final --dry-run
    python scripts/push_model.py --checkpoint checkpoints/chiikamini-v4-final
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi

from chiikamini.checkpoint import push_to_hub

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"


def build_model_card(history: dict, manifest: dict | None) -> str:
    cfg = history["config"]
    epochs = [e for e in history["epochs"] if "eval_ppl" in e]
    best = min(epochs, key=lambda e: e["eval_ppl"])
    bpb = history["bits_per_byte"]
    ref = history["reference_bits_per_byte"]
    n_params = history.get("n_parameters")
    params_txt = f"{n_params / 1e6:.2f} M parametres" if n_params else "quelques millions de parametres"

    rows = "\n".join(
        f"| {split} | {value:.3f} | {ref[split]:.3f} | {value - ref[split]:+.3f} ({100 * (value / ref[split] - 1):+.0f} %) |"
        for split, value in bpb.items()
    )
    if manifest and "hub_licences" in manifest:
        lic = ", ".join(f"{k} ({v})" for k, v in sorted(manifest["hub_licences"].items(), key=lambda kv: -kv[1]))
        hub = (f"- **Code public de GitHub** : {len(manifest['hub_fichiers'])} fichiers "
               f"({sum(manifest['hub_bytes_par_categorie'].values()) / 1e6:.1f} Mo) tires de "
               f"[{manifest['source']}](https://huggingface.co/datasets/{manifest['source']}), "
               f"**licences permissives uniquement** : {lic}. Provenance fichier par fichier "
               "(depot, chemin, licence) dans `training_data_manifest.json`.")
    else:
        hub = "- (pas de code public externe dans ce run)"

    return f"""---
license: mit
tags:
- chiikamini
- toy-model
- from-scratch
- xielu
- code
---

# chiikamini

Petit modele de langage **decoder-only** ({params_txt}) ecrit from scratch en PyTorch, pour tester
des applications locales sans charger un vrai LLM en RAM. **Ce n'est pas un modele destine a un
usage reel** : il complete du code de facon plausible mais ne raisonne pas.

## Architecture
RoPE, Grouped Query Attention ({cfg['n_heads']} tetes Q / {cfg['n_kv_heads']} tetes KV), RMSNorm + QK-Norm,
FFN avec activation **xIELU** (Huang et al. 2024, arXiv:2411.13010) ; {cfg['n_layers']} couches, dim {cfg['dim']},
contexte {cfg['max_seq_len']} tokens ; embeddings d'entree et de sortie partages ; tokenizer BPE byte-level
de {cfg['vocab_size']} tokens entraine sur le corpus d'entrainement.

## Donnees d'entrainement
- **Code de l'auteur** (chiikaml en C++, insta-trend-ai, diamant) et notes de cours d'informatique.
{hub}

## Resultats
Bits par octet de texte (plus bas = mieux), sur des fichiers jamais vus a l'entrainement. Meilleur epoch
choisi sur `eval` ({best['epoch']}, perplexite {best['eval_ppl']:.1f} par token) ; `test` n'a ete mesure qu'une fois.

| jeu | ce modele | modele precedent (GPT-2, 27k tokens) | ecart |
|---|---|---|---|
{rows}

**Limites** : un seul run, une seule graine ; les jeux eval/test sont des fichiers de l'auteur, presque
uniquement du C++ et du TypeScript (tres peu de Python) ; l'amelioration cumule plus de donnees, un tokenizer de
domaine, une correction d'initialisation, un reglage du LR et un contexte de {cfg['max_seq_len']} tokens (64 pour
les versions precedentes), sans ablation pour les separer.

## Utilisation
L'architecture est custom : elle ne se charge pas avec `transformers`. Voir
[le depot du code](https://github.com/Alphaeo/chiika0.0) (`chiikamini.checkpoint.load_checkpoint`,
`scripts/chat.py --checkpoint <dossier>`). Le tokenizer est dans `tokenizer.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publie un checkpoint chiikamini sur le Hub")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", default="Crocolil/Chiikamini")
    parser.add_argument("--public", action="store_true", help="depot public (defaut : prive)")
    parser.add_argument("--dry-run", action="store_true", help="ecrit le model card sans rien envoyer")
    args = parser.parse_args()

    history = json.loads((args.checkpoint / "history.json").read_text(encoding="utf-8"))
    manifest = None
    tag = Path(history.get("train_file", "")).stem.split("_")[-1]          # train_v5.txt -> v5
    manifest_path = DATA / f"manifest_{tag}.json"
    if tag != "v3" and manifest_path.exists():                              # v3 = ton code seul, pas de code du Hub
        shutil.copyfile(manifest_path, args.checkpoint / "training_data_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    card = build_model_card(history, manifest)
    (args.checkpoint / "README.md").write_text(card, encoding="utf-8")
    print(card)

    if args.dry_run:
        print("\n[dry-run] rien n'a ete envoye.")
        return
    url = push_to_hub(args.checkpoint, args.repo_id, private=not args.public)
    visibility = "PRIVE" if HfApi().repo_info(args.repo_id).private else "PUBLIC"
    print(f"\nPublie : {url}  (visibilite reelle sur le Hub : {visibility})")


if __name__ == "__main__":
    main()
