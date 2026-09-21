"""Entraine un ChiikaMiniForCausalLM (xielu, retenu par defaut suite a
scripts/compare_activations.py) sur l'integralite de data/cs_code_corpus.txt,
sauvegarde le checkpoint, et le publie sur le Hugging Face Hub.

Usage : python scripts/train_and_push.py
"""

import sys
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chiikamini.checkpoint import push_to_hub, save_checkpoint
from chiikamini.config import TextConfig
from chiikamini.data import ToyTextDataset
from chiikamini.generate import generate
from chiikamini.model import ChiikaMiniForCausalLM
from chiikamini.tokenizer import ChiikaTokenizer
from chiikamini.train import train_loop

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "cs_code_corpus.txt"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "chiikamini-xielu"
HF_REPO_ID = "Crocolil/Chiikamini"

SEQ_LEN = 64
N_EPOCHS = 40

MODEL_CARD = """---
license: mit
tags:
- chiikamini
- toy-model
- xielu
- from-scratch
---

# chiikamini (xielu)

Tout petit modele de langage (decoder-only, ~7M parametres) implemente
from scratch en PyTorch, pour tester des applications locales
(chiikaScreen) sans charger un vrai LLM en RAM pendant le dev. **Pas un
modele destine a un usage reel** -- entraine sur un corpus jouet de
~27k tokens (notes CS + extraits de code de chiikaml/chiikamini).

Architecture : RoPE, Grouped Query Attention, RMSNorm + QK-Norm,
FeedForward avec activation **xIELU** (Huang et al. 2024,
arXiv:2411.13010) plutot que SwiGLU -- voir
[compare_activations.py](https://github.com/Alphaeo/chiika0.0/blob/master/scripts/compare_activations.py)
pour la comparaison qui a motive ce choix.

Code source : https://github.com/Alphaeo/chiika0.0
"""


def main() -> None:
    sys.exit(
        "OBSOLETE : ce script reentraine l'ancienne recette (27k tokens, tokenizer GPT-2) puis la publie sur "
        f"{HF_REPO_ID}, ce qui ECRASERAIT le modele actuel.\n"
        "Utiliser : python scripts/train_v2.py --report-test --out <dossier>   puis   "
        "python scripts/push_model.py --checkpoint <dossier>"
    )
    tokenizer = ChiikaTokenizer()
    dataset = ToyTextDataset(CORPUS_PATH, tokenizer, seq_len=SEQ_LEN)

    cfg = TextConfig(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=176,
        max_seq_len=SEQ_LEN,
        ffn_activation="xielu",
    )
    model = ChiikaMiniForCausalLM(cfg)

    train_loop(model, dataset, n_epochs=N_EPOCHS, batch_size=8, lr=3e-4, warmup_steps=30)

    model.eval()
    for prompt in ["A binary search tree", "class Matrix", "Gradient descent"]:
        ids = torch.tensor([tokenizer.encode(prompt)])
        out = generate(model, ids, images=None, max_new_tokens=25, temperature=0)
        print(f"{prompt!r} -> {tokenizer.decode(out[0].tolist())!r}")

    save_checkpoint(model, cfg, CHECKPOINT_DIR)
    tokenizer.tokenizer.save(str(CHECKPOINT_DIR / "tokenizer.json"))
    (CHECKPOINT_DIR / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    print(f"Checkpoint sauvegarde dans {CHECKPOINT_DIR}")

    url = push_to_hub(CHECKPOINT_DIR, HF_REPO_ID, private=True)
    print(f"Pousse sur le Hub : {url}")


if __name__ == "__main__":
    main()
