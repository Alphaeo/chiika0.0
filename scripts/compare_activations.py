"""Compare swiglu vs xielu sur le meme corpus, meme seed, meme nombre de
steps -- pour repondre objectivement (loss sur donnees NON vues) plutot
qu'a l'oeil sur une seule generation.

Usage : python scripts/compare_activations.py
"""

import math
import sys
from pathlib import Path

# `python scripts/x.py` met scripts/ dans sys.path, pas la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from chiikamini.config import TextConfig
from chiikamini.data import ToyTextDataset
from chiikamini.generate import generate
from chiikamini.model import ChiikaMiniForCausalLM
from chiikamini.tokenizer import ChiikaTokenizer
from chiikamini.train import train_loop
from chiikamini.utils import get_device, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "cs_code_corpus.txt"
SEQ_LEN = 64
TRAIN_FRACTION = 0.9
N_EPOCHS = 15


def split_corpus() -> tuple[Path, Path]:
    text = CORPUS_PATH.read_text(encoding="utf-8")
    split_idx = int(len(text) * TRAIN_FRACTION)
    train_path = REPO_ROOT / "data" / "_tmp_train.txt"
    val_path = REPO_ROOT / "data" / "_tmp_val.txt"
    train_path.write_text(text[:split_idx], encoding="utf-8")
    val_path.write_text(text[split_idx:], encoding="utf-8")
    return train_path, val_path


@torch.no_grad()
def held_out_loss(model: ChiikaMiniForCausalLM, dataset, device) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=8)
    total_loss, total_tokens = 0.0, 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        _, loss = model(input_ids, labels=labels)
        n = input_ids.numel()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / total_tokens


def run(activation: str, tokenizer, train_ds, val_ds, device):
    set_seed(0)
    cfg = TextConfig(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=176,
        max_seq_len=SEQ_LEN,
        ffn_activation=activation,
    )
    model = ChiikaMiniForCausalLM(cfg)
    train_loop(model, train_ds, n_epochs=N_EPOCHS, batch_size=8, lr=3e-4, warmup_steps=20)

    val_loss = held_out_loss(model, val_ds, device)
    ppl = math.exp(val_loss)

    prompts = ["A binary search tree", "class Matrix", "def forward(self"]
    model.eval()
    completions = []
    for p in prompts:
        ids = torch.tensor([tokenizer.encode(p)]).to(device)
        out = generate(model, ids, images=None, max_new_tokens=20, temperature=0)
        completions.append(tokenizer.decode(out[0].tolist()))

    return val_loss, ppl, completions


def main() -> None:
    tokenizer = ChiikaTokenizer()
    train_path, val_path = split_corpus()
    train_ds = ToyTextDataset(train_path, tokenizer, seq_len=SEQ_LEN)
    val_ds = ToyTextDataset(val_path, tokenizer, seq_len=SEQ_LEN)
    device = get_device()

    results = {}
    for activation in ["swiglu", "xielu"]:
        print(f"\n=== {activation} ===")
        val_loss, ppl, completions = run(activation, tokenizer, train_ds, val_ds, device)
        results[activation] = (val_loss, ppl, completions)
        print(f"held-out loss: {val_loss:.4f}  perplexity: {ppl:.2f}")
        for c in completions:
            print("  ->", repr(c))

    print("\n=== Resume ===")
    for activation, (val_loss, ppl, _) in results.items():
        print(f"{activation:8s} held-out loss={val_loss:.4f}  perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
