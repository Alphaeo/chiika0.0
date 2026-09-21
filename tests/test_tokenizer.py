"""Tests de specification pour TODO.md (point 5) : tokenizer BPE de domaine
et bits par octet. Rouges tant que ce n'est pas implemente.

Lancer : python -m pytest tests/test_tokenizer.py -v
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from chiikamini.evals import bits_per_byte
from chiikamini.tokenizer import ChiikaTokenizer, train_bpe_tokenizer

DATA = Path(__file__).resolve().parent.parent / "data"

CORPUS = (
    "def forward(self, x):\n        return x + 1\n" * 200
    + "for (std::size_t i = 0; i < n; ++i) {\n        sum += a[i];\n    }\n" * 200
)


@pytest.fixture(scope="module")
def tok():
    return train_bpe_tokenizer([CORPUS], vocab_size=320, special_tokens=["<image>"])


# ------------------------------------------------- train_bpe_tokenizer


def test_special_token_gets_id_zero_and_encodes_as_one_id(tok):
    assert tok.token_to_id("<image>") == 0
    assert tok.encode("<image>").ids == [0]


def test_vocab_size_is_within_target(tok):
    assert 256 < tok.get_vocab_size() <= 320


@pytest.mark.parametrize("text", [
    "def forward(self, x):\n        return x\n",
    "café ☕ 日本語 \U0001f642",   # accents, symbole, CJK, emoji : jamais vus a l'entrainement
    "   \t\n\n  ",
])
def test_roundtrip_is_exact_even_for_unseen_characters(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_every_id_is_inside_the_vocabulary(tok):
    ids = tok.encode("日本語 \U0001f642 zzzz").ids
    assert ids and all(0 <= i < tok.get_vocab_size() for i in ids)


def test_it_learned_domain_merges(tok):
    assert len(tok.encode("std::size_t").ids) < len("std::size_t")
    assert len(tok.encode("        ").ids) <= 2, "8 espaces d'indentation ne doivent pas couter 8 tokens"


# ------------------------------------------------- ChiikaTokenizer.from_file


def test_chiikatokenizer_loads_a_saved_tokenizer_file(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    loaded = ChiikaTokenizer(tokenizer_path=path)
    assert loaded.vocab_size == tok.get_vocab_size()
    assert loaded.image_token_id == 0
    text = "for (std::size_t i = 0; i < n; ++i) {"
    assert loaded.decode(loaded.encode(text)) == text


# ------------------------------------------------- comparaison avec GPT-2 (donnees reelles)


def test_domain_tokenizer_is_more_compact_than_gpt2_on_held_out_code():
    train_path, eval_path = DATA / "train_v3.txt", DATA / "eval_v3.txt"
    if not (train_path.exists() and eval_path.exists()):
        pytest.skip("data/train_v3.txt / eval_v3.txt absents : lancer scripts/build_dataset.py")
    try:
        gpt2 = ChiikaTokenizer()
    except Exception as e:  # pas de cache/reseau pour telecharger GPT-2
        pytest.skip(f"tokenizer GPT-2 indisponible : {e}")

    ours = train_bpe_tokenizer([train_path.read_text(encoding="utf-8")], vocab_size=4096, special_tokens=["<image>"])
    held_out = eval_path.read_text(encoding="utf-8")
    n_bytes = len(held_out.encode("utf-8"))
    ours_bytes_per_token = n_bytes / len(ours.encode(held_out).ids)
    gpt2_bytes_per_token = n_bytes / len(gpt2.encode(held_out))
    assert ours_bytes_per_token > gpt2_bytes_per_token


# ------------------------------------------------- bits_per_byte


class UniformModel(torch.nn.Module):
    """Modele idiot : repartit la probabilite uniformement sur `vocab` tokens,
    donc loss = ln(vocab) par token, quelle que soit l'entree."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, labels=None):
        logits = torch.zeros(*input_ids.shape, self.vocab)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab), labels.reshape(-1)) if labels is not None else None
        return logits, loss


class OneTokenPerByte:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class OneTokenPerTwoBytes:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8")[::2])


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("abcdefghij" * 100, encoding="utf-8")   # 1000 octets, nombre pair
    return path


def test_bpb_of_uniform_model_over_bytes_is_log2_of_vocab(text_file):
    """256 choix equiprobables par octet = exactement 8 bits par octet."""
    bpb = bits_per_byte(UniformModel(256), OneTokenPerByte(), text_file, seq_len=16, batch_size=4)
    assert bpb == pytest.approx(8.0, abs=1e-4)


def test_bpb_accounts_for_bytes_per_token(text_file):
    """Un token = 2 octets : la meme incertitude par token vaut 2x moins par octet."""
    bpb = bits_per_byte(UniformModel(256), OneTokenPerTwoBytes(), text_file, seq_len=16, batch_size=4)
    assert bpb == pytest.approx(4.0, abs=1e-4)


def test_bpb_matches_the_formula_for_another_vocab(text_file):
    bpb = bits_per_byte(UniformModel(1000), OneTokenPerByte(), text_file, seq_len=16, batch_size=4)
    assert bpb == pytest.approx(math.log2(1000), abs=1e-4)
