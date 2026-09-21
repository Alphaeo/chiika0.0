"""Tests de specification pour TODO.md (point 4) : dropout residuel et
fenetres a offset aleatoire. Rouges tant que ce n'est pas implemente.

Lancer : python -m pytest tests/test_regularization_and_windows.py -v
"""

import pytest
import torch

from chiikamini.config import TextConfig
from chiikamini.data import RandomWindowTextDataset
from chiikamini.layers.block import TransformerBlock
from chiikamini.layers.rope import precompute_rope_freqs
from chiikamini.model import ChiikaMiniForCausalLM
from chiikamini.train import train_loop

N_TOKENS = 500
SEQ = 16
WINDOWS = 64


class CountingTokenizer:
    """encode() renvoie [0, 1, ..., n-1] : input_ids[0] d'une fenetre est alors
    exactement son offset de depart dans le corpus, donc facile a verifier."""

    def __init__(self, n: int) -> None:
        self.n = n

    def encode(self, text: str) -> list[int]:
        return list(range(self.n))


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("le contenu est ignore par CountingTokenizer", encoding="utf-8")
    return path


def make_ds(corpus, n_tokens: int = N_TOKENS, seed: int = 0) -> RandomWindowTextDataset:
    return RandomWindowTextDataset(corpus, CountingTokenizer(n_tokens), seq_len=SEQ,
                                   windows_per_epoch=WINDOWS, seed=seed)


def starts(ds) -> list[int]:
    return [ds[i][0][0].item() for i in range(len(ds))]


# ------------------------------------------------- RandomWindowTextDataset


def test_len_is_windows_per_epoch(corpus):
    assert len(make_ds(corpus)) == WINDOWS


def test_item_shapes_and_label_shift(corpus):
    x, y = make_ds(corpus)[3]
    assert x.shape == (SEQ,) and y.shape == (SEQ,)
    assert torch.equal(x[1:], y[:-1]), "labels doivent etre input_ids decale d'un token"
    assert y[-1].item() == x[-1].item() + 1


def test_every_window_fits_in_corpus(corpus):
    ds = make_ds(corpus, n_tokens=SEQ + 5)
    for i in range(len(ds)):
        x, y = ds[i]
        assert x[0].item() >= 0
        assert y[-1].item() <= SEQ + 4, "fenetre qui depasse la fin du corpus"


def test_same_seed_epoch_idx_gives_same_window(corpus):
    a, b = make_ds(corpus, seed=1), make_ds(corpus, seed=1)
    a.set_epoch(2)
    b.set_epoch(2)
    assert starts(a) == starts(b)


def test_epoch_changes_the_windows(corpus):
    ds = make_ds(corpus)
    epoch0 = starts(ds)
    ds.set_epoch(1)
    assert starts(ds) != epoch0


def test_offsets_are_not_aligned_on_seq_len(corpus):
    """Des fenetres alignees sur des multiples de seq_len, c'est le comportement
    de ToyTextDataset -- ici on veut des offsets libres."""
    assert any(s % SEQ != 0 for s in starts(make_ds(corpus)))


def test_too_short_corpus_raises(corpus):
    with pytest.raises(ValueError):
        make_ds(corpus, n_tokens=SEQ)  # il faut au moins seq_len + 1 tokens


def test_train_loop_calls_set_epoch_every_epoch(corpus):
    calls = []

    class Logging(RandomWindowTextDataset):
        def set_epoch(self, epoch):
            calls.append(epoch)
            super().set_epoch(epoch)

    ds = Logging(corpus, CountingTokenizer(N_TOKENS), seq_len=SEQ, windows_per_epoch=16)
    cfg = TextConfig(vocab_size=N_TOKENS, dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
                     ffn_hidden_dim=64, max_seq_len=SEQ)
    train_loop(ChiikaMiniForCausalLM(cfg), ds, n_epochs=3, batch_size=8, warmup_steps=1)
    assert calls == [0, 1, 2]


# ------------------------------------------------- dropout residuel

DIM, HEADS, KV_HEADS, T = 32, 4, 2, 8


def make_block(dropout: float) -> TransformerBlock:
    return TransformerBlock(DIM, HEADS, KV_HEADS, ffn_hidden_dim=64, dropout=dropout)


def run_twice(module, *args):
    return module(*args), module(*args)


@pytest.fixture
def block_inputs():
    torch.manual_seed(0)
    return torch.randn(2, T, DIM), precompute_rope_freqs(DIM // HEADS, T)


def test_dropout_zero_is_deterministic_even_in_train_mode(block_inputs):
    block = make_block(0.0).train()
    a, b = run_twice(block, *block_inputs)
    torch.testing.assert_close(a, b)


def test_block_dropout_is_active_in_train_mode(block_inputs):
    block = make_block(0.5).train()
    a, b = run_twice(block, *block_inputs)
    assert not torch.allclose(a, b), "avec dropout=0.5 en train(), deux passes doivent differer"


def test_block_dropout_is_inactive_in_eval_mode(block_inputs):
    block = make_block(0.5).eval()
    a, b = run_twice(block, *block_inputs)
    torch.testing.assert_close(a, b)


def test_dropout_flows_from_config_to_model():
    cfg = TextConfig(vocab_size=100, dim=DIM, n_layers=2, n_heads=HEADS, n_kv_heads=KV_HEADS,
                     ffn_hidden_dim=64, max_seq_len=16, dropout=0.3)
    model = ChiikaMiniForCausalLM(cfg)
    ids = torch.randint(0, 100, (2, 8))

    model.train()
    (a, _), (b, _) = model(ids), model(ids)
    assert not torch.allclose(a, b)

    model.eval()
    (a, _), (b, _) = model(ids), model(ids)
    torch.testing.assert_close(a, b)
