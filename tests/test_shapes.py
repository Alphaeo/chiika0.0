"""Tests de specification. Ecrits a l'avance, ils servent de guide :
chaque test doit rester rouge tant que le(s) TODO correspondant(s) ne
sont pas implementes, et passer une fois que c'est fait -- pas besoin
d'ajouter des tests toi-meme pour avancer, seulement d'implementer
jusqu'a ce que ceux-la passent (tu peux bien sur en ajouter si tu veux
couvrir un cas de plus).

Lancer : python -m pytest tests/ -v
Lancer un seul test : python -m pytest tests/test_shapes.py::test_rmsnorm_shape -v
"""

import math

import torch

from chiikamini.config import ChiikaMiniConfig, TextConfig, VisionConfig
from chiikamini.layers.attention import Attention
from chiikamini.layers.block import TransformerBlock
from chiikamini.layers.mlp import FeedForward
from chiikamini.layers.norm import RMSNorm
from chiikamini.layers.rope import apply_rotary_emb, precompute_rope_freqs
from chiikamini.model import ChiikaMiniForCausalLM, ChiikaMiniVLM
from chiikamini.vision.patch_embed import PatchEmbed
from chiikamini.vision.vision_encoder import VisionEncoder

BATCH = 2
SEQ = 16
DIM = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = DIM // N_HEADS


# ---------------------------------------------------------------- norm


def test_rmsnorm_shape_and_dtype():
    norm = RMSNorm(DIM)
    x = torch.randn(BATCH, SEQ, DIM)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_rmsnorm_unit_weight_matches_manual_formula():
    """Avec weight=1 (init par defaut), RMSNorm(x) doit correspondre a
    la formule x / sqrt(mean(x^2) + eps) -- verifie qu'aucune
    normalisation supplementaire (recentrage type LayerNorm) n'a ete
    ajoutee par erreur."""
    norm = RMSNorm(DIM, eps=1e-6)
    x = torch.randn(BATCH, SEQ, DIM)
    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(x), expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- rope


def test_rope_preserves_norm():
    """Une rotation ne change pas la norme d'un vecteur -- propriete
    fondamentale de RoPE, cf. PAPERS.md section 'Position'."""
    freqs = precompute_rope_freqs(HEAD_DIM, max_seq_len=SEQ, theta=10000.0)
    x = torch.randn(BATCH, N_HEADS, SEQ, HEAD_DIM)
    rotated = apply_rotary_emb(x, freqs[:SEQ])
    assert rotated.shape == x.shape
    norm_before = x.norm(dim=-1)
    norm_after = rotated.norm(dim=-1)
    torch.testing.assert_close(norm_after, norm_before, atol=1e-3, rtol=1e-3)


def test_rope_zero_position_is_identity():
    """A la position 0, l'angle de rotation est 0 pour toutes les
    paires -- appliquer RoPE au token de position 0 ne doit rien
    changer."""
    freqs = precompute_rope_freqs(HEAD_DIM, max_seq_len=SEQ, theta=10000.0)
    x = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM)
    rotated = apply_rotary_emb(x, freqs[:1])
    torch.testing.assert_close(rotated, x, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- attention


def test_causal_attention_output_shape():
    attn = Attention(DIM, N_HEADS, N_KV_HEADS, causal=True)
    x = torch.randn(BATCH, SEQ, DIM)
    freqs = precompute_rope_freqs(HEAD_DIM, SEQ, 10000.0)
    out = attn(x, freqs[:SEQ])
    assert out.shape == (BATCH, SEQ, DIM)


def test_causal_attention_ignores_future_tokens():
    """Propriete cle du masque causal : changer un token FUTUR ne doit
    pas changer la sortie d'un token PASSE. On compare la sortie a la
    position 0 avant/apres avoir modifie le token a la position SEQ-1."""
    torch.manual_seed(0)
    attn = Attention(DIM, N_HEADS, N_KV_HEADS, causal=True)
    attn.eval()
    freqs = precompute_rope_freqs(HEAD_DIM, SEQ, 10000.0)

    x = torch.randn(BATCH, SEQ, DIM)
    out_a = attn(x, freqs[:SEQ])

    x_modified = x.clone()
    x_modified[:, -1, :] += 10.0  # perturbe seulement le dernier token
    out_b = attn(x_modified, freqs[:SEQ])

    torch.testing.assert_close(out_a[:, 0, :], out_b[:, 0, :], atol=1e-5, rtol=1e-5)


def test_bidirectional_attention_output_shape():
    attn = Attention(DIM, N_HEADS, N_KV_HEADS, causal=False)
    x = torch.randn(BATCH, SEQ, DIM)
    freqs = precompute_rope_freqs(HEAD_DIM, SEQ, 10000.0)
    out = attn(x, freqs[:SEQ])
    assert out.shape == (BATCH, SEQ, DIM)


# ---------------------------------------------------------------- ffn


def test_feedforward_swiglu_shape():
    ffn = FeedForward(DIM, hidden_dim=128, activation="swiglu")
    x = torch.randn(BATCH, SEQ, DIM)
    assert ffn(x).shape == (BATCH, SEQ, DIM)


def test_feedforward_xielu_shape():
    ffn = FeedForward(DIM, hidden_dim=128, activation="xielu")
    x = torch.randn(BATCH, SEQ, DIM)
    assert ffn(x).shape == (BATCH, SEQ, DIM)


# ---------------------------------------------------------------- block


def test_transformer_block_shape():
    block = TransformerBlock(DIM, N_HEADS, N_KV_HEADS, ffn_hidden_dim=128, causal=True)
    x = torch.randn(BATCH, SEQ, DIM)
    freqs = precompute_rope_freqs(HEAD_DIM, SEQ, 10000.0)
    out = block(x, freqs[:SEQ])
    assert out.shape == x.shape


# ---------------------------------------------------------------- vision


def test_patch_embed_shape():
    cfg = VisionConfig(image_size=32, patch_size=16, dim=DIM, n_channels=3)
    patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.n_channels, cfg.dim)
    images = torch.randn(BATCH, 3, 32, 32)
    out = patch_embed(images)
    assert out.shape == (BATCH, cfg.n_patches, DIM)  # (32/16)^2 = 4 patches


def test_vision_encoder_shape():
    cfg = VisionConfig(image_size=32, patch_size=16, dim=DIM, n_layers=2, n_heads=N_HEADS)
    encoder = VisionEncoder(cfg)
    images = torch.randn(BATCH, 3, 32, 32)
    out = encoder(images)
    assert out.shape == (BATCH, cfg.n_patches, DIM)


# ---------------------------------------------------------------- full models


def _tiny_text_config() -> TextConfig:
    return TextConfig(
        vocab_size=100,
        dim=DIM,
        n_layers=2,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        ffn_hidden_dim=128,
        max_seq_len=SEQ,
    )


def test_causal_lm_forward_shape_and_loss():
    cfg = _tiny_text_config()
    model = ChiikaMiniForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))
    labels = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))

    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (BATCH, SEQ, cfg.vocab_size)
    assert loss.ndim == 0
    assert loss.item() > 0

    logits_no_labels, loss_none = model(input_ids)
    assert logits_no_labels.shape == (BATCH, SEQ, cfg.vocab_size)
    assert loss_none is None


def test_causal_lm_loss_decreases_with_training():
    """Test de bout en bout : quelques pas de gradient sur un batch fixe
    doivent faire baisser la loss. Si ce test echoue alors que tous les
    tests unitaires ci-dessus passent, le bug est probablement dans le
    branchement (ex: mauvais ordre residuel/norm, RoPE applique deux
    fois, etc.) plutot que dans une brique individuelle."""
    torch.manual_seed(0)
    cfg = _tiny_text_config()
    model = ChiikaMiniForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))
    labels = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_vlm_forward_shape():
    text_cfg = _tiny_text_config()
    vision_cfg = VisionConfig(image_size=32, patch_size=16, dim=32, n_layers=2, n_heads=4)
    cfg = ChiikaMiniConfig(text=text_cfg, vision=vision_cfg, image_token_id=1)

    model = ChiikaMiniVLM(cfg)
    n_patches = vision_cfg.n_patches  # 4 pour image_size=32, patch_size=16
    n_text_tokens = SEQ - n_patches
    assert n_text_tokens > 0, "augmente SEQ ou reduis n_patches pour ce test"

    input_ids = torch.cat(
        [
            torch.full((BATCH, n_patches), cfg.image_token_id, dtype=torch.long),
            torch.randint(0, text_cfg.vocab_size, (BATCH, n_text_tokens)),
        ],
        dim=1,
    )
    images = torch.randn(BATCH, 3, 32, 32)

    logits, loss = model(input_ids, images=images)
    assert logits.shape == (BATCH, SEQ, text_cfg.vocab_size)
    assert loss is None


def test_initial_loss_is_close_to_uniform_prediction():
    """Un modele non entraine doit predire ~uniformement : loss ~ ln(vocab).
    Regression : avec nn.Embedding en N(0,1) partage avec lm_head, la loss
    initiale valait ~107 pour un vocabulaire de 4096 (attendu : 8.3), et les
    premiers epochs servaient juste a reparer l'initialisation."""
    torch.manual_seed(0)
    vocab = 4096
    cfg = TextConfig(vocab_size=vocab, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
                     ffn_hidden_dim=176, max_seq_len=64)
    model = ChiikaMiniForCausalLM(cfg)
    ids = torch.randint(0, vocab, (8, 64))
    labels = torch.randint(0, vocab, (8, 64))
    _, loss = model(ids, labels=labels)
    assert abs(loss.item() - math.log(vocab)) < 1.0
