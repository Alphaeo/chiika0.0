"""Self-attention avec Grouped Query Attention (GQA), RoPE et QK-Norm
optionnelle. Voir PAPERS.md sections "Attention" et "Position".

Un seul module `Attention` sert pour le decoder texte (causal=True) et
l'encodeur vision (causal=False) -- seul le masque change.
"""

import torch
from torch import nn

from chiikamini.layers.norm import QKNorm
from chiikamini.layers.rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads   # combien de tetes Q partagent chaque tete KV
        self.head_dim = dim // n_heads
        self.causal = causal

        # TODO: projections lineaires, SANS biais (cf. Qwen3 -- le
        # biais QKV est retire au profit de QK-Norm pour la stabilite).
        # - wq : dim -> n_heads * head_dim
        # - wk, wv : dim -> n_kv_heads * head_dim
        # - wo : n_heads * head_dim -> dim (projection de sortie)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.qk_norm = QKNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, dim)
        rope_freqs: sortie de `precompute_rope_freqs`, tronquee a `seq`
        return: (batch, seq, dim)

        TODO, dans l'ordre :
        1. Projeter x en q, k, v avec wq/wk/wv.
        2. Reshape en (batch, seq, n_heads_ou_n_kv_heads, head_dim) puis
           transpose en (batch, n_heads_ou_n_kv_heads, seq, head_dim).
        3. Si self.qk_norm is not None: appliquer QKNorm a (q, k).
        4. Appliquer apply_rotary_emb a q et k (PAS a v -- RoPE encode
           la position pour le produit scalaire Q.K, v n'en a pas besoin).
        5. GQA : repeter k et v `self.n_rep` fois sur la dimension des
           tetes pour matcher n_heads (torch.repeat_interleave sur la
           dim des tetes, ou k.unsqueeze + expand + reshape -- les deux
           marchent, repeat_interleave est plus lisible).
        6. Attention : scores = q @ k.transpose(-2, -1) / sqrt(head_dim).
           Si self.causal: appliquer un masque triangulaire superieur a
           -inf avant le softmax (un token ne voit pas le futur).
           Sinon (encodeur vision) : pas de masque, attention complete.
        7. softmax(scores, dim=-1) @ v -> (batch, n_heads, seq, head_dim).
        8. Transpose + reshape pour revenir a (batch, seq, dim), puis
           projeter avec wo.

        Note perf (optionnelle, a faire une fois que ca marche) :
        `torch.nn.functional.scaled_dot_product_attention` fait les
        etapes 6-7 en un seul appel, avec un kernel fusionne (Flash
        Attention si dispo) -- interessant pour les vitesses d'inference
        une fois la version "manuelle" validee par les tests.
        """
        batch, seq, _ = x.shape

        # 1. Projeter x en q, k, v
        q = self.wq(x)  # (batch, seq, n_heads * head_dim)
        k = self.wk(x)  # (batch, seq, n_kv_heads * head_dim)
        v = self.wv(x)  # (batch, seq, n_kv_heads * head_dim)

        # 2. Reshape et transpose
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (batch, n_kv_heads, seq, head_dim)
        v = v.view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (batch, n_kv_heads, seq, head_dim)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        q = apply_rotary_emb(q, rope_freqs)
        k = apply_rotary_emb(k, rope_freqs)

        # 5. GQA : repetition de k et v
        k = k.repeat_interleave(self.n_rep, dim=1)  # (batch, n_heads, seq, head_dim)
        v = v.repeat_interleave(self.n_rep, dim=1)  # (batch, n_heads, seq, head_dim)

        # 6. Attention : calcul des scores
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        if self.causal:
            mask = torch.triu(torch.ones(seq, seq), diagonal=1).to(scores.device)
            scores = scores.masked_fill(mask == 1, float('-inf'))

        # 7. Softmax et application a v
        attn = scores.softmax(dim=-1) @ v

        # 8. Transpose + reshape
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(attn)


class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, 2024) -- voir PAPERS.md,
    section "Attention" (optionnel). Compresse K et V dans un vecteur
    latent de rang faible avant mise en cache, au lieu de garder les
    K/V complets par tete comme GQA. Interessant pour reduire le
    KV-cache si tu pousses `generate.py` vers des sequences longues,
    mais PAS necessaire pour la v1 du modele -- phase 2, une fois que
    `Attention` (GQA) tourne et passe les tests.

    Non demarre : pas de TODO detaille ici tant que la version GQA
    n'est pas validee. Si tu veux t'y attaquer, relire la section 2.1
    ("Multi-Head Latent Attention") de https://arxiv.org/abs/2405.04434
    d'abord.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Phase 2 -- voir docstring de la classe.")
