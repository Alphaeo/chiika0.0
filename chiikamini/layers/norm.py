"""RMSNorm (Zhang & Sennrich, 2019) et QK-Norm (Qwen3).
Voir PAPERS.md section "Normalisation" pour le detail.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm : normalise par la racine de la moyenne des carres, sans
    recentrer (contrairement a LayerNorm qui soustrait la moyenne).

    Formule : y = x / sqrt(mean(x**2, dim=-1) + eps) * weight

    `weight` est un parametre appris de shape (dim,), initialise a 1.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim)
        return: (..., dim), meme shape/dtype que x.

        TODO:
        1. Calculer la RMS sur la derniere dimension (garder la dim avec
           keepdim=True pour le broadcasting).
        2. Diviser x par (rms + eps) -- attention, additionner eps
           *avant* la racine carree ou apres, choisis une convention et
           reste coherent avec le test (`tests/test_shapes.py`).
        3. Multiplier par `self.weight`.

        Piege frequent : fais le calcul de variance en float32 meme si x
        est en float16/bfloat16 (x.float()), puis reconverti a la fin --
        RMSNorm est numeriquement fragile en basse precision.
        """
        raise NotImplementedError


class QKNorm(nn.Module):
    """QK-Norm (Qwen3) : normalise Q et K independamment, par tete,
    avant le produit scalaire d'attention. Stabilise l'entrainement
    quand on retire le biais QKV. C'est juste une RMSNorm appliquee sur
    la derniere dimension (head_dim) de Q et de K separement -- pas de
    nouvelle formule, reutilise `RMSNorm` en interne.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, n_heads, seq, head_dim)
        k: (batch, n_kv_heads, seq, head_dim)
        return: (q_normed, k_normed), memes shapes.

        TODO: appliquer self.q_norm sur q et self.k_norm sur k (la RMSNorm
        agit deja sur la derniere dimension, donc rien de special a faire
        pour le multi-tete -- juste appeler les deux sous-modules).
        """
        raise NotImplementedError
