"""Rotary Position Embeddings (Su et al., 2021, RoFormer).
Voir PAPERS.md section "Position".

Idee generale : on regroupe les dimensions de Q/K par paires (x1, x2),
(x3, x4), ... et on fait tourner chaque paire dans le plan 2D d'un angle
qui depend de la position du token et de l'indice de la paire. Une
rotation ne change pas la norme du vecteur -- c'est une propriete a
verifier dans les tests.
"""

import torch


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precalcule les angles de rotation pour toutes les positions et
    toutes les paires de dimensions.

    return: tensor complexe (ou tensor reel (max_seq_len, head_dim/2, 2)
    si tu preferes eviter torch.complex) contenant, pour chaque position
    `pos` et chaque paire d'indice `i` (i in [0, head_dim/2)) :
        angle = pos / (theta ** (2*i / head_dim))
    stocke sous forme (cos(angle), sin(angle)) ou cis(angle) = e^{i*angle}.

    TODO:
    1. freqs_i = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))  -- shape (head_dim/2,)
    2. positions = arange(max_seq_len)  -- shape (max_seq_len,)
    3. angles = outer(positions, freqs_i)  -- shape (max_seq_len, head_dim/2)
    4. Retourner torch.polar(torch.ones_like(angles), angles) (forme
       complexe, pratique pour l'etape suivante) -- ou (cos(angles), sin(angles))
       si tu evites les tensors complexes.
    """
    freqs_i = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))  # shape (head_dim/2,)
    positions = torch.arange(max_seq_len)  # shape (max_seq_len,)
    angles = torch.outer(positions, freqs_i)  # shape (max_seq_len, head_dim/2)
    return torch.polar(torch.ones_like(angles), angles)  # forme complexe


def apply_rotary_emb(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """Applique la rotation RoPE a Q ou K.

    x: (batch, n_heads, seq, head_dim) -- reel
    rope_freqs: sortie de `precompute_rope_freqs`, tronquee a `seq`
                (shape (seq, head_dim/2))
    return: (batch, n_heads, seq, head_dim), meme shape que x, norme
            de chaque vecteur de tete inchangee par rapport a x.

    TODO (approche avec nombres complexes, la plus courte a ecrire) :
    1. Reinterpreter x comme des paires (x1, x2), (x3, x4), ... =
       x.reshape(*x.shape[:-1], -1, 2), puis les voir comme des
       nombres complexes x1 + i*x2 (torch.view_as_complex, attention
       au dtype: il faut un tensor contigu de dtype float mappable en
       complex64, donc x.float() avant).
    2. Multiplier terme a terme par rope_freqs[:seq] (broadcast sur
       batch et n_heads) -- une multiplication complexe = une rotation.
    3. Repasser en reel (torch.view_as_real) et reshape a la shape
       d'origine (..., head_dim). Reconvertir au dtype d'entree.

    Test attendu : ||apply_rotary_emb(x, freqs)|| == ||x|| (a epsilon
    pres) pour chaque vecteur de tete -- une rotation preserve la norme.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))  # shape (batch, n_heads, seq, head_dim/2)
    rope_freqs_complex = rope_freqs.reshape(1,1,*rope_freqs.shape)  # shape (1, 1, seq, head_dim/2)
    x_rotated_complex = x_complex * rope_freqs_complex  # multiplication complexe
    x_rotated = torch.view_as_real(x_rotated_complex).reshape(*x.shape)
    return x_rotated.to(x.dtype)  # reconvertir au dtype d'entree
