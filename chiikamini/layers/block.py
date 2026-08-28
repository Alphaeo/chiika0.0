"""Bloc Transformer pre-norm : residual + norm + attention, puis
residual + norm + FFN. "Pre-norm" (norme avant la sous-couche, pas
apres) est le choix standard post-GPT-2 -- stabilise l'entrainement de
modeles profonds en gardant un chemin residuel "propre" (sans norme
dessus), cf. discussions autour de Llama/Qwen si tu veux creuser le
pourquoi.
"""

import torch
from torch import nn

from chiikamini.layers.attention import Attention
from chiikamini.layers.mlp import FeedForward
from chiikamini.layers.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_hidden_dim: int,
        norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        ffn_activation: str = "swiglu",
        causal: bool = True,
    ) -> None:
        super().__init__()
        # TODO: instancier
        # - self.attn_norm = RMSNorm(dim, eps=norm_eps)
        # - self.attn = Attention(dim, n_heads, n_kv_heads, norm_eps, use_qk_norm, causal)
        # - self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        # - self.ffn = FeedForward(dim, ffn_hidden_dim, ffn_activation)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, dim)
        rope_freqs: passe tel quel a Attention
        return: (batch, seq, dim)

        TODO:
            x = x + self.attn(self.attn_norm(x), rope_freqs)
            x = x + self.ffn(self.ffn_norm(x))
            return x

        (Note l'ordre : norm PUIS sous-couche, et le residuel se fait
        sur x d'ORIGINE, pas sur la sortie normee -- c'est ca, "pre-norm".)
        """
        raise NotImplementedError
