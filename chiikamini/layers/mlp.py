"""Feed-forward du bloc Transformer. Deux variantes selon
`ffn_activation` ("swiglu" ou "xielu"), voir layers/activations.py et
PAPERS.md.
"""

import torch
from torch import nn

from chiikamini.layers.activations import XIELU, swiglu


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "swiglu") -> None:
        super().__init__()
        assert activation in ("swiglu", "xielu")
        self.activation = activation

        # TODO:
        # - si "swiglu": une seule couche lineaire dim -> 2*hidden_dim
        #   (le chunk en deux se fait dans `swiglu()`), sans biais, puis
        #   une couche lineaire hidden_dim -> dim en sortie, sans biais.
        # - si "xielu": couche lineaire dim -> hidden_dim, une instance
        #   de XIELU(), puis couche lineaire hidden_dim -> dim. Meme
        #   schema qu'un FFN classique (pas de gate ici).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, dim)
        return: (batch, seq, dim)

        TODO: brancher les couches definies dans __init__, en passant
        par `swiglu()` ou l'instance XIELU selon self.activation.
        """
        raise NotImplementedError
