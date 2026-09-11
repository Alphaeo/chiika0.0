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
        
        if activation == "swiglu":
            self.proj_in = nn.Linear(dim, 2 * hidden_dim, bias=False)
            self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        elif activation == "xielu":
            self.proj_in = nn.Linear(dim, hidden_dim, bias=False)
            self.activation_layer = XIELU()
            self.proj_out = nn.Linear(hidden_dim, dim, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, dim)
        return: (batch, seq, dim)

        TODO: brancher les couches definies dans __init__, en passant
        par `swiglu()` ou l'instance XIELU selon self.activation.
        """
        if self.activation == "swiglu":
            x = self.proj_in(x)
            x = swiglu(x)
            x = self.proj_out(x)
        elif self.activation == "xielu":
            x = self.proj_in(x)
            x = self.activation_layer(x)
            x = self.proj_out(x)
        return x
