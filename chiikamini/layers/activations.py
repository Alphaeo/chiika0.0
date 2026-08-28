"""Activations pour le FFN. Deux options, choisies via
`TextConfig.ffn_activation` :

- "swiglu" : standard Llama/Qwen, cf. Shazeer 2020.
- "xielu"  : experimentale, cf. Huang et al. 2024 (EPFL/ETHZ).

Voir PAPERS.md section "Feed-forward / activations" pour le detail et
les liens.
"""

import torch
from torch import nn


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU applique a une projection deja faite.

    x: (..., 2 * hidden_dim) -- concatenation de deux projections
       lineaires de l'entree (le "gate" et la "valeur"), calculees en
       amont dans `layers/mlp.py` (une seule matmul plus large, coupee
       en deux ici, plutot que deux matmuls separees).
    return: (..., hidden_dim)

    TODO:
    1. Couper x en deux moities egales sur la derniere dimension :
       gate, value = x.chunk(2, dim=-1)
    2. Retourner F.silu(gate) * value
       (SiLU(z) = z * sigmoid(z), aussi appele Swish -- disponible en
       torch.nn.functional.silu, mais tu peux aussi l'ecrire a la main
       pour bien voir la formule).
    """
    raise NotImplementedError


class XIELU(nn.Module):
    """xIELU (Huang et al., 2024) -- activation entrainable derivee en
    integrant des proprietes de gradient desirees plutot qu'en bricolant
    une formule a la main (approche "gradient-first"). Contrairement a
    SwiGLU, ce n'est PAS un mecanisme de porte (gate) sur deux
    projections : c'est une non-linearite element-wise appliquee a une
    seule projection, comme ReLU ou GELU. A brancher dans
    `layers/mlp.py` sur le meme schema qu'un FFN classique (une matmul
    -> activation -> une matmul), pas sur le schema "deux projections"
    de SwiGLU.

    Formule (piecewise, avec alpha_p, alpha_n, beta entrainables,
    contraints positifs via softplus pour eviter des gradients qui
    explosent) :

        xIELU(x) = alpha_p * x^2 + beta * x                si x > 0
                 = alpha_n * (exp(min(x, 0)) - 1 - x) + beta * x   si x <= 0

    Relire https://arxiv.org/abs/2411.13010 (section derivation, eq. de
    xIELU) avant d'implementer pour verifier cette formule au mot pres --
    c'est le genre de detail ou une erreur de signe change tout le
    comportement du gradient.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: parametres entrainables alpha_p, alpha_n, beta (des
        # scalaires nn.Parameter suffisent pour une premiere version).
        # Astuce : parametrer via une variable "brute" passee dans
        # softplus() au forward, pour garantir alpha_p, alpha_n > 0
        # sans avoir a clip apres chaque step d'optimiseur.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., hidden_dim)
        return: (..., hidden_dim), meme shape.

        TODO: implementer la formule piecewise ci-dessus avec
        torch.where(x > 0, branche_positive, branche_negative).
        """
        raise NotImplementedError
