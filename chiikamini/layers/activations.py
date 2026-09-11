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
    gate, value = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * value


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

    Formule exacte, Eq. 9 du papier (verifiee sur arxiv.org/abs/2411.13010,
    version HTML) :

        xIELU(x) = alpha_p * x^2 + 0.5 * x                  si x > 0
                 = alpha_n * (exp(x) - 1 - x) + 0.5 * x     si x <= 0

    avec :
        alpha_p = softplus(alpha_p_raw)          -- garantit alpha_p > 0
        alpha_n = 0.5 + softplus(alpha_n_raw)    -- garantit alpha_n > 0.5

    beta_p = beta_n = 0.5 sont des CONSTANTES fixes (pas entrainables,
    contrairement a ce qu'on pourrait deviner) -- seuls alpha_p_raw et
    alpha_n_raw sont des nn.Parameter, un scalaire chacun.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: deux scalaires nn.Parameter, alpha_p_raw et alpha_n_raw
        # (valeur initiale : torch.zeros(1) ou torch.tensor(0.0) suffit,
        # softplus(0) = ln(2) ~ 0.69, un point de depart raisonnable).
        # Ce sont des tenseurs "bruts" -- alpha_p/alpha_n eux-memes sont
        # calcules a la volee dans forward() via softplus(), jamais
        # stockes directement (sinon rien n'empecherait l'optimiseur de
        # les rendre negatifs).
        self.alpha_p_raw = nn.Parameter(torch.zeros(1))
        self.alpha_n_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., hidden_dim)
        return: (..., hidden_dim), meme shape.

        TODO: implementer la formule piecewise ci-dessus avec
        torch.where(x > 0, branche_positive, branche_negative).

        Piege numerique : torch.where calcule TOUJOURS ses deux branches
        pour tous les elements avant de choisir (pas de court-circuit).
        Si la branche negative utilise exp(x) tel quel, alors pour les
        elements ou x est grand et positif (donc la branche negative sera
        de toute facon jetee), exp(x) peut deborder (overflow -> inf).
        Au forward ce n'est pas grave (torch.where jette cette valeur),
        mais au backward, le gradient de exp(x) a cet endroit vaut aussi
        inf, multiplie par un gradient-amont de 0 (puisque cette branche
        n'est pas selectionnee) -> 0 * inf = NaN, qui contamine ensuite
        tout le gradient de alpha_n_raw (somme sur tous les elements).
        Fix : calculer exp(x.clamp(max=0)) au lieu de exp(x) dans la
        branche negative -- ca ne change rien pour x <= 0 (clamp(max=0)
        ne fait rien), et ca empeche l'overflow/NaN pour les x > 0 qui de
        toute facon ne seront jamais choisis par le where.
        """
        alpha_p = torch.nn.functional.softplus(self.alpha_p_raw)
        alpha_n = 0.5 + torch.nn.functional.softplus(self.alpha_n_raw)

        positive_branch = alpha_p * x.pow(2) + 0.5 * x
        negative_branch = alpha_n * (torch.exp(x.clamp(max=0)) - 1 - x) + 0.5 * x

        return torch.where(x > 0, positive_branch, negative_branch)
