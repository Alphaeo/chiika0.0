"""Projection des embeddings vision vers l'espace d'embedding texte
(LLaVA-style, Liu et al. 2023). Voir PAPERS.md section "Vision".
"""

import torch
from torch import nn


class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int) -> None:
        super().__init__()
        # TODO: MLP simple, 2 couches suffisent pour ce modele "tiny" :
        #   Linear(vision_dim, text_dim) -> GELU -> Linear(text_dim, text_dim)
        # (LLaVA original utilise GELU ici, pas SwiGLU/xIELU -- pas
        # besoin de reprendre l'activation du decoder texte pour ce
        # petit MLP de projection).
        raise NotImplementedError

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        """
        vision_embeds: (batch, n_patches, vision_dim) -- sortie de VisionEncoder
        return: (batch, n_patches, text_dim) -- prets a etre inseres dans
                la sequence de tokens texte (voir model.py, ChiikaMiniVLM).
        """
        raise NotImplementedError
