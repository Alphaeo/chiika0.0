"""Decoupage d'une image en patches + projection lineaire (ViT,
Dosovitskiy et al. 2020). Voir PAPERS.md section "Vision".
"""

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, n_channels: int, dim: int) -> None:
        super().__init__()
        assert image_size % patch_size == 0
        self.n_patches_per_side = image_size // patch_size

        # TODO: une seule Conv2d avec kernel_size=stride=patch_size fait
        # a la fois le decoupage en patches non-chevauchants ET la
        # projection lineaire de chaque patch (astuce standard ViT --
        # equivalent a "flatten chaque patch puis Linear", mais en une
        # seule operation vectorisee) :
        #   nn.Conv2d(n_channels, dim, kernel_size=patch_size, stride=patch_size)
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (batch, n_channels, image_size, image_size)
        return: (batch, n_patches, dim)  -- une "sequence" de patches,
                prete a etre traitee comme des tokens par un Transformer.

        TODO:
        1. x = self.proj(images) -> (batch, dim, H/patch_size, W/patch_size)
        2. Aplatir les deux dimensions spatiales et transposer pour
           obtenir (batch, n_patches, dim) :
           x.flatten(2).transpose(1, 2)
        """
        raise NotImplementedError
