"""Encodeur vision : PatchEmbed + position embedding appris + pile de
TransformerBlock en attention *bidirectionnelle* (causal=False -- une
image n'a pas d'ordre temporel, chaque patch doit pouvoir voir tous les
autres). Voir PAPERS.md section "Vision".
"""

import torch
from torch import nn

from chiikamini.config import VisionConfig
from chiikamini.layers.block import TransformerBlock
from chiikamini.layers.norm import RMSNorm
from chiikamini.vision.patch_embed import PatchEmbed


class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # TODO:
        # - self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.n_channels, cfg.dim)
        # - self.pos_embed : position embedding APPRIS (pas RoPE ici --
        #   RoPE est pense pour des sequences 1D ordonnees ; pour un
        #   ViT "simple", un nn.Parameter de shape (1, n_patches, dim)
        #   ajoute directement aux embeddings de patches suffit).
        #   Initialisation : torch.randn * 0.02 est une convention
        #   frequente (petite variance).
        # - self.blocks = nn.ModuleList([TransformerBlock(..., causal=False)
        #   for _ in range(cfg.n_layers)])
        #   Note : pas de RoPE -> passe un `rope_freqs` factice (None)
        #   aux blocks n'a pas de sens avec l'implementation actuelle
        #   d'Attention, qui suppose RoPE. Deux options : (a) rendre
        #   `apply_rotary_emb` no-op quand rope_freqs est None, geree
        #   directement dans Attention.forward ; (b) donner a
        #   VisionEncoder son propre bloc d'attention sans RoPE. Choisis
        #   l'option (a) si tu veux reutiliser TransformerBlock tel
        #   quel -- plus simple a maintenir.
        # - self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)  (norme finale)
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (batch, n_channels, image_size, image_size)
        return: (batch, n_patches, dim)

        TODO:
        1. x = self.patch_embed(images)
        2. x = x + self.pos_embed
        3. for block in self.blocks: x = block(x, rope_freqs=None)
        4. return self.norm(x)
        """
        raise NotImplementedError
