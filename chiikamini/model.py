"""Assemblage final. Trois classes :

- `ChiikaMiniTrunk` : le decoder texte seul (embedding + blocks + norme
  finale), sans tete de sortie -- reutilise tel quel par les deux
  classes suivantes.
- `ChiikaMiniForCausalLM` : trunk + tete lineaire vers le vocabulaire +
  perte cross-entropy. Modele texte-only.
- `ChiikaMiniVLM` : ChiikaMiniForCausalLM + VisionEncoder + VisionProjector.
  Remplace les embeddings des tokens `<image>` par les embeddings de
  patches projetes (cf. LLaVA, PAPERS.md) avant de passer dans le trunk.
"""

import torch
from torch import nn

from chiikamini.config import ChiikaMiniConfig, TextConfig
from chiikamini.layers.block import TransformerBlock
from chiikamini.layers.norm import RMSNorm
from chiikamini.layers.rope import precompute_rope_freqs
from chiikamini.vision.projector import VisionProjector
from chiikamini.vision.vision_encoder import VisionEncoder


class ChiikaMiniTrunk(nn.Module):
    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # TODO:
        # - self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        # - self.blocks = nn.ModuleList([TransformerBlock(dim=cfg.dim,
        #     n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
        #     ffn_hidden_dim=cfg.ffn_hidden_dim, norm_eps=cfg.norm_eps,
        #     use_qk_norm=cfg.use_qk_norm, ffn_activation=cfg.ffn_activation,
        #     causal=True) for _ in range(cfg.n_layers)])
        # - self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        # - self.register_buffer("rope_freqs",
        #     precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
        #     persistent=False)
        raise NotImplementedError

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        inputs_embeds: (batch, seq, dim) -- deja les embeddings, PAS les
                       ids. Ce choix (accepter des embeddings plutot que
                       des ids en entree du trunk) est ce qui permet a
                       `ChiikaMiniVLM` de melanger tokens texte et
                       patches image AVANT d'entrer dans les blocks --
                       voir `token_embeddings()` ci-dessous pour la
                       version "a partir d'ids".
        return: (batch, seq, dim)

        TODO:
        1. seq_len = inputs_embeds.shape[1]
        2. rope_freqs = self.rope_freqs[:seq_len]
        3. x = inputs_embeds
        4. for block in self.blocks: x = block(x, rope_freqs)
        5. return self.norm(x)
        """
        raise NotImplementedError

    def token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """TODO: return self.embed_tokens(input_ids) -- (batch, seq) -> (batch, seq, dim)."""
        raise NotImplementedError


class ChiikaMiniForCausalLM(nn.Module):
    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = ChiikaMiniTrunk(cfg)

        # TODO:
        # - self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # - si cfg.tie_embeddings: self.lm_head.weight = self.trunk.embed_tokens.weight
        #   (weight tying -- embedding d'entree et matrice de sortie
        #   partagent les memes poids, cf. Press & Wolf 2016 ; divise
        #   le nombre de parametres du vocabulaire par deux, souvent
        #   sans perte de qualite voire un gain sur petits modeles).
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        """
        input_ids: (batch, seq)
        labels: (batch, seq) ou None -- si fourni, calcule la loss
                cross-entropy en plus des logits (labels decales de 1
                par rapport a input_ids : labels[i] = input_ids[i+1]
                pour la prediction du token suivant -- a faire faire par
                l'appelant dans train.py, PAS ici, pour garder ce
                forward simple).
        return: logits (batch, seq, vocab_size), et loss (scalar) si
                labels is not None sinon None.

        TODO:
        1. embeds = self.trunk.token_embeddings(input_ids)
        2. hidden = self.trunk(embeds)
        3. logits = self.lm_head(hidden)
        4. loss = None
        5. if labels is not None:
             loss = F.cross_entropy(
                 logits.reshape(-1, logits.size(-1)),
                 labels.reshape(-1),
                 ignore_index=-100,   # convention standard pour masquer certains tokens de la loss
             )
        6. return logits, loss
        """
        raise NotImplementedError


class ChiikaMiniVLM(nn.Module):
    def __init__(self, cfg: ChiikaMiniConfig) -> None:
        super().__init__()
        assert cfg.vision is not None, "ChiikaMiniVLM requiert cfg.vision (sinon utilise ChiikaMiniForCausalLM)"
        self.cfg = cfg

        # TODO:
        # - self.lm = ChiikaMiniForCausalLM(cfg.text)
        # - self.vision_encoder = VisionEncoder(cfg.vision)
        # - self.projector = VisionProjector(cfg.vision.dim, cfg.text.dim)
        raise NotImplementedError

    def _merge_text_and_image_embeds(
        self, input_ids: torch.Tensor, images: torch.Tensor | None
    ) -> torch.Tensor:
        """Construit la sequence d'embeddings d'entree du trunk texte,
        en substituant les embeddings de patches image aux positions du
        token special `<image>` (cf. LLaVA, PAPERS.md section "Vision").

        input_ids: (batch, seq) -- contient `cfg.image_token_id` a la
                   ou aux positions ou l'image doit s'inserer.
        images: (batch, n_channels, image_size, image_size) ou None
                (mode texte seul, aucun token image dans input_ids).
        return: (batch, seq, dim)

        TODO (cas le plus simple : un seul groupe contigu de tokens
        <image>, dont le nombre == cfg.vision.n_patches, par exemple
        l'utilisateur a pre-rempli input_ids avec n_patches copies du
        token <image>) :
        1. text_embeds = self.lm.trunk.token_embeddings(input_ids)  -- (batch, seq, dim)
        2. if images is None: return text_embeds
        3. vision_feats = self.projector(self.vision_encoder(images))  -- (batch, n_patches, dim)
        4. Repere le masque des positions image :
           image_mask = (input_ids == self.cfg.image_token_id)  -- (batch, seq), bool
        5. Remplace text_embeds[image_mask] par vision_feats.reshape(-1, dim)
           (attention a l'ordre : ca suppose que le nombre de True dans
           image_mask, par exemple, correspond exactement a
           batch * n_patches, et que l'ordre des positions correspond a
           l'ordre des patches -- valable si tu construis tes sequences
           d'entree ainsi dans data.py).
        6. return text_embeds (modifie en place ou reconstruit, au choix)
        """
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """
        Meme contrat que ChiikaMiniForCausalLM.forward, avec `images` en
        plus. return: (logits, loss).

        TODO:
        1. embeds = self._merge_text_and_image_embeds(input_ids, images)
        2. hidden = self.lm.trunk(embeds)
        3. logits = self.lm.lm_head(hidden)
        4. loss = None ; meme calcul de cross-entropy que ChiikaMiniForCausalLM
           si labels is not None.
        5. return logits, loss
        """
        raise NotImplementedError
