"""Config du modele. Ce fichier ne contient pas de "vraie" logique --
juste des dataclasses de valeurs. Ajuste les nombres librement selon ton
budget RAM (voir README, section "Budget RAM cible"); rien ici n'est
fige, ce sont des points de depart raisonnables pour un modele "tiny".

Conventions de nommage alignees sur Llama/Qwen (voir PAPERS.md, derniere
section) pour pouvoir comparer facilement avec leur code source.
"""

from dataclasses import dataclass, field


@dataclass
class TextConfig:
    vocab_size: int = 32000          # doit matcher tokenizer.vocab_size, voir tokenizer.py
    dim: int = 256                   # dimension d'embedding / du residual stream
    n_layers: int = 6
    n_heads: int = 8                 # tetes de requete (Q)
    n_kv_heads: int = 2              # tetes clef/valeur (GQA) -- doit diviser n_heads
    ffn_hidden_dim: int = 704        # ~ round(2/3 * 4 * dim) a un multiple de 32 pres, cf. SwiGLU (PAPERS.md)
    max_seq_len: int = 512
    rope_theta: float = 10000.0      # base de RoPE, cf. Su et al. 2021
    norm_eps: float = 1e-6
    use_qk_norm: bool = True         # QK-Norm (Qwen3), voir layers/norm.py
    ffn_activation: str = "xielu"     # "swiglu" ou "xielu" -- xielu retenu par defaut, cf.
                                      # scripts/compare_activations.py : perplexite held-out
                                      # 1061 (xielu) vs 1793 (swiglu) sur le corpus CS+code,
                                      # avec moins de parametres. Un seul run/seed, a confirmer
                                      # sur un corpus plus gros si besoin de certitude.
    tie_embeddings: bool = True      # partager les poids entre embedding et lm_head

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads doit etre un multiple de n_kv_heads (GQA)"
        assert self.dim % self.n_heads == 0, "dim doit etre un multiple de n_heads"

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16             # 224/16 = 14 -> 196 patches, cf. ViT (PAPERS.md)
    dim: int = 192
    n_layers: int = 4
    n_heads: int = 4
    ffn_hidden_dim: int = 512
    norm_eps: float = 1e-6
    n_channels: int = 3

    @property
    def n_patches(self) -> int:
        assert self.image_size % self.patch_size == 0
        side = self.image_size // self.patch_size
        return side * side

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


@dataclass
class ChiikaMiniConfig:
    """Config racine. `vision=None` -> modele texte seul (ChiikaMiniForCausalLM)."""

    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig | None = field(default_factory=VisionConfig)
    image_token_id: int = 32001       # id du token special <image> dans le vocab texte
