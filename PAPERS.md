# Papiers de reference

Une source par brique architecturale. Objectif : comprendre le *pourquoi*
avant d'ecrire le *comment* dans les TODO. Qwen (et la plupart des labs
serieux en 2024-2025) gagnent moins en jetant plus de data que via des
choix d'architecture cumules -- chaque ligne ci-dessous est un de ces choix.

## Normalisation

- **RMSNorm** -- Zhang & Sennrich, 2019, *"Root Mean Square Layer
  Normalization"* -- https://arxiv.org/abs/1910.07467
  Retire le recentrage (mean subtraction) de LayerNorm, garde seulement le
  rescaling par la RMS. Moins cher, aussi stable. Utilise par Llama, Qwen,
  Mistral etc.

- **QK-Norm** -- Henry et al., 2020, *"Query-Key Normalization for
  Transformers"* -- https://arxiv.org/abs/2010.04245 ; repris explicitement
  dans le **Qwen3 Technical Report** (2025) -- https://arxiv.org/abs/2505.09388
  pour stabiliser l'entrainement apres avoir retire le biais QKV. Idee : on
  normalise Q et K (RMSNorm par tete) avant le produit scalaire, ce qui
  borne la magnitude des logits d'attention independamment de la longueur
  de sequence ou de l'echelle des poids.

## Position

- **RoPE (Rotary Position Embedding)** -- Su et al., 2021, *"RoFormer:
  Enhanced Transformer with Rotary Position Embedding"* --
  https://arxiv.org/abs/2104.09864
  Encode la position en faisant tourner des paires de coordonnees de
  Q/K dans le plan complexe, d'un angle proportionnel a la position.
  Proprietes cles a verifier dans tes tests : la rotation preserve la
  norme du vecteur, et le produit scalaire post-rotation ne depend que
  de la position *relative* entre deux tokens.

## Attention

- **Grouped Query Attention (GQA)** -- Ainslie et al., 2023, *"GQA:
  Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints"* -- https://arxiv.org/abs/2305.13245
  Plusieurs tetes de requete (Q) partagent une seule tete clef/valeur
  (K/V). Compromis entre Multi-Head Attention (qualite max, KV-cache
  gros) et Multi-Query Attention (KV-cache minuscule, qualite en baisse).
  Qwen3-8B par exemple : 32 tetes Q pour seulement 8 tetes KV.

- (Optionnel, phase 2) **Multi-head Latent Attention (MLA)** -- DeepSeek-AI,
  2024, *"DeepSeek-V2"* -- https://arxiv.org/abs/2405.04434
  Compresse K et V dans un vecteur latent de rang faible avant de les
  mettre en cache, puis les reconstruit a la volee. Reduit drastiquement
  la taille du KV-cache par rapport a GQA. Voir `layers/attention.py`,
  classe `MultiHeadLatentAttention` (stub, non requis pour la v1).

## Feed-forward / activations

- **GLU Variants (SwiGLU)** -- Shazeer, 2020, *"GLU Variants Improve
  Transformer"* -- https://arxiv.org/abs/2002.05202
  Le FFN classique `Linear -> ReLU -> Linear` devient
  `(SiLU(xW) * xV) -> Linear`, un mecanisme de porte (gate) plutot qu'une
  simple non-linearite. A parametres egaux, ameliore systematiquement la
  perplexite. Standard dans Llama/Qwen/Mistral.

- **xIELU (experimental)** -- Huang et al., 2024, *"Deriving Activation
  Functions via Integration"* -- https://arxiv.org/abs/2411.13010
  Papier recent (EPFL/ETHZ) qui derive une activation en integrant des
  proprietes de gradient desirees plutot qu'en bricolant une formule.
  xIELU melange un gradient positif croissant entrainable (comme ReLU^2)
  et un flux de gradient negatif entrainable (comme xSiLU). Battrait
  SwiGLU et ReLU^2 a cout de calcul egal sur des Llama 1.1B / 126B tokens.
  Bonne piste pour "vraiment" tester une architecture novatrice plutot
  que de recopier Llama. Voir `layers/activations.py`, classe `XIELU`.

## Vision (pour la variante VLM)

- **ViT** -- Dosovitskiy et al., 2020, *"An Image is Worth 16x16 Words"*
  -- https://arxiv.org/abs/2010.11929
  Une image est decoupee en patches (ex: 16x16 px), chaque patch est
  projete lineairement en un "token", puis on applique un Transformer
  standard (attention *bidirectionnelle*, pas causale) sur la sequence
  de patches.

- **LLaVA (projection vision -> texte)** -- Liu et al., 2023, *"Visual
  Instruction Tuning"* -- https://arxiv.org/abs/2304.08485
  Un simple MLP (1-2 couches) projette les embeddings de sortie de
  l'encodeur vision dans l'espace d'embedding du decoder texte. Les
  tokens image resultants sont ensuite inseres dans la sequence de
  tokens texte a la position d'un token special `<image>`, et le
  decoder texte les traite comme des tokens normaux (self-attention
  causale sur le tout).

## Vocabulaire du fichier config.py

`dim`, `n_layers`, `n_heads`, `n_kv_heads`, `ffn_hidden_dim`, `vocab_size`
suivent les conventions Llama/Qwen -- si un terme n'est pas clair en
ecrivant une fonction, chercher ce nom exact dans le code source public de
Llama ou Qwen (HuggingFace `transformers`, fichier `modeling_llama.py` /
`modeling_qwen2.py`) donne presque toujours l'implementation de reference.
