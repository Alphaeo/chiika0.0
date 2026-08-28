# chiika0.0 -- chiikamini

Un tout petit modele multimodal (texte + image) implemente "proprement"
en PyTorch, dans le seul but de **tester chiikaScreen (et les autres apps
locales) sans charger un vrai VLM de plusieurs Go en RAM** pendant le dev.

Ce n'est pas un modele qu'on espere competitif -- c'est un banc d'essai
architectural : les choix (RoPE, GQA, RMSNorm, QK-Norm, SwiGLU/xIELU,
ViT + projection LLaVA) sont ceux de vrais papiers de recherche recents,
pas des simplifications jouet. L'idee (cf. Qwen) : optimiser
l'architecture, pas juste balancer plus de data sur un Transformer
vanilla. Voir [PAPERS.md](PAPERS.md) pour une reference par brique.

## Philosophie de ce depot

Le squelette (fichiers, classes, signatures, shapes attendues, TODO) est
en place. **Le corps des fonctions "interessantes" (la vraie math) est a
ecrire par toi.** Chaque TODO precise :
- ce que la fonction doit calculer (avec la formule si non triviale),
- les shapes d'entree/sortie attendues,
- le papier a relire si besoin.

Les tests dans `tests/` servent de specification -- ils sont deja
ecrits et verifient des proprietes connues (ex: RoPE preserve la norme,
la sortie du modele a la bonne shape). Ils doivent rester rouges tant
que le TODO correspondant n'est pas implemente ; les faire passer un par
un est une bonne facon d'avancer.

Pose-moi des questions dans le chat quand tu bloques sur une formule, un
choix de shape, ou pourquoi tel papier fait tel choix -- pas besoin de
tout deviner seul.

## Structure

```
chiikamini/
  config.py              # dataclasses de config (deja remplies, valeurs "tiny")
  tokenizer.py            # wrapper autour d'un tokenizer BPE existant
  layers/
    norm.py                # RMSNorm, QK-Norm
    rope.py                # Rotary Position Embeddings
    activations.py          # SwiGLU, xIELU (experimental)
    attention.py            # Self-attention causale avec GQA + RoPE (+ MLA optionnel)
    mlp.py                  # FeedForward (SwiGLU/xIELU)
    block.py                # Bloc Transformer pre-norm (attn + FFN + residuelles)
  vision/
    patch_embed.py          # Decoupage image en patches + projection lineaire (ViT)
    vision_encoder.py       # Pile de blocs Transformer bidirectionnels
    projector.py            # MLP de projection vision -> texte (LLaVA)
  model.py                 # Assemblage : ChiikaMiniForCausalLM / ChiikaMiniVLM
  generate.py               # Boucle de generation autoregressive (KV-cache, sampling)
  train.py                  # Boucle d'entrainement minimale
  data.py                   # Datasets jouets (texte / paires image-texte)
  utils.py                  # seed, device, comptage de parametres (deja implemente)
tests/
  test_shapes.py            # Tests de specification (shapes + proprietes mathematiques)
```

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Lancer les tests (au fur et a mesure de l'implementation)

```
.venv\Scripts\python -m pytest tests/ -v
```

## Budget RAM cible

Tout est parametre dans `config.py`. Les valeurs par defaut visent un
modele de quelques dizaines de millions de parametres (dim=256, 6
couches texte, 4 couches vision) -- largement suffisant pour valider la
plomberie d'une app (prompt -> tokens -> reponse) sans jamais s'approcher
des ~3.3 Go du Qwen2.5-VL-3B GGUF utilise par chiikaScreen en "vrai"
usage. Ajuste `dim`/`n_layers` a la baisse si besoin.
