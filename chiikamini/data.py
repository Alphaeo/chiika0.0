"""Datasets jouets, juste assez pour verifier que la boucle
d'entrainement (train.py) tourne et que la loss descend sur un tout
petit corpus -- pas pour obtenir un modele de qualite.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from chiikamini.tokenizer import ChiikaTokenizer


class ToyTextDataset(Dataset):
    """Charge un fichier texte brut, le tokenize une fois, et decoupe en
    fenetres de longueur fixe pour l'entrainement du decoder texte seul.
    """

    def __init__(self, text_path: str | Path, tokenizer: ChiikaTokenizer, seq_len: int) -> None:
        """
        TODO:
        1. Lire le fichier texte (Path(text_path).read_text(encoding="utf-8")).
        2. Tokenizer tout le texte d'un coup avec tokenizer.encode(...).
        3. Stocker self.ids (list[int] ou torch.tensor) et self.seq_len.
        """
        text = Path(text_path).read_text(encoding="utf-8")
        self.ids = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self) -> int:
        """TODO: nombre de fenetres de seq_len+1 tokens qu'on peut extraire
        de self.ids sans chevauchement (ou avec un stride de ton choix)."""
        return (len(self.ids) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return: (input_ids, labels), chacun (seq_len,)
        labels = input_ids decale de 1 position vers la gauche (le
        modele apprend a predire le token suivant) -- ex: si la fenetre
        est [t0, t1, t2, t3], input_ids = [t0, t1, t2], labels = [t1, t2, t3].

        TODO: extraire une fenetre de seq_len+1 tokens a partir de
        self.ids[idx * seq_len : idx * seq_len + seq_len + 1], puis
        couper en input_ids (les seq_len premiers) et labels (les
        seq_len derniers).
        """
        input_ids = self.ids[idx * self.seq_len : idx * self.seq_len + self.seq_len]
        labels = self.ids[idx * self.seq_len + 1 : idx * self.seq_len + self.seq_len + 1]
        return torch.tensor(input_ids), torch.tensor(labels)


class ToyVLDataset(Dataset):
    """Paires (image, texte) jouets pour tester ChiikaMiniVLM. Pour la
    v1, une poignee d'images synthetiques (formes/couleurs generees a la
    volee avec PIL, cf. la fonction `_make_synthetic_image`) associees a
    une legende template ("un carre rouge", "un cercle bleu", ...)
    suffit largement -- pas besoin d'un vrai dataset de captioning pour
    valider la plomberie.
    """

    def __init__(self, tokenizer: ChiikaTokenizer, n_samples: int, image_size: int, n_patches: int) -> None:
        """
        TODO:
        1. Stocker tokenizer, image_size, n_patches.
        2. Generer (ou lister) n_samples paires (forme, couleur) ->
           servira a produire l'image et la legende a la volee dans
           __getitem__ (pas besoin de tout pre-generer en memoire pour
           un dataset aussi petit).
        """
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.n_patches = n_patches
        self.shapes = ["carre", "cercle", "triangle"]
        self.colors = ["rouge", "vert", "bleu"]
        self.samples = [(shape, color) for shape in self.shapes for color in self.colors]
        if len(self.samples) < n_samples:
            raise ValueError(f"n_samples ({n_samples}) is greater than the number of unique shape-color pairs ({len(self.samples)}).")
        self.samples = self.samples[:n_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _make_synthetic_image(self, shape: str, color: str) -> torch.Tensor:
        """
        return: (n_channels, image_size, image_size), float, normalise
                dans [0, 1] ou [-1, 1] (choisis une convention et reste
                coherent avec ce que VisionEncoder/PatchEmbed attendent).

        TODO: utiliser PIL (Image.new + ImageDraw) pour dessiner une
        forme simple (rectangle/ellipse) de la couleur donnee sur fond
        neutre, puis convertir en tensor via torch.from_numpy(np.array(img)).
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return: (input_ids, images, labels)
        - images: (n_channels, image_size, image_size)
        - input_ids: (n_patches + n_tokens_texte,) -- n_patches copies du
          token <image> suivies des tokens de la legende
        - labels: meme longueur que input_ids, avec -100 sur les
          positions correspondant aux tokens <image> (on ne veut pas que
          le modele soit penalise/entraine a "predire" un token image --
          seule la legende compte dans la loss). Voir la convention
          `ignore_index=-100` dans ChiikaMiniForCausalLM.forward.

        TODO: assembler tout ca a partir de self._make_synthetic_image
        et de tokenizer.encode(legende).
        """
        raise NotImplementedError
