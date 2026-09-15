"""Sauvegarde/chargement des poids + publication sur le Hugging Face Hub.

Format : safetensors, pas .pt/pickle -- standard sur le Hub, et plus sur
(torch.load sur un fichier .pt peut executer du code arbitraire au
chargement si le fichier est malveillant ; safetensors ne contient que
des tenseurs bruts, rien d'executable).

Piege verifie empiriquement : ChiikaMiniForCausalLM fait du weight
tying (`lm_head.weight` et `trunk.embed_tokens.weight` sont litteralement
le meme tenseur). `safetensors.torch.save_file` refuse un state_dict
avec des tenseurs partages (erreur explicite). `save_model`/`load_model`
(au lieu de `save_file`/`load_file`) gerent ce cas automatiquement --
un seul exemplaire sauvegarde sur disque, le tying est recree tel quel
au chargement. Verifie avec un aller-retour + comparaison de logits.
"""

import dataclasses
import json
from pathlib import Path

from safetensors.torch import load_model, save_model

from chiikamini.config import TextConfig
from chiikamini.model import ChiikaMiniForCausalLM


def save_checkpoint(model: ChiikaMiniForCausalLM, cfg: TextConfig, out_dir: str | Path) -> None:
    """Sauvegarde les poids (model.safetensors) et la config (config.json)
    dans out_dir -- les deux fichiers necessaires pour recharger le
    modele sans deviner les hyperparametres utilises a l'entrainement.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))


def load_checkpoint(checkpoint_dir: str | Path) -> ChiikaMiniForCausalLM:
    """Reconstruit un ChiikaMiniForCausalLM a partir d'un dossier produit
    par save_checkpoint (config.json + model.safetensors)."""
    checkpoint_dir = Path(checkpoint_dir)
    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg = TextConfig(**cfg_dict)
    model = ChiikaMiniForCausalLM(cfg)
    load_model(model, str(checkpoint_dir / "model.safetensors"))
    return model


def push_to_hub(checkpoint_dir: str | Path, repo_id: str, private: bool = True) -> str:
    """Publie un checkpoint (deja sauvegarde via save_checkpoint) sur le
    Hugging Face Hub. Necessite d'etre deja authentifie (`hf auth login`
    ou variable d'environnement HF_TOKEN) -- pas de token en argument ici
    pour eviter d'en faire circuler un en clair dans le code/l'historique
    git.

    repo_id: "ton-namespace/nom-du-modele" (ex: "Crocolil/chiikamini-toy")
    private: True par defaut -- pousse en prive tant que le modele n'est
             qu'un test de plomberie, pas un vrai modele a partager.
    return: l'URL du repo sur le Hub.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(checkpoint_dir), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
