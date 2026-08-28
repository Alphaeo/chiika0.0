"""Wrapper de tokenizer. Pour un modele "tiny" dont le but est de
tester la plomberie d'une app (pas la qualite linguistique), entrainer
un BPE from scratch est un detour couteux pour peu de valeur -- charger
un tokenizer BPE deja entraine (ex: GPT-2 via la lib `tokenizers`) et
lui ajouter un token special `<image>` est largement suffisant.

Si un jour tu veux entrainer ton propre BPE (coherent avec l'esprit
"from scratch" de chiikaml), `tokenizers` a un `trainers.BpeTrainer`
pret a l'emploi -- pas necessaire pour la v1.
"""

from tokenizers import Tokenizer


class ChiikaTokenizer:
    def __init__(self, hf_tokenizer_name: str = "gpt2", image_token: str = "<image>") -> None:
        """
        TODO:
        1. Charger un tokenizer pre-entraine : Tokenizer.from_pretrained(hf_tokenizer_name)
           (necessite le paquet `tokenizers`, deja dans requirements.txt ;
           telecharge le fichier tokenizer.json au premier appel, mis en
           cache ensuite).
        2. Ajouter le token special `<image>` s'il n'existe pas deja
           dans le vocabulaire (voir `Tokenizer.add_special_tokens`).
        3. Stocker self.image_token_id = self.tokenizer.token_to_id(image_token)
           -- doit correspondre a `ChiikaMiniConfig.image_token_id`, a
           synchroniser toi-meme entre les deux (pas de verif automatique
           pour l'instant).
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """TODO: retourner self.tokenizer.get_vocab_size()."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """TODO: retourner self.tokenizer.encode(text).ids"""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """TODO: retourner self.tokenizer.decode(ids)"""
        raise NotImplementedError
