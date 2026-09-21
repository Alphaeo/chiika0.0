"""Wrapper de tokenizer. Pour un modele "tiny" dont le but est de
tester la plomberie d'une app (pas la qualite linguistique), entrainer
un BPE from scratch est un detour couteux pour peu de valeur -- charger
un tokenizer BPE deja entraine (ex: GPT-2 via la lib `tokenizers`) et
lui ajouter un token special `<image>` est largement suffisant.

Si un jour tu veux entrainer ton propre BPE (coherent avec l'esprit
"from scratch" de chiikaml), `tokenizers` a un `trainers.BpeTrainer`
pret a l'emploi -- pas necessaire pour la v1.
"""

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Entraine un tokenizer BPE *byte-level* sur `texts` (voir TODO.md, 5).

    Pourquoi un tokenizer maison : le vocabulaire GPT-2 (50k tokens, web
    anglais) est mal adapte a du code -- 94 % des parametres du modele
    finissaient dans la table d'embedding, dont 87 % de lignes jamais
    vues a l'entrainement. Un vocabulaire de ~4k tokens appris sur NOTRE
    corpus est plus court a encoder (~2,9 octets/token contre 2,4) et
    chaque token est vu souvent.

    texts: iterable de chaines (ex: [contenu_de_data/train_v2.txt]). NE PAS
           y mettre le jeu d'eval (le vocabulaire fuirait des stats de l'eval).
    vocab_size: taille cible. Le vrai vocabulaire peut etre plus petit si le
           corpus ne permet plus de fusions utiles (vu a 16k sur 485 Ko).
    special_tokens: ex ["<image>"] -- recoivent des ids fixes en tete.
    return: le Tokenizer entraine (l'appelant fait .save(chemin)).

    TODO (bibliotheque `tokenizers`, deja installee) :
    1. tok = Tokenizer(models.BPE())
       (pas d'unk_token : en byte-level, tout octet est representable, il
       n'existe donc jamais de token "inconnu")
    2. tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
       -- decoupe le texte en "mots" au sens GPT-2 et le convertit en octets.
       add_prefix_space=False comme GPT-2 : ne pas inserer d'espace fantome
       au debut, sinon decode(encode(x)) != x.
    3. tok.decoder = decoders.ByteLevel()
       -- reconstitue les octets a la decompression. Sans lui, decode()
       renvoie des caracteres etranges ("Ġ" pour un espace).
    4. trainer = trainers.BpeTrainer(
           vocab_size=vocab_size,
           special_tokens=special_tokens or [],
           initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
           show_progress=False)
       -- initial_alphabet force les 256 octets dans le vocabulaire de base :
       c'est ce qui garantit qu'un caractere jamais vu a l'entrainement
       (emoji, accent) reste encodable octet par octet.
    5. tok.train_from_iterator(texts, trainer)
    6. return tok
    Import a ajouter en haut du fichier : models, pre_tokenizers, decoders,
    trainers, depuis `tokenizers`.
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens or [],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


class ChiikaTokenizer:
    def __init__(
        self,
        hf_tokenizer_name: str = "gpt2",
        image_token: str = "<image>",
        tokenizer_path: str | Path | None = None,
    ) -> None:
        """
        `tokenizer_path` : si fourni, charge ce tokenizer.json (ex: produit
        par scripts/train_tokenizer.py, ou celui sauvegarde a cote d'un
        checkpoint) au lieu de telecharger `hf_tokenizer_name`.

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
        if tokenizer_path is not None:
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            self.tokenizer = Tokenizer.from_pretrained(hf_tokenizer_name)
        if self.tokenizer.token_to_id(image_token) is None:
            self.tokenizer.add_special_tokens([image_token])
        self.image_token_id = self.tokenizer.token_to_id(image_token)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
