"""Evaluation du modele : une metrique quantitative (perplexite) +
une inspection qualitative (generations sur des prompts fixes).

Sert surtout a comparer objectivement deux variantes du modele (ex:
ffn_activation="swiglu" vs "xielu") sur les MEMES donnees, plutot que de
juger a l'oeil sur une seule generation aleatoire a chaque fois.
"""

import math

import torch
from torch.utils.data import DataLoader

from chiikamini.generate import generate
from chiikamini.model import ChiikaMiniForCausalLM
from chiikamini.tokenizer import ChiikaTokenizer


@torch.no_grad()
def perplexity(
    model: ChiikaMiniForCausalLM,
    dataset,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> float:
    """
    Perplexite = exp(cross-entropy moyenne PAR TOKEN) sur tout `dataset`.

    Intuition : "en moyenne, parmi combien de mots equiprobables le
    modele hesite pour deviner le mot suivant". Perplexite de 1 = modele
    parfait (jamais surpris) ; perplexite ~= vocab_size = aussi mauvais
    qu'un choix uniforme au hasard parmi tout le vocabulaire.

    TODO:
    1. model.eval() -- desactive tout comportement d'entrainement (pas
       de dropout dans ce modele pour l'instant, mais bonne habitude
       systematique avant une evaluation).
    2. Si device is not None: model.to(device)
    3. loader = DataLoader(dataset, batch_size=batch_size)
    4. Boucle sur les batches, accumuler :
       - total_loss += loss.item() * nb_tokens_du_batch
       - total_tokens += nb_tokens_du_batch
       (ponderer par le nombre de tokens, PAS juste faire la moyenne des
       moyennes de chaque batch -- le dernier batch peut etre plus petit
       que batch_size, une moyenne non ponderee le sur-representerait)
       nb_tokens_du_batch = input_ids.numel() (batch_size * seq_len)
    5. model.train() avant de retourner, pour laisser le modele pret a
       continuer l'entrainement si l'appelant enchaine dessus.
    6. return math.exp(total_loss / total_tokens)
    """
    model.eval()
    if device is not None:
        model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in loader :
        if device is not None:
            input_ids, labels = input_ids.to(device), labels.to(device)
        _, loss = model(input_ids=input_ids, labels=labels)
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
    model.train()
    return math.exp(total_loss / total_tokens)


def eval_fixed_prompts(
    model: ChiikaMiniForCausalLM,
    tokenizer: ChiikaTokenizer,
    prompts: list[str],
    max_new_tokens: int = 20,
    device: torch.device | None = None,
) -> list[str]:
    """
    Genere une completion pour chaque prompt de `prompts`, en greedy
    (temperature=0, donc deterministe) -- pour pouvoir comparer deux
    checkpoints sur EXACTEMENT les memes entrees, sans que la variance
    du sampling aleatoire ne brouille la comparaison.

    return: liste de textes generes, meme ordre que `prompts`.

    TODO:
    1. model.eval()
    2. Pour chaque prompt :
       a. ids = torch.tensor([tokenizer.encode(prompt)])
          (si device is not None: ids = ids.to(device))
       b. out = generate(model, ids, images=None,
                          max_new_tokens=max_new_tokens, temperature=0)
       c. texte = tokenizer.decode(out[0].tolist())
    3. return la liste des textes.
    """
    model.eval()
    generated_texts = []
    for prompt in prompts:
         ids = torch.tensor([tokenizer.encode(prompt)])
         if device is not None:
               ids = ids.to(device)
         out = generate(model, ids, images=None, max_new_tokens=max_new_tokens, temperature=0)
         texte = tokenizer.decode(out[0].tolist())
         generated_texts.append(texte)
    return generated_texts
