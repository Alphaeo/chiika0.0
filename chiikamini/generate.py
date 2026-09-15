"""Boucle de generation autoregressive : greedy ou sampling
(temperature / top-k / top-p), sans KV-cache pour la v1 (recalcule tout
l'historique a chaque nouveau token -- simple et correct, mais O(seq^2)
plutot que O(seq) ; suffisant pour valider la plomberie d'une app sur
des sequences courtes). Le KV-cache est une optimisation a ajouter
ensuite si la latence pose probleme -- pas un pre-requis pour tester
chiikaScreen.
"""

import torch
import torch.nn.functional as F

from chiikamini.model import ChiikaMiniForCausalLM, ChiikaMiniVLM


@torch.no_grad()
def generate(
    model: ChiikaMiniForCausalLM | ChiikaMiniVLM,
    input_ids: torch.Tensor,
    images: torch.Tensor | None,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    input_ids: (batch, seq) -- prompt initial
    images: passe a model.forward si c'est un ChiikaMiniVLM, ignore sinon
    return: (batch, seq + n) -- input_ids etendu des tokens generes
            (n <= max_new_tokens, s'arrete plus tot si eos_token_id est
            genere par TOUTES les sequences du batch)

    TODO, boucle for _ in range(max_new_tokens):
    1. Appeler le modele sur la sequence courante (tronquee a
       cfg.max_seq_len si besoin -- garder les `max_seq_len` derniers
       tokens) :
         if isinstance(model, ChiikaMiniVLM): logits, _ = model(input_ids, images=images)
         else: logits, _ = model(input_ids)
    2. Prendre les logits du DERNIER token : next_token_logits = logits[:, -1, :]
    3. Si temperature == 0: next_token = argmax (greedy).
       Sinon :
         a. next_token_logits = next_token_logits / temperature
         b. si top_k: ne garder que les top_k logits, mettre le reste a -inf
            (torch.topk puis masquer, ou torch.where)
         c. probs = F.softmax(next_token_logits, dim=-1)
         d. next_token = torch.multinomial(probs, num_samples=1)
    4. input_ids = torch.cat([input_ids, next_token], dim=1)
    5. Si eos_token_id is not None et que next_token == eos_token_id
       pour toutes les sequences du batch : break.
    6. return input_ids
    """
    for _ in range(max_new_tokens):
        if isinstance(model, ChiikaMiniVLM):
            logits, _ = model(input_ids, images=images)
        else:
            logits, _ = model(input_ids)

        next_token_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                top_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                threshold = top_values[:, -1, None]
                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids
