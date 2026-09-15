"""Boucle d'entrainement minimale : AdamW + clipping de gradient +
scheduler de LR basique (warmup lineaire puis cosine decay -- standard
sur a peu pres tous les papiers cites dans PAPERS.md). Pas de
distribution multi-GPU, pas de mixed precision avancee -- ce modele est
prevu pour tourner sur un seul CPU/GPU de dev.
"""

import torch
import math
from torch.utils.data import DataLoader

from chiikamini.utils import count_parameters, get_device, set_seed


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, grad_clip: float = 1.0) -> float:
    """
    batch: tuple retourne par le Dataset (voir data.py) -- (input_ids, labels)
           pour texte seul, ou (input_ids, images, labels) pour le VLM.
    return: la valeur scalaire de la loss (float, pour logging).

    TODO:
    1. optimizer.zero_grad()
    2. Deballer batch, appeler model(...) avec labels pour recuperer
       (logits, loss) -- voir la signature de forward dans model.py
       selon que c'est ChiikaMiniForCausalLM ou ChiikaMiniVLM.
    3. loss.backward()
    4. torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
       (evite les explosions de gradient en debut d'entrainement --
       frequent avec RMSNorm + petits batches)
    5. optimizer.step()
    6. return loss.item()
    """
    optimizer.zero_grad()
    input_ids, labels = batch
    logits, loss = model(input_ids, labels=labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def make_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """
    return: un objet avec une methode .step() qui ajuste optimizer.param_groups[*]['lr'].

    TODO: le plus simple est torch.optim.lr_scheduler.LambdaLR avec une
    fonction lr_lambda(step) qui fait :
    - montee lineaire de 0 a 1 pendant les `warmup_steps` premiers pas
    - puis decroissance cosine de 1 a ~0 jusqu'a `total_steps`
    (formule cosine standard : 0.5 * (1 + cos(pi * progress)), avec
    progress = (step - warmup_steps) / (total_steps - warmup_steps))
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(torch.tensor(progress * 3.141592653589793)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_loop(
    model: torch.nn.Module,
    dataset,
    n_epochs: int,
    batch_size: int,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    seed: int = 0,
) -> None:
    """
    Boucle complete : DataLoader -> train_step par batch -> logging
    (print) de la loss toutes les N steps.

    TODO:
    1. set_seed(seed)
    2. device = get_device() ; model.to(device)
    3. print(f"parametres entrainables: {count_parameters(model):,}")
    4. loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    5. optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    6. total_steps = n_epochs * len(loader)
    7. scheduler = make_lr_scheduler(optimizer, warmup_steps, total_steps)
    8. Boucle sur les epochs puis les batches :
       - deplacer le batch sur `device`
       - loss = train_step(model, batch, optimizer)
       - scheduler.step()
       - logguer la loss (ex: tous les 10 steps)
    """
    set_seed(seed)
    device = get_device()
    model.to(device)
    print(f"parametres entrainables: {count_parameters(model):,}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    total_steps = n_epochs * len(loader)
    scheduler = make_lr_scheduler(optimizer, warmup_steps, total_steps)
    for epoch in range(n_epochs):
        for step, batch in enumerate(loader):
            batch = [item.to(device) for item in batch]
            loss = train_step(model, batch, optimizer)
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            scheduler.step()
