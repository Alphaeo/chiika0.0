"""Boucle d'entrainement minimale : AdamW + clipping de gradient +
scheduler de LR basique (warmup lineaire puis cosine decay -- standard
sur a peu pres tous les papiers cites dans PAPERS.md). Pas de
distribution multi-GPU, pas de mixed precision avancee -- ce modele est
prevu pour tourner sur un seul CPU/GPU de dev.
"""

import torch
import math
from torch.utils.data import DataLoader

from chiikamini.evals import perplexity
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
    eval_dataset=None,
    eval_every: int = 1,
    patience: int | None = None,
    restore_best: bool = False,
    steps_per_epoch: int | None = None,
) -> list[dict]:
    """
    Boucle complete : DataLoader -> train_step par batch -> logging
    (print) de la loss toutes les N steps.

    Suivi train/eval (hygiene d'evaluation) : si `eval_dataset` est fourni,
    la perplexite sur ce jeu NON VU est calculee tous les `eval_every`
    epochs et comparee a la perplexite d'entrainement (exp de la loss
    moyenne de l'epoch). Un ecart qui se creuse = memorisation. Avec
    `patience=k`, l'entrainement s'arrete si l'eval n'a pas progresse
    depuis k evaluations. Avec `restore_best=True`, les poids du MEILLEUR epoch
    (eval_ppl minimale) sont remis dans le modele a la fin -- sinon avec
    `patience` on garderait celui des k derniers epochs, deja moins bon.
    `steps_per_epoch=k` : un "epoch" ne fait que k pas (le DataLoader re-melange a
    chaque epoch, donc chaque epoch voit un sous-ensemble different). Utile sur un gros
    corpus, ou un epoch complet dure trop longtemps pour suivre l'eval et s'arreter a temps.
    Retourne l'historique (une entree par epoch).

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
    steps_in_epoch = len(loader) if steps_per_epoch is None else min(len(loader), steps_per_epoch)
    total_steps = n_epochs * steps_in_epoch
    scheduler = make_lr_scheduler(optimizer, warmup_steps, total_steps)
    history = []
    best_eval_ppl = float("inf")
    evals_without_progress = 0
    best_state, best_epoch = None, None
    for epoch in range(n_epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        epoch_losses = []
        for step, batch in enumerate(loader):
            if step >= steps_in_epoch:
                break
            batch = [item.to(device) for item in batch]
            loss = train_step(model, batch, optimizer)
            epoch_losses.append(loss)
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            scheduler.step()

        train_loss = sum(epoch_losses) / len(epoch_losses)
        record = {"epoch": epoch, "train_loss": train_loss, "train_ppl": math.exp(train_loss)}
        if eval_dataset is not None and (epoch + 1) % eval_every == 0:
            record["eval_ppl"] = perplexity(model, eval_dataset, batch_size=batch_size, device=device)
            improved = record["eval_ppl"] < best_eval_ppl
            best_eval_ppl = min(best_eval_ppl, record["eval_ppl"])
            evals_without_progress = 0 if improved else evals_without_progress + 1
            if improved and restore_best:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            print(
                f"[epoch {epoch}] train_ppl={record['train_ppl']:.1f}  eval_ppl={record['eval_ppl']:.1f}  "
                f"ecart x{record['eval_ppl'] / record['train_ppl']:.1f}" + ("  <- meilleur" if improved else "")
            )
        history.append(record)
        if patience is not None and evals_without_progress >= patience:
            print(f"Arret anticipe : eval_ppl sans progres depuis {patience} evaluation(s).")
            break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Poids du meilleur epoch restaures (epoch {best_epoch}, eval_ppl={best_eval_ppl:.1f}).")
    return history
