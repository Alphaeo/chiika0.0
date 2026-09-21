# TODO -- ameliorer les donnees / la regularisation

Contexte : le modele est limite par les DONNEES (27k tokens pour 6,8M de
parametres, perplexite train ~1 contre ~2000-6000 sur donnees non vues).
Les points 1, 3 et 5 sont faits (dataset v3, dedup/equilibrage, suivi
train/eval, tokenizer BPE 4096). Reste le point 4 : deux techniques pour retarder la
memorisation quand les donnees sont rares. Les tests dans
`tests/test_regularization_and_windows.py` servent de specification.

```
python -m pytest tests/test_regularization_and_windows.py -v
```

## 4a. Fenetres a offset aleatoire  (`chiikamini/data.py`)

Aujourd'hui `ToyTextDataset` decoupe le corpus en fenetres alignees sur des
multiples de `seq_len` : chaque epoch montre exactement les memes fenetres.
`RandomWindowTextDataset` (squelette deja en place) tire un offset libre
different a chaque epoch.

- [ ] `__init__` : lire + tokenizer, stocker `ids`, `seq_len`, `windows_per_epoch`,
      `seed`, `epoch = 0`, calculer `max_start`, `ValueError` si corpus trop court
- [ ] `set_epoch(epoch)`
- [ ] `__len__` -> `windows_per_epoch`
- [ ] `__getitem__(idx)` : offset deterministe a partir de `(seed, epoch, idx)`
      (generateur local, PAS d'etat aleatoire global), fenetre + labels decales d'1
- [ ] Le branchement dans `train_loop` (`set_epoch` appele a chaque epoch) est
      deja fait ; `test_train_loop_calls_set_epoch_every_epoch` le verifie

Pourquoi deterministe par `(seed, epoch, idx)` : reproductibilite, et compatibilite
avec `DataLoader(shuffle=True)` / plusieurs workers.

## 4b. Dropout residuel  (`chiikamini/layers/block.py`)

`TextConfig.dropout` existe deja et est transmis a `TransformerBlock(dropout=...)`.
Tant que rien n'est implemente, `dropout > 0` leve `NotImplementedError`
(garde-fou volontaire : pas de dropout ignore en silence).

- [ ] Dans `TransformerBlock.__init__` : `self.drop = nn.Dropout(dropout)`, puis
      supprimer le garde-fou `raise NotImplementedError`
- [ ] Dans `forward` : `x = x + self.drop(self.attn(...))` puis idem pour le FFN
      (le dropout s'applique a la SORTIE de la sous-couche, avant l'addition)
- [ ] Optionnel : dropout sur les embeddings dans `ChiikaMiniTrunk`

`nn.Dropout` est actif en `model.train()` et coupe en `model.eval()` tout seul :
c'est pour ca que `perplexity()` et `generate` ne sont pas affectes.

## 5. Tokenizer BPE de domaine  (`chiikamini/tokenizer.py`, `chiikamini/evals.py`)

Mesure (scripts/train_tokenizer.py) : GPT-2 = 2,37 octets/token, 13 % de son
vocabulaire vu au train, table d'embedding = 94 % des parametres. Un BPE de
4096 tokens : 2,90 octets/token, 92 % du vocabulaire vu, embedding 0,52M.
Taille retenue : 4096 (a 16k le vocabulaire sur-apprend le train : 9,6 % de
tokens d'eval inconnus). Tests : `tests/test_tokenizer.py`.

- [ ] `train_bpe_tokenizer(texts, vocab_size, special_tokens)` : BPE + ByteLevel
      (pre_tokenizer ET decoder) + BpeTrainer avec l'alphabet des 256 octets
      (etapes detaillees dans la docstring)
- [ ] `bits_per_byte(model, tokenizer, text_path, ...)` dans `evals.py` :
      `loss / ln2 * n_tokens / n_octets` -- seule metrique comparable entre
      deux tokenizers (la perplexite par token ne l'est pas)
- [ ] Lancer `python scripts/train_tokenizer.py` : doit reproduire le tableau ci-dessus
- [ ] Recalculer en bits/octet le modele publie (GPT-2, 27k tokens) : c'est la
      vraie reference, a la place de la perplexite 6214

Deja fait : `ChiikaTokenizer(tokenizer_path=...)` charge un tokenizer.json.

Apres le tokenizer, pour reentrainer avec : `scripts/train_and_push.py` et
`scripts/chat.py` utilisent encore `ChiikaTokenizer()` (GPT-2) ; ils devront
charger le `tokenizer.json` du checkpoint, sinon un modele entraine avec le
nouveau tokenizer serait mal decode. Le `vocab_size` du modele = celui du tokenizer.

## Etat des donnees (a ne pas casser)

- **train_v4** (defaut de `train_v2.py` / `train_tokenizer.py`) = train_v3 + 5 Mo de code
  du Hub (`codeparrot/github-code-clean`, licences permissives seulement, 1037 fichiers de
  989 depots, anti-contamination contre eval/test) : 2,38 M tokens GPT-2, x12. Provenance
  fichier par fichier (depot, chemin, licence) dans `data/manifest_v4.json`. Reconstruire :
  `python scripts/fetch_hub_code.py` (telecharge un shard de 356 Mo). eval_v3/test_v3 inchanges.
- Tokenizer par defaut : `data/tokenizer_bpe8192.json` (98 % du vocabulaire vu au train,
  3,07 octets/token). Sur 137k tokens, 4096 etait le bon choix ; sur 1,7M tokens 8192 l'est.

- Dataset **v3 gele** : `data/train_v3.txt` (200 504 tokens GPT-2), `eval_v3.txt`
  (VALIDATION : choisir LR / epoch / variantes), `test_v3.txt` (TEST : rapporter le
  chiffre final, UNE fois par modele). Empreintes sha1 dans `data/manifest_v3.json` ;
  `train_v2.py` previent si les fichiers ne correspondent plus.
- Ne pas relancer `scripts/build_dataset.py` sans raison : ca change les donnees et
  rend les runs precedents non comparables. `chiikamini` est volontairement
  hors des sources (repo en chantier => cible mouvante).
- Limite connue : eval/test contiennent tres peu de Python (le code Python du corpus
  venait de chiikamini) ; ils mesurent surtout C++ et TypeScript.
- Reference : modele publie (GPT-2, 27k tokens) = **5.579** bits/octet (eval_v3),
  **5.224** (test_v3).

## Ensuite (dans l'ordre)

- [ ] Ecrire 4a puis 4b (voir plus haut) -- les tests passent au vert
- [ ] `python scriptsblate.py` : entraine baseline / +fenetres aleatoires / +dropout /
      +les deux dans des conditions identiques et compare en bits/octet sur eval_v3
- [ ] Modele final : `python scripts	rain_v2.py --lr 3e-3 [--random-windows] [--dropout 0.1]
      --report-test` (le test n'est mesure qu'ici)
- [ ] Puis seulement : commit, republier sur HF (`scripts/train_and_push.py` utilise
      encore GPT-2 : a adapter pour lire le tokenizer.json du checkpoint)
