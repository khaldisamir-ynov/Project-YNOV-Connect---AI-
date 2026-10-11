# Oral N+1 — Tes 3 GNN, tes chiffres, ce que RDF a changé

Projet Connect AI (Samir KHALDI).  
Données : **LI-Small** (train) → **LI-Medium** (hold-out jamais vu).  
Pas Cora. Pas le jouet à 8 comptes.

**Question type :**  
*Pourquoi HeteroGNN marche mieux que GraphSAGE et GAT ? Qu’est-ce que RDF a apporté à chacun ?*

**Réponse 25 s**  
« Les trois sont des GNN. SAGE et GAT voient un graphe **aplati** : compte–lien–compte, une seule oreille. HeteroGNN voit le RDF **tel quel** : une oreille par prédicat, plus 225 k faits inférés. Sur le test interne ils se valent presque. Sur le hold-out, seul HeteroGNN **généralise** : precision 54 % contre 13 % / 11 %. RDF n’entraîne pas. Il fabrique le graphe. SAGE jette la couleur. HeteroGNN la garde. »

---

## 1. Tes trois modèles (ce qu’ils voient)

| | GraphSAGE | GAT | HeteroGNN |
|---|---|---|---|
| Fichier | `aml_graph.pt` | même graphe | `aml_hetero_graph.pt` |
| Graphe | Homogène | Homogène | **Hétérogène** |
| Nœuds | Comptes | Comptes | Compte **+** tx **+** banque |
| Lien | La tx est un **trait** | Idem | La tx est un **nœud** |
| Oreille | Une (échantillon) | Une (attention) | **Une par prédicat RDF** |
| Tâche | Classer l’**arête** (tx) | Idem | Classer le **nœud** tx |
| Recette | Moyenne d’un échantillon | Qui écouter plus fort | HeteroConv (SAGE **par** couleur) |

Analogie.

- SAGE : sondage dans la rue, tout le monde est un « passant ».  
- GAT : même rue, micro plus fort sur certains.  
- HeteroGNN : tu sépares client / virement / banque. « Envoie » n’est pas « même banque ».

**Piège oral :** SAGE n’est pas « nul ». Dans *ce* projet, SAGE est homogène parce que `--mode homo` **aplatit**. Le même SAGE **dans** HeteroConv, c’est déjà le HeteroGNN.

---

## 2. Pourquoi les résultats **diffèrent**

Deux jeux. Ne les mélange pas.

### Test interne (LI-Small, 15 %, même distribution)

Fraude rare : 1,78 %. Hasard PR-AUC ≈ 0,018.

| | PR-AUC | Recall | Precision |
|---|---|---|---|
| HeteroGNN | 0,25 | 95 % | 6 % |
| GraphSAGE | 0,25 | 93 % | 6 % |
| GAT | 0,05 | 63 % | 3 % |

Ici **Hetero ≈ SAGE**. Dire « Hetero est meilleur sur tout » est faux.  
GAT : early stop epoch 3, graphe creux (beaucoup de degré 1), attention instable. **Contre-exemple**, pas une baseline.

### Hold-out (LI-Medium, graphe **jamais vu**)

Fraude 8 %. Degré moyen plus bas (3,3 → 1,9). 63 % des comptes n’ont **qu’un** voisin.

| | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| **HeteroGNN** | **0,60** | 70 % | **54 %** | **0,61** |
| GraphSAGE | 0,23 | **78 %** | **13 %** | 0,23 |
| GAT | 0,15 | 64 % | 11 % | 0,19 |

SAGE **rappelle** encore (78 %). Il **alerte trop** (13 % des alertes sont justes).  
Hetero : plus d’**une alerte sur deux** est une vraie fraude. C’est ça « meilleur » en AML : la file du contrôle.

Pourquoi l’homogène casse hors train : il a appris un **mélange** de câbles. Le schéma `sends_via` + `sameBankAs` ne survit pas si tout est peint pareil.  
Hold-out ≠ nœud nouveau (inductif SAGE). Hold-out = **autre graphe**, mêmes types.

---

## 3. Pourquoi HeteroGNN est meilleur **ici**

Pas « plus moderne ». Trois raisons **mesurées**.

| Raison | Concret dans le projet |
|---|---|
| **Couleur des liens** | `sends_via` ≠ `hosted_by` ≠ `transactedWith` → un \(W\) chacun |
| **Faits OWL déjà là** | +224 744 triples (`transactedWith`, `sameBankAs`) **avant** le train |
| **Généralise** | Hold-out : PR-AUC ×2,6 vs SAGE, precision 54 % vs 13 % |

Sur LI-Small, la vitesse compte aussi : Hetero plateau ~epoch 30, SAGE ~80.

**Honnêteté (jury) :** on compare **architecture + graphe enrichi** vs **architecture homogène simple**. Ce n’est pas « HeteroConv magique à graphe égal ». RDF **et** HeteroConv bougent ensemble.

---

## 4. Ce que RDF a apporté à **chaque** GNN

Le Turtle n’entre pas dans le `backward`. Pipeline : CSV → TTL → inférence → `.pt` → GNN.

| Étape RDF | GraphSAGE / GAT (`--mode homo`) | HeteroGNN (`--mode hetero`) |
|---|---|---|
| RDF asserté (`sends_via`, `deposits_to`, …) | **Aplati** : tx → arête compte→compte | **Gardé** : 3 types de nœuds, 1 conv par prédicat |
| Inférence +225 k | Perdue comme **couleur** (au mieux un trait de plus, même oreille) | **Deux canaux** : `transacted_with`, `same_bank_as` |
| OWL / T-Box | Sert peu : le `Data` n’a plus les tiroirs | Aligne HeteroData sur le livret |
| SPARQL / URI après score | Possible à côté, pas dans le mixage | Même A-Box que le modèle a vue |

**Une phrase chacun**

- **SAGE / GAT :** RDF a surtout **préparé** le CSV en graphe. Ensuite on **jette** la sémantique. Une oreille.  
- **HeteroGNN :** RDF **est** le plan d’écoute. Chaque verbe a son \(W\). Les inférences sont des voisins **typés**, pas à redécouvrir.

Sans RDF, tu peux encore coder un HeteroData à la main (noms de colonnes).  
Avec RDF, le plan est **écrit**, partageable, et **ajoute** des faits certains que le CSV n’avait pas.

---

## 5. Mini-carte orale

```
CSV IBM
  → RDF (02b) + inférence OWL (02c, +225 k)
      ├─ mode homo  → SAGE / GAT   → hold-out precision ~13 % / 11 %
      └─ mode hetero → HeteroGNN   → hold-out precision 54 %, PR-AUC 0,60
```

Si on t’interrompt :  
« RDF n’a pas rendu SAGE plus malin. SAGE n’a pas reçu les couleurs. HeteroGNN si. C’est pour ça que seul lui tient sur LI-Medium. »

Chiffres détaillés : [`comparaison_modeles_gnn.md`](comparaison_modeles_gnn.md).
