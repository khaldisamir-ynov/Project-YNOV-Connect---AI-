# Application Streamlit — HeteroGNN AML

Interface pour scorer les transactions du graphe hétérogène et afficher **PR-AUC, recall, precision, F1**, matrice de confusion et courbes PR.

## Prérequis

```powershell
cd "c:\Users\MASTER DATA&IA\Desktop\YNOV\Projet - Connect AI\Project"
pip install -r requirements.txt
```

Artefacts nécessaires (profil **train**) :

- `data/train_data/pyg/aml_hetero_graph.pt`
- `data/train_data/results/gnn_hetero_best.pt`
- `data/train_data/csv/cleaned/LI-Small_Trans_subgraph.csv` (détails transactions)

Pour le hold-out (**unknown**) :

```powershell
python scripts/04_build_pyg.py --mode hetero --data-profile unknown --holdout-eval
```

Puis sélectionner le profil **LI-Medium** dans l’interface.

**Checkpoint** : toujours `data/train_data/results/gnn_hetero_best.pt` (modèle entraîné sur Small).  
**Graphe** : `data/unknown_data/pyg/aml_hetero_graph.pt` (Medium).  
L’app ne cherche pas le checkpoint dans `unknown_data/results/` (il n’y est pas après un hold-out correct).

## Lancement

```powershell
streamlit run app/streamlit_app.py
```

## Fonctionnalités

| Onglet | Contenu |
|--------|---------|
| **Analyser une transaction** | Formulaire → **Fraude / Légitime**, score, comparaison au label réel |
| Vue d'ensemble | Infos modèle/graphe, métriques live, confusion, comparaison JSON entraînement |
| Métriques détaillées | Tableau par split, courbe precision/recall/F1 vs seuil |
| Courbe PR | PR curve, distribution des scores, matrice de confusion |
| Transactions | Tableau trié par score, filtres, export CSV |

### Saisie d’une transaction

1. **Transaction existante** : champs identiques au CSV sous-graphe, ou recherche par **index** (0 … 199 999).
2. **Nouvelle transaction** (comptes déjà dans le graphe) : mêmes champs ; le système **injecte** la transaction en mémoire et relance le GNN.
3. Boutons **Exemple fraude / légitime** pour préremplir le formulaire.

**Limite** : si les comptes émetteur/destinataire ne sont pas dans le graphe ego, l’analyse n’est pas possible sans reconstruire tout le pipeline RDF.
