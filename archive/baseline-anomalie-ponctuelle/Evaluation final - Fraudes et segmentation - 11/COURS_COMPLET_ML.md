# 🎓 Cours Complet — Projet Fraude et Segmentation
### M1 IA DATA — Machine Learning — YNOV
> Explication pédagogique de A à Z, du débutant à l'expert

---

## Table des matières
1. [Vue d'ensemble globale](#1-vue-densemble-globale)
2. [Concepts Fondamentaux](#2-concepts-fondamentaux)
3. [Théorie Mathématique](#3-théorie-mathématique)
4. [Fonctionnement Interne des Modèles](#4-fonctionnement-interne-des-modèles)
5. [Explication Ligne par Ligne du Code Clé](#5-explication-ligne-par-ligne-du-code-clé)
6. [Visualisation Mentale et Intuition](#6-visualisation-mentale-et-intuition)
7. [Bonnes Pratiques et Pièges Courants](#7-bonnes-pratiques-et-pièges-courants)
8. [Niveau Avancé — Sous le Capot](#8-niveau-avancé--sous-le-capot)
9. [Résumé Final](#9-résumé-final)

---

## 1. Vue d'ensemble globale

### Quel est le problème résolu ?

Une banque reçoit des milliers de transactions par seconde. Parmi elles, une infime fraction (~0.17%) sont frauduleuses. Le défi est de **les identifier automatiquement** sans bloquer les transactions légitimes.

C'est un problème à **trois niveaux** dans ce projet :

1. **Détection supervisée** : apprendre à reconnaître une fraude à partir d'exemples passés étiquetés
2. **Segmentation** : regrouper les transactions par comportement similaire (sans étiquettes)
3. **Détection d'anomalies** : trouver les transactions qui "ne ressemblent à rien d'autre"

### Pourquoi le Machine Learning ?

Les règles manuelles ("bloquer si montant > 5000€") ne fonctionnent pas : les fraudeurs s'adaptent. Le ML apprend des **patterns complexes et multidimensionnels** (30 variables à la fois) qu'un humain ne pourrait pas coder à la main.

### Quel type de ML ?

| Partie | Type | Pourquoi |
|--------|------|----------|
| Régression Logistique + Random Forest | **Supervisé** | On a des étiquettes (0/1) pour entraîner |
| K-Means, DBSCAN | **Non supervisé** | On cherche des groupes sans étiquettes |
| Isolation Forest, LOF | **Non supervisé** (détection d'anomalies) | On cherche les points "différents" |

---

## 2. Concepts Fondamentaux

### Dataset

Un dataset, c'est un **tableau de 284 807 lignes × 31 colonnes**. Chaque ligne = une transaction bancaire. Chaque colonne = une information sur cette transaction.

```
         Time    V1     V2    ...   V28   Amount   Class
ligne 0   0.0  -1.36  -0.07  ...   0.02    149.62    0
ligne 1   0.0   1.19   0.26  ...  -0.05      2.69    0
...
ligne X  406.0  -2.31   1.95  ...  -0.15    239.93    1  ← fraude
```

### Features (variables d'entrée = X)

Ce sont les **colonnes d'entrée** que le modèle va lire pour faire sa prédiction : `V1` à `V28`, `Amount_log`, `V14_ecart`, etc.

> **Analogie :** si tu veux prédire si quelqu'un est malade, les features sont sa température, son pouls, ses résultats de sang...

Les `V1–V28` sont déjà le résultat d'une **PCA** faite par la banque pour anonymiser les données. On ne sait pas ce qu'elles représentent concrètement.

### Label (variable cible = y)

C'est la colonne `Class` : **0 = normale, 1 = fraude**. C'est ce que le modèle doit apprendre à prédire.

### Train / Validation / Test

```
Dataset complet (284 807 transactions)
          │
          ├── Train (80% = 227 845) → le modèle apprend dessus
          │         │
          │         └── + SMOTE → données synthétiques pour équilibrer
          │
          └── Test  (20% =  56 962) → on évalue les vraies performances
```

> **Règle d'or :** le modèle ne doit JAMAIS voir le test pendant l'entraînement. Sinon, c'est de la triche — il "connaît" déjà les réponses.

### Modèle

C'est la "boîte" mathématique qui, à partir des features, produit une prédiction. Ce projet utilise :
- `LogisticRegression` : une équation linéaire avec une fonction sigmoid
- `RandomForestClassifier` : une forêt de centaines d'arbres de décision

### Entraînement (`model.fit`)

C'est la phase où le modèle **ajuste ses paramètres internes** en regardant les exemples du train. Il cherche les paramètres qui minimisent l'erreur de prédiction.

### Inférence (`model.predict`)

C'est la phase où le modèle **utilise ce qu'il a appris** pour prédire des nouvelles données (ici, le test set).

### Overfitting / Underfitting

```
     Underfitting          Bon ajustement          Overfitting
     (trop simple)                                 (trop complexe)

  ────────────────     ·  ·/·  ·  ·             ·  /·  ·  ·
  ────────────────    ·  / ·  ·  ·             ·  /\·/·  ·
  ────────────────   · /  ·  ·  ·             · /  · \/ ·

  Erreur train: haute  Erreur train: basse       Erreur train: très basse
  Erreur test:  haute  Erreur test:  basse       Erreur test:  haute ⚠️
```

La **cross-validation** sert justement à détecter l'overfitting avant de toucher au test.

### Généralisation

Un bon modèle généralise : ses performances sur le **test** (données jamais vues) sont proches de ses performances sur le **train**. Un modèle qui mémorise les données d'entraînement sans généraliser est dit "overfitté".

---

## 3. Théorie Mathématique

### 3.1 Régression Logistique

L'idée : transformer une somme pondérée des features en une **probabilité** entre 0 et 1.

**Étape 1 : combinaison linéaire**
```
z = w₀ + w₁×V1 + w₂×V2 + ... + w₃₅×V14_ecart
```
`w₀, w₁, ..., w₃₅` sont les **poids** (paramètres appris). Chaque poids indique "dans quelle mesure cette variable est importante".

**Étape 2 : fonction sigmoid**
```
P(fraude) = σ(z) = 1 / (1 + e^(-z))
```

La sigmoid "écrase" n'importe quel nombre entre 0 et 1 :
- Si z est très grand → P ≈ 1 (fraude probable)
- Si z est très petit → P ≈ 0 (normale probable)
- Si z = 0 → P = 0.5 (incertain)

**Étape 3 : décision**
```python
y_pred = 1  si P(fraude) > 0.5  else 0
```

**La Loss Function (Binary Cross-Entropy) :**
```
L = -[ y × log(P) + (1-y) × log(1-P) ]
```

| Vraie classe | Prédiction P | Loss | Interprétation |
|---|---|---|---|
| 1 (fraude) | 0.99 | ≈ 0.01 | Bonne prédiction, faible pénalité |
| 1 (fraude) | 0.01 | ≈ 4.6  | Mauvaise prédiction, forte pénalité |
| 0 (normal) | 0.01 | ≈ 0.01 | Bonne prédiction |
| 0 (normal) | 0.99 | ≈ 4.6  | Mauvaise prédiction |

L'optimisation (gradient descent) ajuste les `w` pour **minimiser cette loss** sur tout le dataset.

---

### 3.2 Random Forest

Un Random Forest, c'est un **vote démocratique de 100 arbres de décision**.

**Un arbre de décision, c'est quoi ?**
```
             [V14 < -3.2 ?]
            /              \
          OUI              NON
          /                  \
  [Amount_log < 2.1 ?]    [V17 < -5.5 ?]
    /        \               /       \
 FRAUDE    NORMALE       FRAUDE    NORMALE
```
Chaque nœud pose une question sur une feature. On suit les branches jusqu'à arriver à une feuille (prédiction).

**Pourquoi 100 arbres et pas 1 ?**

Un seul arbre est instable et peut overfitter. En combinant 100 arbres entraînés sur des sous-ensembles **aléatoires** du dataset et des features, on obtient :
- Plus de stabilité
- Moins d'overfitting
- Une meilleure généralisation

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
#                            ↑ 100 arbres      ↑ graine aléatoire (reproductibilité)
```

**Comment il vote ?**
```
Arbre 1   → FRAUDE
Arbre 2   → FRAUDE
Arbre 3   → NORMALE
...
Arbre 100 → FRAUDE

Vote final   → FRAUDE (majorité = 73/100)
Probabilité  → 0.73
```

---

### 3.3 SMOTE — La Formule

SMOTE crée des exemples synthétiques de fraude en **interpolant entre 2 exemples réels** :

```
Nouveau_point = exemple_existant + λ × (voisin - exemple_existant)

où λ est un nombre aléatoire entre 0 et 1
```

Visuellement :
```
Fraude A  ●────────────────●  Fraude B
               ↑
          Nouveau point synthétique
          (à 40% du chemin entre A et B)
```

> **Important :** on ne doit JAMAIS appliquer SMOTE sur le test. On créerait des données fictives mélangées avec de vraies transactions, ce qui fausserait l'évaluation.

---

### 3.4 Métriques — Les Formules

Depuis la matrice de confusion :
```
                  Prédit 0      Prédit 1
Réel 0 (Normal)    TN=56470      FP=10
Réel 1 (Fraude)    FN=5          TP=90
```

```
Precision = TP / (TP + FP) = 90 / (90 + 10) = 0.90
→ "90% de mes alertes sont de vraies fraudes"

Recall    = TP / (TP + FN) = 90 / (90 + 5) = 0.947
→ "Je détecte 94.7% de toutes les fraudes réelles"

F1-Score  = 2 × (Precision × Recall) / (Precision + Recall) = 0.923
→ Moyenne harmonique (punit les mauvais équilibres)

Accuracy  = (TP + TN) / Total = 99.85%
→ TROMPEUSE : même en ratant 5 fraudes, on a 99.85%
```

**AUC-ROC** : aire sous la courbe qui trace TPR vs FPR pour tous les seuils possibles.
- Score = 1.0 → modèle parfait
- Score = 0.5 → aléatoire (modèle inutile)

---

### 3.5 PCA — Réduction de Dimension

```
Données originales : 34 dimensions (impossible à visualiser)
         ↓ PCA
Composante 1 (PC1) : direction qui explique le + de variance
Composante 2 (PC2) : direction perpendiculaire à PC1

Résultat : chaque transaction = un point (x, y) sur un graphique 2D
```

La formule : `X_pca = X_scaled × V` où V est la matrice des vecteurs propres de la covariance.

---

### 3.6 K-Means — La Formule d'Inertie

K-Means minimise la **somme des distances au carré** entre chaque point et son centroïde :

```
Inertie = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

où μᵢ = centroïde du cluster i
   ||x - μᵢ||² = distance euclidienne au carré
```

**L'algorithme en 4 étapes :**
1. Placer K centroïdes aléatoirement
2. Assigner chaque point au centroïde le plus proche
3. Recalculer les centroïdes (moyenne des points du cluster)
4. Répéter jusqu'à convergence

---

### 3.7 Gini Impurity (critère de coupure des arbres)

```
Gini(nœud) = 1 - Σ pᵢ²

Exemple : nœud avec 60% fraudes, 40% normales
Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48

Nœud pur (100% fraudes) : Gini = 1 - (1² + 0²) = 0  ← optimal
```

On choisit la coupure qui **minimise le Gini** dans les nœuds enfants.

---

## 4. Fonctionnement Interne des Modèles

### 4.1 Régression Logistique — Entraînement

```python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sm, y_train_sm)
```

Scikit-learn exécute un algorithme d'optimisation (L-BFGS par défaut) :

```
Itération 1   : w aléatoires → prédictions aléatoires → loss = 0.693
Itération 2   : gradient calculé → w ajustés → loss = 0.45
Itération 3   : recalculer gradient → ajuster w → loss = 0.32
...
Itération 847 : w convergés → loss = 0.08  ✅
```

`max_iter=1000` fixe le nombre maximum d'itérations.

---

### 4.2 Random Forest — Entraînement

```python
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
```

Pour **chaque arbre** (100 fois) :
1. **Bootstrap** : tirer aléatoirement ~63% des lignes avec remplacement
2. **Feature sampling** : à chaque nœud, considérer seulement √34 ≈ 6 features aléatoires
3. **Construire l'arbre** : trouver la meilleure coupure (Gini) jusqu'aux feuilles pures

`n_jobs=-1` = utiliser tous les cœurs du processeur en parallèle.

---

### 4.3 DBSCAN — Fonctionnement

```python
dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters_db = dbscan.fit_predict(X_pca)
```

**L'algorithme :**
1. Pour chaque point, compter ses voisins à distance ≤ `eps`
2. Si un point a ≥ `min_samples` voisins → **point cœur** (dans un cluster)
3. Les points accessibles depuis un point cœur → même cluster
4. Les points non accessibles → **bruit** (label = -1) = anomalies potentielles

Les points "bruit" (label = -1) sont les transactions qui ne ressemblent à aucun groupe.

---

### 4.4 Isolation Forest — Fonctionnement

```python
iso_forest = IsolationForest(contamination=0.002, random_state=42)
pred_if = iso_forest.fit_predict(X_sample_scaled)
```

**L'algorithme :**
1. Construire des arbres en coupant l'espace aléatoirement
2. Les **anomalies** sont faciles à isoler (proches de la racine = peu de coupures)
3. Les **points normaux** sont difficiles à isoler (profonds dans l'arbre)

**Score d'anomalie :**
```
score(x) = 2^( -E(h(x)) / c(n) )

où h(x) = profondeur moyenne dans les arbres
   c(n)  = profondeur moyenne attendue pour n points
```
Score proche de 1 → anomalie. Score proche de 0.5 → normal.

`contamination=0.002` dit à l'algorithme "attends-toi à environ 0.2% d'anomalies".

---

## 5. Explication Ligne par Ligne du Code Clé

### 5.1 SMOTE

```python
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

| Élément | Rôle |
|---------|------|
| `random_state=42` | Graine aléatoire pour reproductibilité |
| `fit_resample` | Analyse la distribution ET génère les nouveaux exemples |
| Sans cette ligne | Modèle qui ignore les fraudes (prédit toujours 0) |

---

### 5.2 StandardScaler

```python
scaler = StandardScaler()
X[cols_a_scaler] = scaler.fit_transform(X[cols_a_scaler])
```

- **Formule :** `x_norm = (x - μ) / σ`
- `fit_transform` = `fit` (calcule μ et σ) + `transform` (applique la normalisation)
- Sans normalisation : `Amount_log` (valeurs 0–9) dominerait `V1` (valeurs -10 à 10)

> **Erreur fréquente :** appliquer `fit_transform` sur le test → il faut uniquement `.transform()` sur le test (utiliser la moyenne/écart-type appris sur le TRAIN).

---

### 5.3 StratifiedKFold

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

| Paramètre | Rôle |
|-----------|------|
| `n_splits=5` | Découpe le train en 5 parties égales |
| `Stratified` | Chaque partie garde la même proportion fraude/normale |
| `shuffle=True` | Mélange avant de découper (évite les biais temporels) |

---

### 5.4 Pipeline

```python
pipeline = ImbPipeline(steps=[
    ('scaler',     StandardScaler()),
    ('smote',      SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(...))
])
```

- Lors du `.fit(X_train, y_train)` : scaler apprend → SMOTE s'applique → RF s'entraîne
- Lors du `.predict(X_test)` : seulement scaler transforme → RF prédit (SMOTE n'est JAMAIS appliqué sur le test)

Sans pipeline, on risque le **data leakage** : appliquer le scaler sur tout le dataset avant le split contaminerait le test avec des informations du train.

---

### 5.5 Feature Engineering

```python
# Feature 1 : log(1 + Amount) — réduit l'asymétrie
df_fe['Amount_log'] = np.log1p(df_fe['Amount'])

# Feature 2 : heure du jour (0-24)
df_fe['Hour'] = (df_fe['Time'] / 3600) % 24

# Feature 3 : indicateur nuit (1 si entre 22h et 6h)
df_fe['IsNight'] = ((df_fe['Hour'] >= 22) | (df_fe['Hour'] <= 6)).astype(int)

# Feature 4 : interaction V1 × V3
df_fe['V1_V3'] = df_fe['V1'] * df_fe['V3']

# Feature 5 : distance de V14 à la médiane
median_v14 = df_fe['V14'].median()
df_fe['V14_ecart'] = np.abs(df_fe['V14'] - median_v14)
```

**Pourquoi `np.log1p` et pas `np.log` ?**
`log(0)` = -∞ (erreur mathématique). `log1p(x) = log(1+x)`, donc si Amount = 0 → log1p(0) = log(1) = 0. Aucun risque.

---

## 6. Visualisation Mentale et Intuition

### Le Random Forest — Le tribunal

Imagine **100 juges** qui examinent le même dossier. Chaque juge a des informations légèrement différentes (bootstrap) et se spécialise sur certains aspects (feature sampling). La décision finale = vote de la majorité. Un seul juge peut se tromper, mais la majorité est beaucoup plus fiable.

### SMOTE — Le professeur qui crée des exercices

Tu as 10 exemples de fraudes et 5 800 exemples normaux. SMOTE, c'est comme un professeur qui dit : "je vais créer des exercices *similaires* à tes 10 exemples de fraude pour que tu en voies plus". Il ne les invente pas de zéro, il les interpole entre des exemples réels.

### PCA — La photo

Tu as un objet en 3D. En prenant une photo (projection 2D), tu perds de l'information, mais tu peux quand même voir la forme générale. PCA choisit **l'angle qui capture le maximum d'information** dans la photo.

### Isolation Forest — Où est Charlie ?

Pour trouver Charlie (l'anomalie) dans une foule uniforme, il est **facile à isoler** : peu de questions nécessaires ("il est à gauche ? non. En haut ? oui. Trouvé !"). Une transaction normale, noyée dans la masse, nécessite beaucoup plus de questions pour l'isoler.

### K-Means — Les centres de gravité

Imagine des boules de couleurs différentes sur une table. K-Means cherche les K points de la table qui **minimisent la distance totale** de chaque boule au point de sa couleur. Les K points finaux sont les centroïdes.

### DBSCAN — Les groupes de personnes

Dans une foule, les gens se regroupent naturellement (familles, amis). DBSCAN trouve ces groupes **denses** automatiquement. Une personne isolée loin de tout groupe = le "bruit" = une transaction inhabituelle.

---

## 7. Bonnes Pratiques et Pièges Courants

### Les 5 erreurs les plus fréquentes

| Erreur | Conséquence | Solution |
|--------|-------------|----------|
| Appliquer SMOTE avant le split | Data leakage, scores artificiellement gonflés | Toujours SMOTE uniquement sur X_train |
| `fit_transform` sur le test | Le scaler "apprend" sur le test | `.transform()` uniquement sur le test, ou Pipeline |
| Optimiser l'Accuracy | Modèle qui ignore les fraudes | Optimiser Recall ou F1-score |
| Ignorer le déséquilibre | Modèle qui prédit toujours 0 | SMOTE + class_weight + métriques adaptées |
| Choisir K arbitrairement | Clusters non représentatifs | Elbow + Silhouette pour valider K |

### Comment améliorer le modèle

**1. GridSearchCV — chercher les meilleurs hyperparamètres automatiquement**
```python
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
gs = GridSearchCV(RandomForestClassifier(), params, scoring='recall', cv=3)
gs.fit(X_train_sm, y_train_sm)
print(gs.best_params_)
```

**2. Ajuster le seuil de décision — augmenter le Recall**
```python
# Au lieu de seuil 0.5 (défaut), utiliser 0.3
y_pred_ajuste = (y_proba_rf > 0.3).astype(int)
```
Baisser le seuil → plus de fraudes détectées (Recall ↑) mais plus de fausses alertes (Precision ↓).

**3. Essayer XGBoost ou LightGBM**
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(scale_pos_weight=577, eval_metric='aucpr')
# scale_pos_weight = nb_normales / nb_fraudes = équivalent class_weight
```

---

## 8. Niveau Avancé — Sous le Capot

### 8.1 Complexité Algorithmique du Random Forest

- Construction d'un arbre : **O(n × p × log(n))** où n = samples, p = features
- Forêt complète : **O(T × n × √p × log(n))** où T = nombre d'arbres
- Pour notre dataset : O(100 × 227 845 × √34 × log(227 845)) ≈ quelques millions d'opérations → d'où les quelques secondes d'entraînement
- `n_jobs=-1` parallélise sur tous les cœurs → divise le temps par le nombre de cœurs

### 8.2 Complexité de l'Isolation Forest

- Score d'anomalie complet :
```
score(x) = 2^( -E(h(x)) / c(n) )

c(n) = 2 × H(n-1) - 2(n-1)/n
H(i) = Σₖ₌₁ⁱ 1/k  (nombre harmonique)
```
- Points normaux : profondeur moyenne ≈ log(n) → score ≈ 0.5
- Anomalies : profondeur moyenne ≈ 1-3 → score ≈ 0.9+

### 8.3 Complexité de DBSCAN

- Naïf : **O(n²)** → trop lent sur 284 807 points
- Avec index spatial (KD-Tree ou Ball-Tree) : **O(n × log(n))**
- C'est pourquoi on l'applique sur le sous-échantillon PCA (10 000 points)

`eps=0.5` : deux points sont "voisins" si leur distance euclidienne dans l'espace PCA 2D est ≤ 0.5. Ce paramètre est critique et dépend de l'échelle des données.

### 8.4 Pourquoi `random_state=42` partout ?

Les algorithmes aléatoires (bootstrap, feature sampling, SMOTE...) donnent des résultats différents à chaque exécution. Fixer `random_state=42` garantit que le code produit **exactement les mêmes résultats** à chaque run, ce qui est indispensable pour :
- Reproduire les résultats d'un collègue
- Déboguer (comparer deux runs)
- Présenter des résultats stables

La valeur 42 n'a pas de signification particulière — c'est une convention de la communauté ML (référence au "Guide du voyageur galactique").

### 8.5 Data Leakage — Le piège silencieux

Le data leakage arrive quand des informations du futur (test set) "contaminent" l'entraînement. Les formes les plus courantes :

```
❌ LEAKAGE : normaliser AVANT le split
scaler.fit_transform(X)          # apprend μ/σ sur TOUT le dataset
X_train, X_test = split(X)       # le test a déjà influencé le scaler

✅ CORRECT : normaliser APRÈS le split
X_train, X_test = split(X)
scaler.fit(X_train)              # apprend μ/σ sur le train seulement
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)   # utilise les μ/σ du train

✅ ENCORE MIEUX : utiliser un Pipeline
```

---

## 9. Résumé Final

### Le projet en 6 étapes

```
Données brutes (284 807 transactions)
        ↓
1. EDA  →  comprendre le déséquilibre, distributions, corrélations
        ↓
2. Feature Engineering  →  Amount_log, V14_ecart, IsNight, V1_V3...
        ↓
3. Split stratifié + SMOTE  →  équilibrer les classes pour l'entraînement
        ↓
4. Modèles supervisés  →  Logistic Regression + Random Forest
        ↓
5. Clustering + Anomalies  →  K-Means + DBSCAN + Isolation Forest + LOF
        ↓
6. Pipeline  →  tout enchaîner proprement sans data leakage
```

### Les 10 points à retenir absolument pour la soutenance

| # | Point clé |
|---|-----------|
| 1 | Le déséquilibre (0.17%) rend l'**accuracy inutile** → utiliser Recall et F1-score |
| 2 | **SMOTE ne s'applique que sur le train** pour ne pas fausser l'évaluation |
| 3 | **StandardScaler apprend sur le train** et s'applique sur le test sans recalculer |
| 4 | **Random Forest = vote de 100 arbres** → plus robuste qu'un seul arbre |
| 5 | La **cross-validation** évalue la robustesse sur plusieurs découpages du train |
| 6 | Le **Pipeline** protège contre le data leakage en ordonnant les étapes |
| 7 | **K-Means cherche des clusters sphériques** → Elbow + Silhouette pour choisir K |
| 8 | **DBSCAN détecte les points "bruit"** = transactions isolées = fraudes potentielles |
| 9 | **Isolation Forest isole facilement les anomalies** (peu de coupures dans l'arbre) |
| 10 | **Recall prioritaire en fraude** : rater une fraude coûte plus cher que bloquer une transaction légitime |

### Tableau de référence rapide — Tous les algorithmes

| Algorithme | Type | Paramètre clé | Ce qu'il optimise |
|------------|------|---------------|-------------------|
| `LogisticRegression` | Supervisé | `max_iter` | Binary Cross-Entropy |
| `RandomForestClassifier` | Supervisé | `n_estimators` | Gini Impurity |
| `SMOTE` | Rééquilibrage | `k_neighbors=5` | Interpolation entre voisins |
| `StandardScaler` | Preprocessing | — | Centre à μ=0, σ=1 |
| `KMeans` | Non supervisé | `n_clusters` | Inertie (distances au carré) |
| `DBSCAN` | Non supervisé | `eps`, `min_samples` | Densité locale |
| `IsolationForest` | Anomalies | `contamination` | Profondeur d'isolation |
| `LocalOutlierFactor` | Anomalies | `n_neighbors`, `contamination` | Densité locale vs voisins |
| `PCA` | Réduction dim. | `n_components` | Variance expliquée |
| `Pipeline` | Architecture | — | Ordre des étapes sans leakage |

---

*Fiche générée pour l'Activité 10 — Fraude et Segmentation — M1 IA DATA — YNOV*
