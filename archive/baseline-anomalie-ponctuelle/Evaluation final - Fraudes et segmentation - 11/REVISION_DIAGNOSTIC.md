# 📋 Fiche de Révision — Diagnostic du Déséquilibre de Classes
### Dataset : Credit Card Fraud Detection

---

## 1. Introduction — Pourquoi le déséquilibre est un problème critique

En détection de fraude, les deux types d'erreur n'ont **pas le même coût** :

| Erreur | Description | Coût réel |
|--------|-------------|-----------|
| **Faux Négatif (FN)** | Une fraude passée inaperçue | 💸 Très élevé — perte financière directe |
| **Faux Positif (FP)** | Une transaction légitime bloquée | 😠 Faible — friction client, annulable |

> **Règle d'or en fraude :** il vaut mieux bloquer 10 transactions légitimes que de laisser passer 1 fraude.

Un dataset avec **0.17% de fraudes** (~492 sur 284 807 transactions) pose un problème fondamental :
les algorithmes classiques "apprennent" à ignorer la minorité car elle pèse trop peu dans l'erreur globale.

---

## 2. Le Paradoxe de l'Accuracy

### Exemple mathématique

Imaginons un modèle **stupide** qui prédit toujours `0` (= "normale") :

```
Total transactions  : 284 807
Transactions normales (classe 0) : 284 315  →  99.83%
Transactions frauduleuses (classe 1) :    492  →   0.17%

Accuracy = Bonnes prédictions / Total
         = 284 315 / 284 807
         = 99.83%  ✅ (en apparence)
```

Pourtant ce modèle **détecte 0 fraude**. Il est totalement inutile.

### Pourquoi l'accuracy ment

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Avec un modèle qui prédit tout à 0 :
  TP = 0    (fraudes détectées)
  TN = 284 315  (normales correctement classées)
  FP = 0
  FN = 492  (fraudes ratées)

Accuracy = (0 + 284 315) / 284 807 = 99.83% 🚨 TROMPEUR
```

> **Conclusion :** Sur un dataset déséquilibré, une accuracy élevée ne signifie rien.
> Il faut absolument utiliser d'autres métriques.

---

## 3. Indicateurs de Détection du Déséquilibre

### 3.1 Analyse du ratio — Value Counts

```python
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True))
```

**Sortie attendue :**
```
Class
0    284315   →  99.83%
1       492   →   0.17%
```

**Règle de seuil :**

| Ratio minorité | Niveau de déséquilibre | Action recommandée |
|---------------|------------------------|-------------------|
| > 20%         | Faible                 | Aucune            |
| 5% – 20%      | Modéré                 | `class_weight`    |
| 1% – 5%       | Fort                   | SMOTE ou ADASYN   |
| < 1%          | Extrême ⚠️             | SMOTE + métriques adaptées |

Notre dataset est dans la catégorie **Extrême** (0.17%).

### 3.2 Visualisation des densités — KDE Plots

Les KDE plots (Kernel Density Estimation) permettent de visualiser la **séparabilité** des deux classes
pour chaque variable. Si les deux courbes se séparent bien → variable discriminante utile.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Exemple sur la variable V14 (très discriminante)
fig, ax = plt.subplots(figsize=(8, 4))
for classe, color, label in [(0, 'steelblue', 'Normale'), (1, 'crimson', 'Fraude')]:
    sns.kdeplot(df[df['Class'] == classe]['V14'],
                ax=ax, color=color, label=label, fill=True, alpha=0.3)
ax.set_title('Distribution de V14 par classe (KDE)')
ax.legend()
plt.show()
```

**Lecture du KDE :**
- Courbes **très séparées** → variable très utile pour le modèle
- Courbes **superposées** → variable peu discriminante, peut être écartée

---

## 4. Métriques de Référence

### 4.1 La Matrice de Confusion

```
                    Prédit : Normale    Prédit : Fraude
Réel : Normale          TN                   FP
Réel : Fraude           FN                   TP
```

- **TP** (True Positive)  : fraude détectée ✅
- **TN** (True Negative)  : normale correctement ignorée ✅
- **FP** (False Positive) : normale bloquée à tort ⚠️
- **FN** (False Negative) : fraude manquée 🚨

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                        display_labels=['Normale', 'Fraude'])
```

### 4.2 Comparaison des métriques

| Métrique | Formule | Ce qu'elle mesure | Priorité en fraude |
|----------|---------|-------------------|--------------------|
| **Accuracy** | (TP+TN) / Total | % de bonnes prédictions globales | ❌ Inutile si déséquilibré |
| **Precision** | TP / (TP+FP) | Parmi les alertes, combien sont vraies | ⚠️ Secondaire |
| **Recall (Rappel)** | TP / (TP+FN) | Parmi les fraudes, combien détectées | ✅ **Prioritaire** |
| **F1-Score** | 2 × (P×R)/(P+R) | Équilibre Precision / Recall | ✅ Bon compromis |
| **AUPRC** | Aire sous courbe Precision-Recall | Performance globale sur la minorité | ✅ **Meilleure métrique** |
| **AUC-ROC** | Aire sous courbe ROC | Capacité de discrimination | ✅ Utile |

### 4.3 Pourquoi l'AUPRC est supérieure à l'AUC-ROC sur données déséquilibrées

- L'**AUC-ROC** intègre les TN, qui sont très nombreux → elle reste artificiellement haute même sur un mauvais modèle.
- L'**AUPRC** ne regarde que la classe minoritaire (fraude) → elle chute immédiatement si le modèle rate des fraudes.

```python
from sklearn.metrics import average_precision_score
auprc = average_precision_score(y_test, y_proba)
print(f'AUPRC : {auprc:.4f}')   # Un modèle aléatoire donne ~0.0017 (= % de fraudes)
```

> **Règle :** Sur un dataset à 0.17% de fraudes, un AUPRC de 0.70+ est excellent.

---

## 5. ✅ Check-list de Diagnostic (avant traitement)

Avant d'appliquer SMOTE ou tout autre technique de rééquilibrage, vérifier systématiquement :

- [ ] **1. Mesurer le ratio** — `value_counts(normalize=True)`. Classer le niveau de déséquilibre (faible / modéré / fort / extrême).

- [ ] **2. Visualiser les distributions** — KDE plots sur les variables les plus corrélées avec la cible. Identifier les variables discriminantes.

- [ ] **3. Choisir ses métriques AVANT d'entraîner** — Décider que le Recall et l'AUPRC sont les métriques prioritaires. Ne jamais regarder l'accuracy seule.

- [ ] **4. Établir un baseline "naïf"** — Calculer les métriques d'un modèle qui prédit toujours la classe majoritaire. Toute amélioration réelle doit dépasser ce seuil.

```python
# Baseline naïf : prédire toujours 0
y_baseline = [0] * len(y_test)
print(f'Recall baseline  : {recall_score(y_test, y_baseline):.2f}')   # → 0.00
print(f'F1 baseline      : {f1_score(y_test, y_baseline):.2f}')        # → 0.00
# Notre modèle doit faire MIEUX que ça
```

---

## Résumé visuel

```
DÉSÉQUILIBRE DÉTECTÉ
       │
       ▼
┌─────────────────────────────────────────────┐
│  1. value_counts()  →  ratio 0.17%          │
│  2. KDE plots       →  séparabilité         │
│  3. Métriques       →  Recall + AUPRC       │
│  4. Baseline naïf   →  référence minimale   │
└─────────────────────────────────────────────┘
       │
       ▼
  TRAITEMENT : SMOTE / class_weight
       │
       ▼
  ÉVALUATION : Matrice de confusion + F1 + AUPRC
```
