# Les métriques d'évaluation — Guide de vulgarisation

**Projet** — Connect AI / AML — Samir KHALDI  
**Public visé** — Tout public, aucun prérequis technique  
**Contexte** — Évaluation d'un modèle de détection de blanchiment d'argent (GNN)

> Ce document explique les quatre métriques utilisées pour évaluer notre modèle :
> **Précision**, **Recall**, **F1-Score** et **PR-AUC**.
> Chaque concept est illustré par des exemples concrets tirés du projet.

---

## Sommaire

1. [Le contexte : pourquoi évaluer un modèle ?](#1-contexte)
2. [La Précision — le modèle vise-t-il juste ?](#2-precision)
3. [Le Recall — le modèle oublie-t-il des fraudes ?](#3-recall)
4. [Le F1-Score — l'équilibre entre les deux](#4-f1)
5. [Le PR-AUC — la note globale](#5-prauc)
6. [Les quatre métriques ensemble](#6-synthese)

---

## 1. Le contexte : pourquoi évaluer un modèle ?

Notre modèle GNN analyse des milliers de transactions financières et doit répondre à une question simple pour chacune :

> *« Cette transaction fait-elle partie d'un schéma de blanchiment d'argent ? »*

Il répond par **oui** ou **non**. Mais il peut se tromper de deux façons :

| Erreur | Ce qui se passe | Conséquence |
|--------|----------------|-------------|
| **Fausse alerte** | Le modèle dit "fraude" sur une transaction légitime | L'équipe de contrôle perd du temps |
| **Fraude ratée** | Le modèle dit "légitime" sur une vraie fraude | Un blanchisseur passe entre les mailles |

Les quatre métriques mesurent ces deux types d'erreurs sous différents angles.

---

## 2. La Précision — le modèle vise-t-il juste ?

### Définition

> **Sur toutes les alertes émises par le modèle, combien sont de vraies fraudes ?**

$$\text{Précision} = \frac{\text{Vraies fraudes détectées}}{\text{Toutes les alertes émises}}$$

### Exemple concret

Le modèle émet **100 alertes** dans la journée aux équipes de contrôle.
Après vérification, seulement **4 sont de vraies fraudes**.

$$\text{Précision} = \frac{4}{100} = 4\%$$

Cela signifie que **96 alertes sur 100 sont de fausses alarmes**. Les analystes perdent leur temps sur des transactions parfaitement légitimes.

### Dans notre projet

| Modèle | Précision (données d'entraînement) | Précision (données inconnues) |
|--------|-----------------------------------|-------------------------------|
| GraphSAGE | ~4 % | 13 % |
| GAT | ~3 % | 11 % |
| **HeteroGNN** | 22 % | **54 %** |

Le HeteroGNN en conditions réelles : **54 alertes sur 100 sont de vraies fraudes** — plus d'une alerte sur deux est exploitable par les équipes.

---

## 3. Le Recall — le modèle oublie-t-il des fraudes ?

### Définition

> **Sur toutes les vraies fraudes présentes dans les données, combien le modèle en a-t-il détectées ?**

$$\text{Recall} = \frac{\text{Vraies fraudes détectées}}{\text{Toutes les vraies fraudes existantes}}$$

### Exemple concret

Dans un jeu de données, il y a **10 vraies fraudes**.
Le modèle en détecte **7**.

$$\text{Recall} = \frac{7}{10} = 70\%$$

Cela signifie que **3 fraudes sont passées inaperçues**.

### Le dilemme Précision / Recall

Ces deux métriques sont en tension permanente. C'est comme régler le volume d'une alarme incendie :

| Réglage | Précision | Recall | Problème |
|---------|-----------|--------|---------|
| Très sensible | Basse | Élevé | Déclenche pour la moindre fumée de cigarette → fausses alertes |
| Peu sensible | Élevée | Bas | Ne déclenche que face aux flammes → rate les incendies débutants |

Il n'existe pas de seuil parfait — le bon réglage dépend du contexte métier.

---

## 4. Le F1-Score — l'équilibre entre les deux

### Le problème de la moyenne classique

Pour combiner Précision et Recall, l'intuition première est de faire une moyenne :

$$\text{Moyenne classique} = \frac{\text{Précision} + \text{Recall}}{2}$$

Mais cette formule est **dangereusement trompeuse**.

Prenons un modèle qui détecte toutes les fraudes sans exception (Recall = 100 %) mais qui étiquette aussi toutes les transactions légitimes comme fraudes (Précision = 2 %) :

$$\text{Moyenne classique} = \frac{2\% + 100\%}{2} = 51\%$$

Un score de 51 % semble "acceptable" — alors que ce modèle est **totalement inutilisable** en production.

### La moyenne harmonique — sensible aux déséquilibres

Le F1-Score utilise la **moyenne harmonique**, qui pénalise sévèrement les déséquilibres :

$$F1 = 2 \times \frac{\text{Précision} \times \text{Recall}}{\text{Précision} + \text{Recall}}$$

Avec le même modèle inutilisable :

$$F1 = 2 \times \frac{0{,}02 \times 1{,}00}{0{,}02 + 1{,}00} = 0{,}04 \rightarrow 4\%$$

Le F1 de **4 %** dit la vérité : ce modèle est quasi-nul.

### L'analogie du cycliste

Un cycliste fait un aller-retour sur une route de montagne :
- **Aller** (montée) : 10 km/h
- **Retour** (descente) : 90 km/h

Quelle est sa vitesse **réelle** sur l'ensemble du trajet ?

| Méthode | Calcul | Résultat |
|---------|--------|----------|
| Moyenne classique | (10 + 90) ÷ 2 | **50 km/h** — faux |
| Moyenne harmonique | (2 × 10 × 90) ÷ (10 + 90) | **18 km/h** — correct |

La moyenne classique dit 50 km/h alors que le cycliste passe **la majorité de son temps** à 10 km/h. La moyenne harmonique en tient compte : si une valeur est très basse, elle tire le résultat vers le bas — c'est exactement ce qu'on veut pour le F1.

> **Règle à retenir** : si Précision **ou** Recall s'effondre, le F1 s'effondre avec lui.

### Dans notre projet

| Modèle | Précision | Recall | F1-Score |
|--------|-----------|--------|----------|
| GraphSAGE | 13 % | 78 % | **23 %** |
| GAT | 11 % | 64 % | **19 %** |
| **HeteroGNN** | **54 %** | **70 %** | **61 %** |

Le F1 de **0,61** confirme que le HeteroGNN n'est pas bon sur une seule métrique : il est **équilibré** sur les deux.

---

## 5. Le PR-AUC — la note globale

### Le problème du seuil fixe

Le F1-Score est calculé à un seuil de décision fixe (généralement 0,5). Mais notre modèle ne dit pas "fraude" ou "légitime" — il donne un **score de probabilité** entre 0 et 1 :

```
Transaction A → score 0,92  → fraude (au-dessus du seuil 0,5)
Transaction B → score 0,55  → fraude (juste au-dessus)
Transaction C → score 0,48  → légitime (juste en dessous)
Transaction D → score 0,03  → légitime
```

Si on déplace le seuil de 0,5 à 0,6, la transaction B change de catégorie. Le F1 change aussi. Qui décide du bon seuil ?

### La courbe Precision-Recall

On calcule la Précision et le Recall **pour chaque seuil possible** (de 0 à 1) et on trace une courbe :

```
Précision
  1,0 ┤╲
      │  ╲          ← Modèle parfait (reste haut même quand recall monte)
  0,5 ┤    ╲____
      │          ╲__
  0,0 ┼────────────── Recall
      0           1,0
```

- Un **bon modèle** maintient une précision élevée même quand on lui demande de trouver plus de fraudes.
- Un **mauvais modèle** voit sa précision s'effondrer dès qu'on monte le recall.

### Le modèle "au hasard"

Un modèle qui choisit les fraudes **au hasard** sans regarder les données aura toujours une précision égale au taux naturel de fraude dans le jeu.

Dans notre jeu LI-Small, **1,78 %** des transactions sont frauduleuses. Un modèle aléatoire a donc une précision de **1,78 %** quel que soit le seuil → sa courbe PR est une **ligne plate à 1,78 %**.

```
Précision
  1,0 ┤
      │
 0,018┤- - - - - - - - - - ← Modèle aléatoire (ligne plate)
  0,0 ┼──────────────────── Recall
```

Tout modèle qui ne dépasse pas cette ligne **n'a aucune valeur** : autant tirer les fraudes au sort.

### L'aire sous la courbe (AUC)

Le PR-AUC mesure l'**aire sous la courbe** Precision-Recall. Plus cette aire est grande, meilleur est le modèle sur **l'ensemble des seuils possibles**.

```
Précision
  1,0 ┤▓▓▓╲
      │▓▓▓▓▓╲
  0,5 ┤▓▓▓▓▓▓▓╲____
      │▓▓▓▓▓▓▓▓▓▓▓▓╲__
  0,0 ┼──────────────────── Recall
          ↑ L'aire grisée = PR-AUC
```

| PR-AUC | Signification |
|--------|--------------|
| **1,0** | Modèle parfait |
| **= taux de fraude** | Équivalent au hasard — modèle inutile |
| Entre les deux | Plus c'est grand, mieux c'est |

### Dans notre projet

| Modèle | PR-AUC (LI-Small) | Fois mieux que le hasard |
|--------|------------------|--------------------------|
| Hasard | ~0,018 | ×1 (référence) |
| GraphSAGE | 0,12 | **×7** |
| HeteroGNN (entraînement) | 0,43 | **×24** |
| **HeteroGNN (données inconnues)** | **0,60** | **×7,5** (taux fraude plus élevé) |

---

## 6. Les quatre métriques ensemble

### Tableau de synthèse

| Métrique | Question posée | Ce qu'elle mesure | Sensible à |
|----------|---------------|------------------|-----------|
| **Précision** | Vise-t-on juste ? | Qualité des alertes | Fausses alertes |
| **Recall** | Oublie-t-on des fraudes ? | Exhaustivité de la détection | Fraudes ratées |
| **F1-Score** | Sommes-nous équilibrés ? | Compromis précision/recall | Déséquilibres |
| **PR-AUC** | Sommes-nous robustes ? | Performance sur tous les seuils | Déséquilibre des classes |

### L'analogie globale

Imaginez un inspecteur des douanes à un aéroport :

- **Précision** : parmi les passagers qu'il fouille, combien transportent vraiment de la drogue ?
- **Recall** : parmi tous les passagers qui transportent de la drogue, combien a-t-il fouillés ?
- **F1-Score** : est-il efficace sans être ni trop méfiant ni trop laxiste ?
- **PR-AUC** : est-il globalement bon quel que soit le niveau de vigilance demandé par sa hiérarchie ce jour-là ?

### Résultats finaux — HeteroGNN sur données inconnues (LI-Medium)

> *« Le modèle HeteroGNN atteint 54 % de précision, 70 % de recall, un F1-Score de 0,61 et un PR-AUC de 0,60 sur un jeu de données totalement inconnu. Concrètement : plus d'une alerte sur deux transmise aux équipes de contrôle est une véritable fraude, et le modèle intercepte 70 % de toutes les fraudes présentes dans le graphe. »*

---

*Document rédigé dans le cadre du projet Connect AI — Master Data & IA — YNOV*  
*Auteur : Samir KHALDI*
