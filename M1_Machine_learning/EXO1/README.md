# Régression linéaire simple : prédiction des salaires

Premier modèle de Machine Learning : prédire le salaire d'un employé à partir de son nombre d'années d'expérience, avec une régression linéaire simple (scikit-learn).

Projet réalisé en suivant le tutoriel vidéo [Entraîner un modèle de machine learning](https://www.youtube.com/watch/E3HbvAH9W0A), adapté à un dataset plus volumineux et non nettoyé.

## Objectif

- Découper un dataset en données d'entraînement et données d'évaluation
- Entraîner un modèle de régression linéaire
- Prédire sur des données jamais vues par le modèle
- Comparer les valeurs prédites aux valeurs réelles

## Dataset

Source : [Salary Data](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data) (Kaggle, `Salary_Data.csv`)

- 6704 lignes, 6 colonnes
- Variable d'entrée (feature) : `Years of Experience`
- Variable cible : `Salary`

| # | Colonne | Type | Valeurs manquantes |
|---|---|---|---|
| 0 | Age | float64 | 2 |
| 1 | Gender | object | 2 |
| 2 | Education Level | object | 3 |
| 3 | Job Title | object | 2 |
| 4 | Years of Experience | float64 | 3 |
| 5 | Salary | float64 | 5 |

## Stack

- Python 3.12
- pandas, numpy
- matplotlib
- scikit-learn
- Notebook Kaggle

## Étapes du notebook

### 1. Chargement et inspection

Lecture du CSV avec `pd.read_csv`, puis contrôle des dimensions (`df.shape`), des types (`df.info()`) et des valeurs manquantes (`df.isna().sum()`).

### 2. Nettoyage

Suppression des lignes où l'expérience ou le salaire est vide :

```python
df = df.dropna(subset=["Years of Experience", "Salary"])
```

Il reste 6699 lignes.

### 3. Extraction des variables

```python
X = df[["Years of Experience"]].values  # tableau 2D (6699, 1)
y = df["Salary"].values                 # tableau 1D (6699,)
```

### 4. Découpage train / test

80 % des données pour l'entraînement, 20 % pour l'évaluation :

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)
```

| Jeu | Lignes |
|---|---|
| Entraînement | 5359 |
| Évaluation | 1340 |

### 5. Visualisation

Nuage de points expérience / salaire pour vérifier que la relation est globalement linéaire.

### 6. Entraînement

```python
reg = LinearRegression()
reg.fit(X_train, Y_train)
```

### 7. Prédiction et comparaison

```python
Y_pred = reg.predict(X_test)
```

Comparaison des salaires prédits avec les salaires réels du jeu de test, sous forme de tableau puis de graphique (points réels et droite de régression).

## Résultats

Droite apprise par le modèle :

```
salaire = 7079.21 × expérience + 57736.66
```

- Chaque année d'expérience supplémentaire ajoute environ 7 079 au salaire prédit
- Le salaire prédit pour 0 année d'expérience est d'environ 57 737

Exemples de prédictions sur le jeu de test :

| Réel | Prédit | Écart |
|---|---|---|
| 70 000 | 71 895 | +1 895 |
| 198 000 | 135 608 | -62 392 |
| 40 000 | 78 974 | +38 974 |

## Limites

- **Forte dispersion** : pour une même expérience, les salaires réels varient beaucoup (de 60 000 à 180 000 pour 5 ans). Une seule variable ne suffit pas à expliquer le salaire.
- **Relation non linéaire** : les salaires plafonnent vers 190 000 après 20 ans d'expérience, alors que la droite continue de monter. À 34 ans, le modèle prédit environ 298 000 pour un salaire réel d'environ 190 000.
- **Juniors surestimés** : entre 0 et 2 ans, beaucoup de salaires réels sont sous la droite.
- **Valeurs aberrantes** : quelques salaires proches de 0 dans le dataset, probablement des erreurs de saisie.

Le modèle capte la tendance générale, mais reste imprécis sur ce dataset.

## Exécution

1. Créer un notebook sur Kaggle
2. Ajouter le dataset `mohithsairamreddy/salary-data` via **Add Input**
3. Exécuter les cellules de haut en bas (**Run All**)

## Auteur

Loïc — [GitHub : LOIC754](https://github.com/LOIC754)
