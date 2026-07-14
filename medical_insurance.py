# ============================================================
# PROJET MACHINE LEARNING : PRÉDICTION DES FRAIS MÉDICAUX
# ============================================================
# Objectifs :
# 1. Charger les données
# 2. Nettoyer les données
# 3. Réaliser une analyse exploratoire (EDA)
# 4. Construire une régression linéaire simple
# 5. Construire une régression linéaire multiple
# 6. Séparer les données en Train/Test
# 7. Standardiser les données
# 8. Construire une régression Ridge
# 9. Trouver le meilleur alpha avec GridSearchCV
# ============================================================


# ============================================================
# 1. IMPORTATION DES BIBLIOTHÈQUES
# ============================================================

# Pandas permet de manipuler les données sous forme de tableau
import pandas as pd

# Seaborn et Matplotlib servent à créer des graphiques
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

# Lecture du fichier CSV
# Les données sont stockées dans le DataFrame df
df = pd.read_csv("Medical_insurance.csv")

# Affiche les 5 premières lignes du dataset
# Permet de vérifier que les données ont bien été chargées
print(df.head())


# ============================================================
# 3. EXPLORATION DU DATASET
# ============================================================

# Affiche :
# - nombre de lignes
# - nombre de colonnes
# - types des variables
# - valeurs manquantes éventuelles
print(df.info())


# ============================================================
# 4. RECHERCHE DES VALEURS MANQUANTES
# ============================================================

# Compte le nombre de valeurs manquantes dans chaque colonne
print(df.isnull().sum())

# Si des valeurs manquantes existaient,
# il faudrait les supprimer ou les remplacer.


# ============================================================
# 5. SUPPRESSION DES DOUBLONS
# ============================================================

# Supprime les lignes identiques
# Cela évite que certaines observations influencent
# plusieurs fois le modèle.
df = df.drop_duplicates()

print("Nombre de lignes après suppression des doublons :", len(df))


# ============================================================
# 6. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ============================================================

# ------------------------------------------------------------
# Distribution de la variable cible : charges
# ------------------------------------------------------------

# Histogramme des frais médicaux
sns.histplot(df["charges"], kde=True)

plt.title("Distribution des frais médicaux")
plt.show()


# ------------------------------------------------------------
# Relation entre l'âge et les frais médicaux
# ------------------------------------------------------------

sns.scatterplot(
    x="age",
    y="charges",
    data=df
)

plt.title("Age vs Charges")
plt.show()

# Chaque point représente une personne.
# On cherche à voir si les frais augmentent avec l'âge.


# ------------------------------------------------------------
# Relation entre le BMI et les frais médicaux
# ------------------------------------------------------------

sns.scatterplot(
    x="bmi",
    y="charges",
    data=df
)

plt.title("BMI vs Charges")
plt.show()

# BMI = Body Mass Index (Indice de Masse Corporelle)
# On cherche à voir si les personnes ayant un BMI élevé
# paient davantage de frais médicaux.


# ------------------------------------------------------------
# Impact du tabagisme sur les frais médicaux
# ------------------------------------------------------------

sns.boxplot(
    x="smoker",
    y="charges",
    data=df
)

plt.title("Smoker vs Charges")
plt.show()

# Le boxplot permet de comparer :
# - les fumeurs
# - les non-fumeurs
#
# Généralement, les fumeurs ont des frais médicaux
# beaucoup plus élevés.


# ============================================================
# 7. TRANSFORMATION DES VARIABLES CATÉGORIELLES
# ============================================================

# Les algorithmes de Machine Learning
# ne comprennent que les nombres.

# Exemple :
# male/female --> 1/0
# yes/no --> 1/0

df2 = pd.get_dummies(
    df,
    drop_first=True
)

# drop_first=True évite les colonnes redondantes.

print(df2.head())


# ============================================================
# 8. ANALYSE DES CORRÉLATIONS
# ============================================================

# Calcul de la matrice de corrélation
corr = df2.corr()

# Affiche les corrélations avec la variable cible "charges"
print(
    corr["charges"].sort_values(
        ascending=False
    )
)

# Cela permet d'identifier les variables
# les plus influentes sur les frais médicaux.


# ============================================================
# 9. RÉGRESSION LINÉAIRE SIMPLE
# ============================================================

from sklearn.linear_model import LinearRegression

# Variable explicative
X = df[['age']]

# Variable cible
y = df['charges']

# Création du modèle
lm = LinearRegression()

# Apprentissage du modèle
lm.fit(X, y)

# Score R²
print(
    "R² Régression simple :",
    lm.score(X, y)
)

# Prédictions
pred = lm.predict(X)

# Affichage de la droite de régression
sns.regplot(
    x="age",
    y="charges",
    data=df
)

plt.title("Régression Linéaire : Age -> Charges")
plt.show()

# Cette droite représente :
# charges = a × age + b


# ============================================================
# 10. RÉGRESSION LINÉAIRE MULTIPLE
# ============================================================

# Conversion des variables catégorielles
df_ml = pd.get_dummies(
    df,
    drop_first=True
)

# Séparation des variables explicatives
X = df_ml.drop(
    "charges",
    axis=1
)

# Variable cible
y = df_ml["charges"]

# Création du modèle
lm = LinearRegression()

# Entraînement
lm.fit(X, y)

# Évaluation
print(
    "R² Régression Multiple :",
    lm.score(X, y)
)

# Ici le modèle utilise :
# age
# bmi
# children
# smoker
# region
# sex

# Il est généralement plus performant
# que la régression simple.


# ============================================================
# 11. DIVISION TRAIN / TEST
# ============================================================

from sklearn.model_selection import train_test_split

# 80% des données pour l'entraînement
# 20% des données pour le test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1
)

print("Taille Train :", X_train.shape)
print("Taille Test :", X_test.shape)

# Pourquoi ?
#
# Si on teste sur les données d'entraînement,
# le score sera souvent trop optimiste.
#
# Les données de test permettent de vérifier
# si le modèle généralise bien.


# ============================================================
# 12. STANDARDISATION DES DONNÉES
# ============================================================

from sklearn.preprocessing import StandardScaler

# Création du scaler
scaler = StandardScaler()

# Calcul de la moyenne et de l'écart-type
# sur les données d'entraînement uniquement
X_train = scaler.fit_transform(X_train)

# Application de la même transformation
# sur les données de test
X_test = scaler.transform(X_test)

# Pourquoi ?
#
# Certaines variables ont des échelles différentes.
#
# Exemple :
#
# age = 30
# bmi = 28
# charges = 15000
#
# La standardisation remet toutes les variables
# sur une même échelle.


# ============================================================
# 13. RÉGRESSION RIDGE
# ============================================================

from sklearn.linear_model import Ridge

# alpha contrôle la pénalité appliquée
# aux coefficients

ridge = Ridge(alpha=10)

# Entraînement
ridge.fit(
    X_train,
    y_train
)

# Évaluation sur les données de test
ridge_score = ridge.score(
    X_test,
    y_test
)

print(
    "R² Ridge :",
    ridge_score
)

# Pourquoi Ridge ?
#
# Réduit le risque de surapprentissage
# (Overfitting).
#
# Empêche les coefficients de devenir
# trop grands.


# ============================================================
# 14. RECHERCHE DU MEILLEUR ALPHA
# ============================================================

from sklearn.model_selection import GridSearchCV

# Liste des valeurs à tester
parameters = [
    {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }
]

# Création du modèle Ridge
RR = Ridge()

# Validation croisée
Grid1 = GridSearchCV(
    RR,
    parameters,
    cv=4
)

# Entraînement et recherche du meilleur alpha
Grid1.fit(
    X_train,
    y_train
)

# Affiche le meilleur alpha
print(
    "Meilleur alpha :",
    Grid1.best_params_
)

# Affiche le meilleur score obtenu
print(
    "Meilleur score CV :",
    Grid1.best_score_
)

# GridSearchCV essaie toutes les valeurs :
#
# alpha = 0.01
# alpha = 0.1
# alpha = 1
# alpha = 10
# alpha = 100
#
# Puis sélectionne automatiquement
# celle qui donne les meilleurs résultats.


# ============================================================
# FIN DU PROJET
# ============================================================

# Workflow complet :
#
# Chargement des données
#        ↓
# Nettoyage
#        ↓
# Analyse exploratoire
#        ↓
# Corrélations
#        ↓
# Régression simple
#        ↓
# Régression multiple
#        ↓
# Train/Test Split
#        ↓
# Standardisation
#        ↓
# Régression Ridge
#        ↓
# Optimisation avec GridSearchCV
#
# Objectif final :
# Prédire les frais médicaux (charges)
# avec la meilleure précision possible.