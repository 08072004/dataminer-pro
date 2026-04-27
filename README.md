c ca qu on a fait en tp:🔷 TP 1 : Fouille de données (Data Mining)
🧠 1. Définition

    La fouille de données (Data Mining) = explorer de grandes données pour extraire des informations utiles cachées

    Objectif : transformer des données brutes → connaissances exploitables
    🛠️ 2. Outils utilisés
    Python
    Scikit-learn (SkLearn)

    Pandas
    📊 3. Exploration des données
    👉 Comprendre le dataset :
    Nombre de lignes (instances)
    Nombre de colonnes (attributs)

    Types des attributs
    📈 4. Statistiques importantes
    ✔️ Les 5 nombres clés
    Minimum
    Maximum
    Médiane
    Q1 (1er quartile)

    Q3 (3e quartile)
    👉 Sert à comprendre la distribution des données
    ✔️ Statistiques descriptives
    Moyenne
    Médiane

    Mode
    📉 5. Visualisation
    Boxplot → voir dispersion + valeurs aberrantes

    Scatter plot → voir relation entre variables
    ⚠️ 6. Prétraitement des données
    Très important en data mining :
    🔹 Valeurs manquantes
    Détection

    Remplacement (ex : moyenne)
    🔹 Normalisation
    Min-Max → valeurs entre 0 et 1

    Z-score → centrage (moyenne = 0, écart-type = 1)
    🔷 TP 2 : Clustering
    🧠 1. Clustering

    Regrouper les données en groupes similaires (clusters) sans labels
    ⚙️ 2. Prétraitement
    Nettoyage
    Normalisation

    Réduction de dimension (optionnel)
    👉 Essentiel avant tout algorithme
    📌 3. Choix du nombre de clusters (k)
    🔹 Méthode du coude (Elbow)
    On trace l’inertie

    On choisit le point où ça change (le “coude”)
    👉 Inertie = distance entre points et centre du cluster
    🤖 4. Algorithmes
    🔸 K-Means
    Utilise la moyenne (centre) des points

    Objectif : minimiser l’inertie
    🔸 K-Medoids
    Utilise un point réel (médoïde) comme centre

    Plus robuste aux valeurs aberrantes
    📏 5. Évaluation
    ✔️ Coefficient de silhouette

    Mesure :
     
        Compacité des clusters
        Séparation entre clusters 👉 Plus proche de 1 = meilleur clustering 📊 6. Comparaison
    Histogramme des inerties

    Comparer :
     
        K-Means vs K-Medoids ✔️ Analyse
    K-Means → rapide mais sensible au bruit
    K-Medoids → plus robuste mais plus lent

je veux faire ce projet:Mini-Projet : Fouille de Données 1
Développement d’une Interface Fouille de Données
Faculté d’Informatique
Année universitaire 2025-2026
Objectif du Projet
L’objectif de ce mini-projet est de concevoir et réaliser une interface interactive per-
mettant d’automatiser le pipeline complet de la fouille de données : du chargement
des données à l’évaluation de modèles.
1 Description du Travail
L’étudiant doit développer une interface graphique (utilisant de préférence Streamlit,
Tkinter ou PyQt) structurée en trois volets principaux. Cette interface servira de base
de travail réutilisable pour vos futures analyses de données.
2 Structure de l’Interface
2.1 Volet 1 : Prétraitement (Preprocessing)
Cette partie doit permettre de préparer le dataset avant toute analyse à travers diffe-
rents boutons. Cette partie doit comprendre tous les traitement vu en TP :
— Importation : Chargement de fichiers (bases).
— Exploration : Affichage des statistiques descriptives et des types de données.
— Nettoyage : Gestion des valeurs manquantes.
— Normalisation : Mise à l’échelle des données (Min-Max Scaling ou Standardisa-
tion).
— Visualisation : Affichage des Boxplots et des Scatter Plots.
2.2 Volet 2 : Clustering
Ce volet permet l’exploration de structures cachées dans les données :
— Algorithmes : Implémentation de K-Means et du K medoids.
— Paramétrage : Choix du nombre de clusters (coubre d’Elbow)k.
— Évaluation : Calcul du score de Silhouette.
— Visualisation : Projection des clusters en 2D (via PCA si nécessaire).
1
2.3 Volet 3 : Apprentissage Supervisé (Classification)
Ce volet concerne la construction et l’évaluation de modèles :
— Partitionnement : Division du dataset en ensembles d’entraînement et de test.
— Modèles : Intégration de modèles d’apprentissage.
— Évaluation : Affichage de la Matrice de Confusion et calcul des métriques
(Accuracy, P recision, Recall, F 1-score).
3 Livrables et Évaluation
1. Code Source : Code structuré, commenté et fonctionnel.
2. Rapport : Un document PDF expliquant le travail.
3. Démonstration : Une présentation orale de l’outil.
2

on basent sur on afait dasn les tps
7:01 PM
Reading frontend design skill

Je vais créer une interface Streamlit complète pour ton mini-projet. Voici d'abord la structure, puis le code complet :
Created 3 files, ran a command, read a file

Voilà ton mini-projet complet ! Voici comment le lancer :
bash

pip install -r requirements.txt
streamlit run app.py

Ce qui est inclus

Volet 1 — Prétraitement (5 onglets)

    Importation CSV/Excel avec choix séparateur et encodage
    Exploration : 5 nombres clés + moyenne + mode, types des colonnes, taux de manquants
    Nettoyage : imputation (moyenne/médiane/mode/constante) + suppression doublons
    Normalisation : Min-Max et Z-score, avec choix des colonnes
    Visualisation : Boxplot + Scatter Plot interactifs

Volet 2 — Clustering

    Courbe d'Elbow pour choisir k
    K-Means avec projection PCA 2D et centres affichés
    K-Medoids implémenté manuellement (sans bibliothèque externe)
    Comparaison des deux algorithmes : silhouette + inertie sur un graphique

Volet 3 — Classification

    Partitionnement train/test configurable (%, random state, features)
    4 modèles : KNN, Arbre de décision, Naive Bayes, SVM
    Matrice de confusion + rapport complet
    Comparaison automatique des 4 modèles avec barplot

L'interface a un thème sombre façon GitHub/IDE pour faire pro en démo. 🎨
App
PY 
Requirements
TXT 
Readme
Document · MD 

decoupe le code en plusieurs fichier dant chauque fichier contient un type precis
7:23 PM
Reading the main app file

Maintenant je crée la structure découpée en fichiers séparés :
Created 12 files, ran 2 commands, read a file

Voilà le projet découpé en 10 fichiers bien séparés. Voici la structure complète :

dataminer/
├── app.py                  → Routeur principal (point d'entrée)
├── sidebar.py              → Navigation + infos dataset
├── requirements.txt
├── README.md
│
├── utils/                  → Modules transversaux (réutilisables partout)
│   ├── styles.py           → Tout le CSS du thème sombre
│   ├── plots.py            → dark_fig() et dark_fig_multi()
│   └── session.py          → init_session() et get_active_df()
│
└── volets/                 → Un fichier par volet fonctionnel
    ├── preprocessing.py    → Volet 1 : Importation, Exploration, Nettoyage…
    ├── clustering.py       → Volet 2 : Elbow, K-Means, K-Medoids, Silhouette
    └── classification.py  → Volet 3 : Split, KNN/DT/NB/SVM, Métriques

Principe de découpe appliqué :

    utils/ = code générique sans logique métier (CSS, graphiques, état)
    volets/ = chaque fichier = un volet du TP avec sa propre fonction render()
    app.py = orchestre tout, ne contient aucune logique propre, juste du routage
    sidebar.py = navigation isolée, facilement modifiable

Pour lancer : streamlit run app.py depuis le dossier dataminer/.
App
PY 
Sidebar
PY 
Styles
PY 
Plots
PY 
Session
PY 
Preprocessing
PY 
Clustering
PY 
Classification
PY 
Requirements
TXT 
Readme
Document · MD 


Claude is AI and can make mistakes. Please double-check responses.
Readme · MD
🔬 DataMiner Pro — Interface Fouille de Données

Mini-Projet Fouille de Données 1 · Année universitaire 2025-2026
▶️ Lancement
bash

pip install -r requirements.txt
streamlit run app.py

🗂️ Structure du Projet

dataminer/
│
├── app.py                   # Point d'entrée — routeur principal
├── sidebar.py               # Barre latérale de navigation
├── requirements.txt
│
├── utils/                   # Modules utilitaires transversaux
│   ├── __init__.py
│   ├── styles.py            # CSS global (thème sombre)
│   ├── plots.py             # Helpers matplotlib thème sombre
│   └── session.py           # Gestion du session_state Streamlit
│
└── volets/                  # Un fichier par volet fonctionnel
    ├── __init__.py
    ├── preprocessing.py     # Volet 1 : Prétraitement
    ├── clustering.py        # Volet 2 : Clustering
    └── classification.py   # Volet 3 : Classification supervisée

📋 Rôle de chaque fichier
Fichier	Contenu
app.py	Configuration Streamlit, injection CSS, routage entre volets
sidebar.py	Navigation (radio), infos dataset en temps réel
utils/styles.py	CSS complet du thème sombre (variables, composants)
utils/plots.py	dark_fig() et dark_fig_multi() — figures matplotlib stylisées
utils/session.py	init_session(), get_active_df() — état partagé entre volets
volets/preprocessing.py	Importation, Exploration, Nettoyage, Normalisation, Visualisation
volets/clustering.py	Elbow, K-Means, K-Medoids (manuel), Silhouette, Comparaison
volets/classification.py	Train/Test split, KNN/DT/NB/SVM, Matrice de confusion, Métriques
📊 Fonctionnalités
Volet 1 — Prétraitement

    Importation CSV / Excel (séparateur + encodage configurables)
    Exploration : 5 nombres clés, moyenne, mode, types, valeurs manquantes
    Nettoyage : imputation (moyenne / médiane / mode / constante), doublons
    Normalisation : Min-Max et Z-score, colonnes sélectionnables
    Visualisation : Boxplot + Scatter Plot

Volet 2 — Clustering

    Courbe d'Elbow pour choisir k
    K-Means avec distribution + projection PCA 2D
    K-Medoids implémenté manuellement (sans bibliothèque externe)
    Coefficient de Silhouette
    Comparaison K-Means vs K-Medoids (silhouette + inertie)

Volet 3 — Classification Supervisée

    Partitionnement train/test configurable + diagramme camembert
    4 modèles avec hyperparamètres : KNN, Arbre de décision, Naive Bayes, SVM
    Matrice de confusion annotée
    Rapport : Accuracy, Precision, Recall, F1-Score
    Comparaison automatique des 4 modèles avec graphique barres

