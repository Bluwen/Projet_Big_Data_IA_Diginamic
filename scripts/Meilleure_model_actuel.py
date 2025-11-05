import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import (
    preprocessing,
    model_selection,
    pipeline,
    compose,
    metrics,
    tree
)


def meilleuremodelparametres():
    """
    Une fonction permettant d'afficher les informations concernant le processing pour un model d'arbre de classification avec les hyperparamètres améliorés.
    """
    st.header("Information sur le modèle avec les hyperparamètres optimisés")
    st.write("Pour la préparation des données, il a été choisi de normaliser les données numérique (ce qui n'influence pas trop les arbres de classification) et d'encoder par la méthode OneHotEncoder (rajoute des colonnes contenant des 1 ou 0)")
    st.write("Les hyperparamètres suivant ont été fixé selon les observations de l’amélioration du modèle par la recherche des hyperparamètres : max_depth à 30, max_features à 40 et max_leaf_nodes à 10. L'hyperparamètres : class_weight a été modifié pour permettre un équilibrage pour la classification de la cible")

def meilleuremodelvisualisation(X_train, y_train,X_test,y_test, num_cols, cat_cols):
    """
    Elle permet d'entrainer et d'afficher les résultats d'un arbre de classification

    Args:
        X_train (DataFrame): tableau contenant les données d'entrainements
        y_train (_DataFrame): tableau contenant les targets des données d'entrainements
        X_test (DataFrame): tableau contenant les données de test
        y_test (DataFrame): tableau contenant les targets des données de test
        num_cals (list): liste contenant les noms des colonnes numériques.
        cat_cols (list): liste contenant les noms des colonnes catégorielles.
    """



    # Parti permetant de déclarer et d'entrainer le model
    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols),
        ("std_scaler", preprocessing.StandardScaler(), num_cols)], remainder="passthrough"
    )

    pipe = pipeline.Pipeline([
    ("preprocessor", preprocessor),
    ('decision_tree', tree.DecisionTreeClassifier(max_depth = 30, max_features=40, max_leaf_nodes=10, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)

    # Parti permetant de visualiser les résultats de performance du model
    st.header("Visualisation des resultats du model le plus performant")
    st.subheader("Résultats sur le jeu d'entrainement")

    train = metrics.classification_report(y_train,pipe.predict(X_train))
    st.write(train)
    metrics.ConfusionMatrixDisplay.from_predictions(y_train, pipe.predict(X_train), cmap="Blues").figure_.savefig('confusion_matrix_train.png')
    st.image('confusion_matrix_train.png')


    st.subheader("Résultats sur le jeu de test")

    train = metrics.classification_report(y_test,pipe.predict(X_test))
    st.write(train)
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, pipe.predict(X_test), cmap="Blues").figure_.savefig('confusion_matrix_test.png')
    st.image('confusion_matrix_test.png')

    st.subheader("Courbe Precision Recall")
    fig = plt.figure(figsize=(10, 4))
    precision, recall, threshold = (
    metrics.precision_recall_curve(
        y_test, pipe.predict(X_test)
    )
    )
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    st.pyplot(fig)