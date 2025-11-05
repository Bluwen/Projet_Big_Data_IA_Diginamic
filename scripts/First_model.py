import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import (
    preprocessing,
    pipeline,
    compose,
    metrics,
    tree
)


def firstmodelparametres():
    """
    Une fonction permettant d'afficher les informations concernant le processing pour un premier model naif d'arbre de classification.
    """
    st.header("Information sur un premier model naif d'arbre de classification")
    st.write("Pour un premier essai sur un arbre de classification, il a été choisi de ne pas normaliser les données numériques et d'encoder par la méthode OneHotEncoder (rajoute des colonnes contenant des 1 ou 0)")
    st.write("Seule l'hyperparamètres : class_weight a été modifié pour permettre un équilibrage pour la classification de la cible")


def firstmodelvisualisation(X_train, y_train,X_test,y_test, cat_cols):
    """
    Elle permet d'entrainer et d'afficher les résultats d'un arbre de classification

    Args:
        X_train (DataFrame): tableau contenant les données d'entrainements
        y_train (_DataFrame): tableau contenant les targets des données d'entrainements
        X_test (DataFrame): tableau contenant les données de test
        y_test (DataFrame): tableau contenant les targets des données de test
        cat_cols (list): liste contenant les noms des colonnes catégorielles.
    """

    
    # Parti permetant de déclarer et d'entrainer le model
    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols)], remainder="passthrough"
    )
    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('tree', tree.DecisionTreeClassifier(class_weight = "balanced"))
    ])
    pipe.fit(X_train, y_train)

    # Parti permetant de visualiser les résultats de performance du model
    st.header("Visualisation des resultats du premier model")

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