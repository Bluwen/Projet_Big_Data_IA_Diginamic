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


def MeilleureModelParametres():
    st.header("Information sur le model le plus performant acutellement")
    st.write("Pour la préparation des données, il a été choisi de normaliser les données numérique (ce qui n'influence pas trop les arbres de clasiification) et d'encoder par la méthode OneHotEncoder (rajoute des colonnes contenant des 1 ou 0)")
    st.write("Les hyperparmètres suivant ont été fixé selon les observations de la recherche des hyperparamètres: max_depth à 30, max_features à 40 et max_leaf_nodes à 10. L'hyperparmètres : class_weight a été modifié pour permettre un equilibrage pour la classification de la target")

def MeilleureModelVisualisation(X_train, y_train,X_test,y_test, num_cols, cat_cols):
    st.header("Visualisation des resultats du model le plus performant")

    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols),
        ("std_scaler", preprocessing.StandardScaler(), num_cols)], remainder="passthrough"
    )

    pipe = pipeline.Pipeline([
    ("preprocessor", preprocessor),
    ('decision_tree', tree.DecisionTreeClassifier(max_depth = 30, max_features=40, max_leaf_nodes=10, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)

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