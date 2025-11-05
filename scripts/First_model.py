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


def firstModelParametres():
    st.header("Information sur le premier model")
    st.write("Pour un premier essais sur un arbre de classification, il a été choisi de ne pas normaliser les données numérique et d'encoder par la méthode OneHotEncoder (rajoute des colonnes contenant des 1 ou 0)")
    st.write("Seule l'hyperparmètres : class_weight a été modifié pour permettre un equilibrage pour la classification de la target")


def firstModelVisualisation(X_train, y_train,X_test,y_test, cat_cols):

    st.header("Visualisation des resultats du 1er model")
    
    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols)], remainder="passthrough"
    )
    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('tree', tree.DecisionTreeClassifier(class_weight = "balanced"))
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