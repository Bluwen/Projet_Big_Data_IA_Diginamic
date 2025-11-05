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
    tree
)


def ameliorationmodelparametres():
    """
    Une fonction permettant d'afficher les informations concernant le processing et les hyperparmètres qui vont être tester pour améliorer le model d'arbre de classification.
    """
    st.header("Information sur l'amélioration du model d'arbre de classification")
    st.write("Pour la préparation des données, il a été choisi de normaliser les données numérique (ce qui n'influence pas trop les arbres de classification) et d'encoder par la méthode OneHotEncoder (rajoute des colonnes contenant des 1 ou 0)")
    st.write("Les hyperparamètres suivant vont être tester : max_depth, max_features et max_leaf_nodes. Ils vont être tester dans un premier temps avec les valeurs suivantes :  max_depth = [30,40,50], max_features = [30,40,50] et max_leaf_nodes = [10,15,20]. L'hyperparamètres : class_weight a été modifié pour permettre un équilibrage pour la classification de la cible")


def ameliorationmodelvisualisation(X_train, y_train, num_cols, cat_cols):
    """
    Elle permet d'entrainer et d'afficher les résultats des hyperparamètres testés d'un arbre de classification

    Args:
        X_train (DataFrame): tableau contenant les données d'entrainements
        y_train (_DataFrame): tableau contenant les targets des données d'entrainements
        num_cals (list): liste contenant les noms des colonnes numériques.
        cat_cols (list): liste contenant les noms des colonnes catégorielles.
    """
    st.header("Visualisation des resultats des améliorations du model")

    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols),
        ("std_scaler", preprocessing.StandardScaler(), num_cols)], remainder="passthrough"
    )

    pipe = pipeline.Pipeline([
    ("preprocessor", preprocessor),
    ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
    ])

    max_depth = [30,40,50]
    max_features = [30,40,50]
    max_leaf_nodes = [10,15,20]

    gridsearch = model_selection.GridSearchCV(
        pipe,
        param_grid={
            'decision_tree__max_depth' : max_depth,
            'decision_tree__max_features' : max_features,
            'decision_tree__max_leaf_nodes' :max_leaf_nodes
        },
        # scoring function
        scoring= 'roc_auc',
        # K-fold cross-validation parameter
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    gridsearch.fit(X_train, y_train)

    st.write(
    pd.DataFrame(gridsearch.cv_results_)
    .sort_values(by="rank_test_score")
    .drop("params", axis=1)
    .style.background_gradient()
    )

    st.write("Après ce premier entrainement nous pouvons en déduire que l'on peut retirer la valeur 20 de l'hyperparamètres max_leaf_nodes.")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
        ])

    max_depth = [30,40,50]
    max_features = [30,40,50]
    max_leaf_nodes = [10,15]

    gridsearch = model_selection.GridSearchCV(
        pipe,
        param_grid={
            'decision_tree__max_depth' : max_depth,
            'decision_tree__max_features' : max_features,
            'decision_tree__max_leaf_nodes' :max_leaf_nodes
        },
        # scoring function
        scoring= 'roc_auc',
        # K-fold cross-validation parameter
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    gridsearch.fit(X_train, y_train)

    st.write(
    pd.DataFrame(gridsearch.cv_results_)
    .sort_values(by="rank_test_score")
    .drop("params", axis=1)
    .style.background_gradient()
    )
    st.write("Ici il est possible de déduire que l'on peut retirer la valeur 15 de l'hyperparamètres max_leaf_nodes.")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
        ])

    max_depth = [30,40,50]
    max_features = [30,40,50]
    max_leaf_nodes = [10]

    gridsearch = model_selection.GridSearchCV(
        pipe,
        param_grid={
            'decision_tree__max_depth' : max_depth,
            'decision_tree__max_features' : max_features,
            'decision_tree__max_leaf_nodes' :max_leaf_nodes
        },
        # scoring function
        scoring= 'roc_auc',
        # K-fold cross-validation parameter
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    gridsearch.fit(X_train, y_train)

    st.write(
    pd.DataFrame(gridsearch.cv_results_)
    .sort_values(by="rank_test_score")
    .drop("params", axis=1)
    .style.background_gradient()
    )
    st.write("Après les résultats de cet entrainement, il sera intéressant de regarder plus en détail entre 30 et 40 pour max_depth (50 n'arrivant qu'en 5ème position)" \
    "et de regarder également plus en détail les valeurs 40 et 50 pour le max_features.")
    st.write( "/!\ Remarque : Les déductions ont été réalisé à un instant t, les résultats afficher peuvent varier après relance de la page.")
    
    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
        ])

    max_depth = [30,40]
    max_features = [40,50]
    max_leaf_nodes = [10]

    gridsearch = model_selection.GridSearchCV(
        pipe,
        param_grid={
            'decision_tree__max_depth' : max_depth,
            'decision_tree__max_features' : max_features,
            'decision_tree__max_leaf_nodes' :max_leaf_nodes
        },
        # scoring function
        scoring= 'roc_auc',
        # K-fold cross-validation parameter
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    gridsearch.fit(X_train, y_train)

    st.write(
    pd.DataFrame(gridsearch.cv_results_)
    .sort_values(by="rank_test_score")
    .drop("params", axis=1)
    .style.background_gradient()
    )
    st.write("A la lecture de ces résultats, nous avons choisi d'essaye des valeurs intermédiaires pour max_depth (35) et pour max_features (45)")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
        ])

    max_depth = [30,35,40]
    max_features = [40,45,50]
    max_leaf_nodes = [10]

    gridsearch = model_selection.GridSearchCV(
        pipe,
        param_grid={
            'decision_tree__max_depth' : max_depth,
            'decision_tree__max_features' : max_features,
            'decision_tree__max_leaf_nodes' :max_leaf_nodes
        },
        # scoring function
        scoring= 'roc_auc',
        # K-fold cross-validation parameter
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    gridsearch.fit(X_train, y_train)

    st.write(
    pd.DataFrame(gridsearch.cv_results_)
    .sort_values(by="rank_test_score")
    .drop("params", axis=1)
    .style.background_gradient()
    )
    st.write("Suite à cet entrainement de model, la meilleure valeur pour le max_depth semble être 30.")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
        ])

    max_depth = [30]
    max_features = [40,45,50]
    max_leaf_nodes = [10]

    gridsearch = model_selection.GridSearchCV(
        pipe,
        param_grid={
            'decision_tree__max_depth' : max_depth,
            'decision_tree__max_features' : max_features,
            'decision_tree__max_leaf_nodes' :max_leaf_nodes
        },
        # scoring function
        scoring= 'roc_auc',
        # K-fold cross-validation parameter
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    gridsearch.fit(X_train, y_train)

    st.write(
    pd.DataFrame(gridsearch.cv_results_)
    .sort_values(by="rank_test_score")
    .drop("params", axis=1)
    .style.background_gradient()
    )
    st.write("En conclusion, suite à tous les entrainements de model, les meilleures valeurs pour les hyperparamètres : 30 pour max_depth, 40 max_features et 10 pour max_leaf_nodes")