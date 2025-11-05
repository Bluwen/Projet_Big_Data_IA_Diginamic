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


def AmeliorationModelParametres():
    st.header("Information sur l'amelioration du models")
    st.write("Pour la préparation des données, il a été choisi de normaliser les données numérique (ce qui n'influence pas trop les arbres de clasiification) et d'encoder par la méthode OneHotEncoder (rajoute des colonnes contenant des 1 ou 0)")
    st.write("Les hyperparmètres suvant vont être tester : max_depth, max_features et max_leaf_nodes. L'hyperparmètres : class_weight a été modifié pour permettre un equilibrage pour la classification de la target")


def AmeliorationModelVisualisation(X_train, y_train,X_test,y_test, num_cols, cat_cols):
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

    st.write("On peut déduire que l'on peut retirer le 20 de max_leaf_nodes.")

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
    st.write("On peut déduire que l'on peut retirer le 15 de max_leaf_nodes.")

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
    st.write("Il serai intéressant de regarder plus en détaille entre 30 et 40 pour max_depth (50 n'arrivant qu'en 5ième position)")
    
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
    st.write("/!\ a faire")

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
    st.write("/!\ a faire")

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
    st.write("En conclusion les meilleurs valeurs pour les paramètres : 30 pour max_depth, 40 max_features et 10 pour max_leaf_nodes")