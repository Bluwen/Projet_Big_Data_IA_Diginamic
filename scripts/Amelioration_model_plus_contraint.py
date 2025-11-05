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

def meilleuremodelvisualisationTest(X_train, y_train,X_test,y_test, num_cols, cat_cols):
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


    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols),
        ("std_scaler", preprocessing.StandardScaler(), num_cols)], remainder="passthrough"
    )

    # Parti permetant de déclarer et d'entrainer le model
    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
        ])

    max_depth = [30,35,40]
    max_features = [10,20,33]
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

    st.write("Après ce premier entrainement nous pouvons en déduire que l'on peut regarder de plus près les valeurs entre 20 et 30 de max_features (33 étant le nombre total de features du jeu de données).")

        # Parti permetant de déclarer et d'entrainer le model
    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
        ])

    max_depth = [30,35,40]
    max_features = [20,25,30]
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

    st.write("Ici il est possible de déduire que l'on peut retirer la valeur 40 de l'hyperparamètres max_depth.")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
    ])

    max_depth = [30,35]
    max_features = [20,25,30]
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

    st.write("Selon les résultats il faudrait regarder entre 30 et 35 pour le max_depth et entre 20, 25 et 30 pour le max_features.")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
    ])
    max_depth = [30,33,35]
    max_features = [20,25,27]
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
    st.write("Après les résultats un zoom va être réaliser sur les valeurs max_depth 30 et 33 et 20 et 25 pour max_features.")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
    ])
    max_depth = [30,33,35]
    max_features = [20,25,27]
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
    st.write("Après les résultats la valeur 25 pour max_features a été selectionnée.")

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ('decision_tree', tree.DecisionTreeClassifier(class_weight="balanced"))
    ])
    max_depth = [30,33,35]
    max_features = [25]
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

    st.write("En conclusion, suite à tous les entrainements de model, les meilleures valeurs pour les hyperparamètres : 33 pour max_depth, 25 max_features et 10 pour max_leaf_nodes")

def meilleuremodelvisualisationTest1(X_train, y_train,X_test,y_test, num_cols, cat_cols):
        # Parti permetant de déclarer et d'entrainer le model
    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols),
        ("std_scaler", preprocessing.StandardScaler(), num_cols)], remainder="passthrough"
    )

    pipe = pipeline.Pipeline([
    ("preprocessor", preprocessor),
    ('decision_tree', tree.DecisionTreeClassifier(max_depth = 33, max_features=25, max_leaf_nodes=10, class_weight="balanced"))
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