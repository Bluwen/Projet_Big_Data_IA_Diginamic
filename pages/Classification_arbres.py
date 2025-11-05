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


from tools_for_dataset import load_data, merge_dataset, save_merge, flagnan
from scripts.First_model import firstmodelvisualisation, firstmodelparametres
from scripts.Amelioration_model import ameliorationmodelparametres, ameliorationmodelvisualisation
from scripts.Meilleure_model_actuel import meilleuremodelparametres, meilleuremodelvisualisation

st.set_page_config(
    page_title='Arbre de classification'
)

try:
    path = "data/merge.csv"
    df_merge = load_data(path)
except FileNotFoundError:
    df_clients = ('data/clients.csv')
    df_clients = load_data(df_clients)

    df_interactions = ('data/interactions.csv')
    df_interactions = load_data(df_interactions)

    df_contracts = ('data/contracts.csv')
    df_contracts = load_data(df_contracts)

    df_usage = ('data/usage.csv')
    df_usage = load_data(df_usage)

    df_contracts = flagnan(df_contracts)
    df_interactions = flagnan(df_interactions)
    df_usage = flagnan(df_usage)
    df_clients = flagnan(df_clients)

    df_merge = merge_dataset(df_clients, df_contracts, df_interactions, df_usage)
    save_merge(df_merge)


tabs_preparation_donnes,tabs_model,tabs_visualisation, tabs_modelisation, tabs_4 = st.tabs(["Preparation des donnees","Preprocessing et Modèle","Visualisations", "Modelisation de l'arbre de classification", "Evaluation"])

with tabs_preparation_donnes:

    st.write("1. Dans un premier temps, les lignes contenant des NaN sont suprimées ainsi que les colonnes 'customerId' 'FeedbackText'. Cette dernière contenant seulement du texte n'est pas forcément pertinant pour un model de classification.")
    
    df_total_nan = df_merge.drop(['Partner_NaN', 'Dependents_NaN',
       'Age_NaN','TotalCharges_NaN',
       'InternetService_NaN','NbContacts_NaN',
       'SatisfactionScore_NaN','AvgDataUsage_GB_NaN', 'TechSupport_NaN','FeedbackText', "customerID"],axis=1)
    
    df_total_nan = df_total_nan.dropna()


    st.write("2. Ensuite la cible est transformé en variable numérique : 1 pour Yes et 0 pour No")

    df_total_nan["Churn"] = df_total_nan["Churn"].map({"Yes":1, "No":0})


    st.write("3. Les colonnes sont séparées selon leurs catégories : numériques ou catégorielles")

    num_cols = [col for col in df_total_nan.columns if df_total_nan[col].dtype !="object"and col != "Churn"]
    # cat_cols = [col for col in df_total_nan.columns if col not in num_cols and col != "customerID" and col != "Churn"]
    cat_cols = [col for col in df_total_nan.columns if col not in num_cols and col != "Churn"]


    st.write("4. Séparation du jeu de données en jeu d'entrainement et de test")

    target = ["Churn"]
    features = [col for col in df_total_nan.columns if col not in target and col != "customerID"]

    X_train, X_test, y_train, y_test = (
        model_selection.train_test_split(
            df_total_nan[features], df_total_nan[target], test_size=0.2, random_state=42
        )
    )

    st.write("Le jeu de donnée a été séparé à 80% dans le jeu d'entrainement et 20% de données dans le test.")

    st.write("Visualisation de la répartition de la traget dans le jeu de données d'entrainement")
    fig = plt.figure(figsize=(10, 4))
    ax = sns.histplot(data=y_train, x="Churn", hue = "Churn")

    # label each bar in histogram
    for p in ax.patches:
        height = p.get_height() # get the height of each bar
        # adding text to each bar
        ax.text(x = p.get_x()+(p.get_width()/2), # x-coordinate position of data label, padded to be in the middle of the bar
        y = height+1.5, # y-coordinate position of data label, padded 0.2 above bar
        s = "{:.0f}".format(height), # data label, formatted to ignore decimals
        ha = "center") # sets horizontal alignment (ha) to center
    st.pyplot(fig)

    st.write("Visualisation de la répartition de la traget dans le jeu de données de test")
    fig = plt.figure(figsize=(10, 4))
    ax = sns.histplot(data=y_test, x="Churn", hue = "Churn")

    # label each bar in histogram
    for p in ax.patches:
        height = p.get_height() # get the height of each bar
        # adding text to each bar
        ax.text(x = p.get_x()+(p.get_width()/2), # x-coordinate position of data label, padded to be in the middle of the bar
        y = height+1.5, # y-coordinate position of data label, padded 0.2 above bar
        s = "{:.0f}".format(height), # data label, formatted to ignore decimals
        ha = "center") # sets horizontal alignment (ha) to center
    st.pyplot(fig)


    st.write("Remarque : dans les deux jeux de données, la target est déséquilibrée ce qui peut rendre difficile l'apprentissage de certains models")

with tabs_model:
    selection = st.segmented_control("Choisissez un modèle", ["Premier modèle", "Recherche des meilleurs hyperparamètres", "Meilleur modèle actuel"], selection_mode= "single", key = "preprossing")
    if selection == "Premier modèle":
        firstmodelparametres()
    if selection == "Recherche des meilleurs hyperparamètres":
        ameliorationmodelparametres()
    if selection == "Meilleur modèle actuel":
        meilleuremodelparametres()

with tabs_visualisation:
    selection = st.segmented_control("Choisissez un modèle", ["Premier modèle", "Recherche des meilleurs hyperparamètres", "Meilleur modèle actuel"], selection_mode= "single", key = "visualisation")
    if selection == "Premier modèle":
        firstmodelvisualisation(X_train, y_train,X_test,y_test, cat_cols)
        st.subheader("Conclusion")
        st.write("Il y a un sur apprentissage des données d'entrainement et le modèle prédit mal sur le jeu de données de test.")
        st.write("L'objectif étant d'identifier les clients qui risques de parties ont souhaité diminuer le nombre de faux négatifs (ici des clients labelisé 0 alors qu'ils sont 1)")
    if selection == "Recherche des meilleurs hyperparamètres":
        ameliorationmodelvisualisation(X_train, y_train,num_cols, cat_cols)
    if selection == "Meilleur modèle actuel":
        meilleuremodelvisualisation(X_train, y_train,X_test,y_test, num_cols, cat_cols)

with tabs_modelisation:
    preprocessor = compose.ColumnTransformer(
        [("encoder", preprocessing.OneHotEncoder(), cat_cols),
        ("std_scaler", preprocessing.StandardScaler(), num_cols)], remainder="passthrough"
    )

    pipe = pipeline.Pipeline([
    ("preprocessor", preprocessor),
    ('decision_tree', tree.DecisionTreeClassifier(max_depth = 30, max_features=40, max_leaf_nodes=10, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)

    st.write(len(X_train.columns))
    st.write(len(pipe[:-1].get_feature_names_out()))

    fig = plt.figure(figsize=(40,30))
    tree.plot_tree(
        pipe[-1],
        max_depth=30,
        feature_names=pipe[:-1].get_feature_names_out(),
        filled=True,
        rounded=True,
        class_names=["No","Yes"],
        fontsize=9
    )
    st.pyplot(fig)
