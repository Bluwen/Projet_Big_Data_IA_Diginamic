import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tools_for_dataset import load_data, merge_dataset, save_merge, flagnan

def hypoarbre():

    st.set_page_config(
        page_title='Focus NaN'
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

    st.subheader("Le temps passer dans la compagnie influence les clients à partir ou non.")

    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=df_merge, x="tenure", hue = "Churn")
    st.pyplot(fig)

    st.write("Conclusion : Il y a bien une diminution du nombre de churn lorsque le temps passer dans la compagnie augmente.")

    st.subheader("Le statu de Senior Citizen influence positivement les clients qui le possède à rester")

    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=df_merge, x="SeniorCitizen", hue = "Churn")
    st.pyplot(fig)

    df_tmp = df_merge[df_merge["SeniorCitizen"] == 1]
    churn_yes = len(df_tmp[df_tmp["Churn"] == "Yes"])*100/len(df_tmp)

    st.write(f"Pourcentage de personne étant partie avec le statu Sénior : {churn_yes}")

    df_tmp = df_merge[df_merge["SeniorCitizen"] == 0]
    churn_yes = len(df_tmp[df_tmp["Churn"] == "Yes"])*100/len(df_tmp)

    st.write(f"Pourcentage de personne étant partie sans le statu Sénior : {churn_yes}")

    st.write(f"Conclusion : En première conclusion il semblerai que ce soit l'inverse, il faudrai réaliser des analyses complémentaires.")

    
    # fig = plt.figure(figsize=(10, 4))
    # sns.histplot(data=df_merge, x="TotalCharges", hue = "SeniorCitizen")
    # st.pyplot(fig)

    st.subheader("Le coût par mois et total semblent influencer les clients à partir. ")

    st.write("Coûts par mois")
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=df_merge, x="MonthlyCharges", hue = "Churn")
    st.pyplot(fig)

    st.write("Coûts total")
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=df_merge, x="TotalCharges", hue = "Churn")
    st.pyplot(fig)

    st.write("Conclusion : On ne peut pas conclure avec seulement ces analyses")

    st.subheader("Le nombre d’appel téléphonique passer par mois semblent influencer les clients à partir")

    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=df_merge, x="NumCalls", hue = "Churn", bins = 40)
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 4))
    sns.stripplot(data=df_merge, x="NumCalls", y = "MonthlyCharges",hue = "Churn", alpha = 0.5)
    st.pyplot(fig)

    st.write("Conclusion : On ne peut pas conclure avec seulement ces analyses")

    st.subheader("Le type de contrat sur deux ans semble pousser les clients à partir")

    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=df_merge, x="ContractType", hue = "Churn", bins = 40)
    st.pyplot(fig)

    df_tmp = df_merge[df_merge["ContractType"] == "Two year"]
    churn_yes = len(df_tmp[df_tmp["Churn"] == "Yes"])*100/len(df_tmp)

    st.write(f"Pourcentage de personne étant partie qui avait un contrat de deux ans : {churn_yes}")

    df_tmp = df_merge[df_merge["ContractType"] == "One year"]
    churn_yes = len(df_tmp[df_tmp["Churn"] == "Yes"])*100/len(df_tmp)

    st.write(f"Pourcentage de personne étant partie qui avait un contrat de un ans : {churn_yes}")

    df_tmp = df_merge[df_merge["ContractType"] == "Month-to-month"]
    churn_yes = len(df_tmp[df_tmp["Churn"] == "Yes"])*100/len(df_tmp)

    st.write(f"Pourcentage de personne étant partie qui avait un contrat de type mois par mois: {churn_yes}")

    st.write("Les pourcentages sont très proches entre les types de contrats, le contrat de deux ans contient le plus fort pourcentage de churn")