import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tools_for_dataset import load_data, merge_dataset, save_merge, flagnan

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


st.subheader("Les NaN de la colonne TotalCharges désignent-ils seulement les clients qui débutent ? Est-ce qu'il faut exclure les clients qui ont moins d'un mois d'ancienneté ?")

fig = plt.figure(figsize=(10, 4))
sns.histplot(data=df_merge, x="tenure", hue = "TotalCharges_NaN")
st.pyplot(fig)

st.write("Conclusion : les NaN de la colonne TotalCharges ne désigne pas que les clients qui débutent.")



st.subheader("Les NaN de la colonne InternetService désignent-ils seulement les clients qui n'ont pas de contrat de service internet ? (Vérifier qu'il n'y a pas de NaN pour des contrats internet)")

fig = plt.figure(figsize=(10, 4))
sns.histplot(data=df_merge, x="TVPackage", hue = "InternetService_NaN")
st.pyplot(fig)

st.write("Conclusion : les NaN de la colonne InternetService ne désigne pas que les clients qui n'ont pas de contrats internet.")


st.subheader("Les NaN de la colonne TechSupport désignent-ils seulement les clients  qui n'ont pas de contrats de  services internet ?")
fig = plt.figure(figsize=(10, 4))
sns.histplot(data=df_merge, x="InternetService_NaN", hue = "TechSupport_NaN")
st.pyplot(fig)

st.write("Les NaN de la colonne TechSupport ne désignent pas seulement les clients qui n'ont pas de contrats de services internet.")

# Focus valeurs NaN d'une colonne par une autre colonne
list_columns = [col for col in df_merge.columns if col !="customerID" and col != "FeedbackText"]


option_1 = st.selectbox(
    "How would you like to be contacted?",
    df_merge.columns[df_merge.isnull().any()],
    key= "Option1_Nan"
)
option_2 = st.selectbox(
    "How would you like to be contacted?",
    list_columns,
    key= "Option2_Nan"
)

#calcul
st.header("Pourcentage de valeur NaN de la première colonne par rapport à la seconde")
total_option_1 = len(df_merge[option_1])
list_freq = []

#list_option_2_v1= [x if str(x) != 'nan' else "nan" for x in df_merge[option_2].unique()]
list_option_2= [x for x in df_merge[option_2].unique() if str(x) != 'nan' ] 

for op2 in list_option_2:
    df_tmp = df_merge[df_merge[option_2] == op2]
    list_freq.append((df_tmp[option_1].isnull().sum()/total_option_1) * 100)

fig, ax = plt.subplots() 

ax.bar(list_option_2, list_freq)
#ax.tick_params(axis='x', labelrotation=90)
st.pyplot(fig)


#
st.header("Représentation de la distribution de la seconde colonne par rapport à la première")
option_1_h = st.selectbox(
    "How would you like to be contacted?",
    list_columns,
    key= "Option1"
)
option_2_h = st.selectbox(
    "How would you like to be contacted?",
    list_columns,
    key= "Option2"
)

fig = plt.figure(figsize=(10, 4))
sns.histplot(data=df_merge, x=option_1_h, hue= option_2_h, bins = 30)
#fig.set(title = selection)
st.pyplot(fig)