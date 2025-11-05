import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from tools_for_dataset import load_data

st.set_page_config(
    page_title='Analyse exploratoire des données clients'
)

DATA_CLIENT = ('data/clients.csv')

data_client = load_data(DATA_CLIENT)

#afficher les données
st.title('Visualisation et exploration des données clients')
st.header('Visualisation des données brutes')
st.write(data_client)

st.header("Informations statistique")
st.write(data_client.describe().T.style.background_gradient().format("{:.3f}"))


st.header('Nombre de NaN')
result_nan = data_client[data_client.columns[data_client.isnull().any()]].isnull().sum()
result_nan = result_nan.to_frame()
result_nan.columns =["NaN Value"]

st.write(result_nan)

st.subheader('Pourcentage de NaN')

list_freq1 = []
for index in result_nan.index :
    list_freq1.append((data_client[index].isnull().sum()/len(data_client[index])) * 100)

fig, ax = plt.subplots() 
ax.bar(result_nan.index, list_freq1)
st.pyplot(fig)

st.header('Distributions univariées')
list_columns = [col for col in data_client.columns if col !="customerID"]
selection = st.segmented_control("Choose a feature", list_columns, selection_mode= "single")

st.write(selection)

fig = plt.figure(figsize=(10, 4))
sns.histplot(data=data_client, x=selection, bins = 30)
#fig.set(title = selection)
st.pyplot(fig)