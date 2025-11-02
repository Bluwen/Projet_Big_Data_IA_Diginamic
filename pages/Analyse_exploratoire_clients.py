import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from load_dataset import load_data_client

DATA_CLIENT = ('data/clients.csv')

data_client = load_data_client(DATA_CLIENT)

#afficher les données
st.subheader('Raw data client')
st.write(data_client)

st.write(data_client.describe().T.style.background_gradient().format("{:.3f}"))

result_nan = data_client[data_client.columns[data_client.isnull().any()]].isnull().sum()
result_nan = result_nan.to_frame()
result_nan.columns =["NaN Value"]

st.write(result_nan)

#Recupterer les collones numériques
list_columns = [col for col in data_client.columns if col !="customerID"]
selection = st.segmented_control("Choose a feature", list_columns, selection_mode= "single")

st.write(selection)

fig = plt.figure(figsize=(10, 4))
sns.histplot(data=data_client, x=selection, bins = 30)
#fig.set(title = selection)
st.pyplot(fig)

#st.bar_chart(data_client)