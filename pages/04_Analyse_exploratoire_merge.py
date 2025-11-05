import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tools_for_dataset import load_data, merge_dataset, save_merge, flagnan

st.set_page_config(
    page_title='Analyse exploratoire du jeu de données merge'
)

try:
    path = "data/merge.csv"
    df_merge = load_data(path)
except NameError:
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

# ----- 

#afficher les données
st.title('Visualisation et exploration du jeu de données interactions')
st.header('Visualisation des données brutes')
st.write(df_merge)

st.header("Informations statistique")
st.write(df_merge.describe().T.style.background_gradient().format("{:.3f}"))


st.header('Nombre de NaN')
result_nan = df_merge[df_merge.columns[df_merge.isnull().any()]].isnull().sum()
result_nan = result_nan.to_frame()
result_nan.columns =["NaN Value"]

st.write(result_nan)

st.subheader('Pourcentage de NaN')

list_freq1 = []
for index in result_nan.index :
    list_freq1.append((df_merge[index].isnull().sum()/len(df_merge[index])) * 100)

fig, ax = plt.subplots() 

ax.bar(result_nan.index, list_freq1)
ax.tick_params(axis='x', labelrotation=90)
st.pyplot(fig)

st.header('Distributions univariées')
list_columns = [col for col in df_merge.columns if col !="customerID" and col != "Churn"]
selection = st.segmented_control("Choose a feature", list_columns, selection_mode= "single")

st.write(selection)

fig = plt.figure(figsize=(10, 4))
sns.histplot(data=df_merge, x=selection, hue = "Churn", bins = 30)
#fig.set(title = selection)
st.pyplot(fig)

#Bivarie
st.header('Distributions bivarié')
pairplot_cols = df_merge.columns

fig = sns.pairplot(df_merge[pairplot_cols], hue="Churn")
st.pyplot(fig)

#Correlation
st.header('Matrice de correlation')

num_cols = [col for col in df_merge.columns if df_merge[col].dtype !="object"]
plt.figure(figsize=(16, 10))

# Compute the correlation matrix
corr = df_merge[num_cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot_kws = {"size" : 6}, annot=True)
st.pyplot(f)