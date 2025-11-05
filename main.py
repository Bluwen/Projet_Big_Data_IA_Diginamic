import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.header("Introduction :")
st.write("Réalisation d’une première analyse exploratoire et d’un premier modèle d’arbre de classification pour le client Telcox.")
st.write("Telcox souhaite pouvoir détecter les clients sui sur le point de résilier leur contrat et d’avoir des recommandations sur les stratégies marketing à développer pour diminuer les résiliations.")
st.write("Le client nous a fourni quatre jeux de données : clients.csv, contracts.csv, interactions.csv et usage.csv.")
st.write("Dans un premier temps nous avons réaliser l’analyse exploratoire des différents jeux de données, notamment pour étudier les valeurs manquantes.")
st.write("Dans un second temps nous avons réaliser et entrainer un modèle d’arbre de classification.")
st.write("Enfin après l’étude des résultats de l’arbres nous avons pu formuler nos premières recommandations.")
