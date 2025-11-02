import pandas as pd
import streamlit as st


@st.cache_data
def load_data_client(data_client):
    data = pd.read_csv(data_client)
    return data