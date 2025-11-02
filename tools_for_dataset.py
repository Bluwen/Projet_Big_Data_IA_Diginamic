import pandas as pd
import streamlit as st


@st.cache_data
def load_data(data):
    data = pd.read_csv(data)
    return data

@st.cache_data
def merge_dataset(df_client, df_contacts, df_interactions, df_usage):
    df_total = pd.merge(df_client, 
                        df_contacts, 
                        how = "left", 
                        left_on = "customerID", 
                        right_on = "customerID"
                        ).merge(
                            df_interactions, 
                            how = "left", 
                            left_on = "customerID", 
                            right_on = "customerID"
                            ).merge(
                                df_usage, 
                                how = "left", 
                                left_on = "customerID", 
                                right_on = "customerID")
    return df_total

def save_merge(df_total, path = "data/merge.csv"):
    df_total.to_csv(path, index = False)