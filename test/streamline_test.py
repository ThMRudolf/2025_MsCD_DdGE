import pandas as pd

import streamlit as st

import load_data from supporting_fcn

st.write("My first Streamlit app 🎈")



st.header("1. Inspect the data 🔍")

st.write("`st.data_editor` allows us to display AND edit data")
df = load_data()
st.data_editor(df)