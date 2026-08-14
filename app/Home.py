import streamlit as st
import pandas as pd
import os

# Get the absolute path to this script's folder, so file loading works
# regardless of what directory the app is launched from (fixes Streamlit
# Cloud's working directory being different from local testing)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="PatrolIQ - Smart Safety Analytics", layout="wide")

st.title("PatrolIQ - Smart Safety Analytics Platform")
st.subheader("Chicago Crime Data Analysis using Unsupervised Machine Learning")

st.markdown("""
This project analyzes Chicago crime records to help identify crime hotspots
and time-based crime patterns using unsupervised machine learning techniques.

**Use the sidebar to navigate between pages:**
- **Geographic Hotspots** - Crime hotspot zones found using K-Means, DBSCAN, and Hierarchical Clustering
- **Temporal Patterns** - When crimes happen most (hour, day, month patterns)
- **Dimensionality Reduction** - PCA and t-SNE visualizations of the crime data
- **Model Performance** - Comparison of clustering algorithms and MLflow experiment results
""")

# Load data just to show a quick dataset summary on the home page
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(APP_DIR, "data", "crime_data_app.csv"))

df = load_data()

st.markdown("### Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Crime Records", f"{len(df):,}")
col2.metric("Crime Types", df["primary_type"].nunique())
col3.metric("Police Districts", df["district"].nunique())
col4.metric("Arrest Rate", f"{df['arrest'].mean() * 100:.1f}%")

st.markdown("### Sample of the Data")
st.dataframe(df.head(10))

st.markdown("""
---
**About this project:** Built as part of the GUVI Data Science course capstone project,
using Python, Pandas, Scikit-learn, MLflow, and Streamlit.
""")