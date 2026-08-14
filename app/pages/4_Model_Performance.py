import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance and MLflow Experiment Tracking")

st.markdown("""
All clustering and dimensionality reduction experiments were tracked using **MLflow**.
Below is a summary of the algorithms we compared and their evaluation scores.
""")

@st.cache_data
def load_results():
    with open("data/model_results.json") as f:
        return json.load(f)

results = load_results()

st.markdown("### Geographic Clustering Comparison")
geo_df = pd.DataFrame(results["geographic_clustering"]).T.reset_index()
geo_df.columns = ["Algorithm", "Silhouette Score", "Davies-Bouldin Score", "Number of Clusters"]
st.dataframe(geo_df, use_container_width=True)

st.markdown("""
**Why K-Means was selected for deployment:** It achieved the highest silhouette
score (0.40) among the three algorithms, and it produces clean, easy-to-interpret
circular zones which are practical for planning patrol routes.

**Why the silhouette score is below 0.5:** Real crime location data doesn't form
perfectly separated clusters like a textbook example - crime happens across the
whole city with overlapping density, so a score of 0.40 is a realistic and honest
result rather than an inflated one.
""")

st.markdown("### Temporal Clustering")
temporal_df = pd.DataFrame(results["temporal_clustering"]).T.reset_index()
temporal_df.columns = ["Algorithm", "Silhouette Score", "Number of Clusters"]
st.dataframe(temporal_df, use_container_width=True)

st.markdown("### Dimensionality Reduction (PCA)")
pca_df = pd.DataFrame(results["dimensionality_reduction"]).T.reset_index()
pca_df.columns = ["Configuration", "Number of Components", "Variance Explained"]
pca_df["Variance Explained"] = (pca_df["Variance Explained"] * 100).round(1).astype(str) + "%"
st.dataframe(pca_df, use_container_width=True)

st.markdown("""
---
### How to view the full MLflow dashboard
Since Streamlit Cloud cannot run a separate MLflow server, the full interactive
MLflow UI (with all logged parameters and metrics) can be viewed locally by running:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
from the project folder, then opening the link it prints (usually `http://localhost:5000`).
""")
