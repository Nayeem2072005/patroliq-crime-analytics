import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Dimensionality Reduction", layout="wide")
st.title("Dimensionality Reduction: PCA and t-SNE")

st.markdown("""
We had 15 input features describing each crime (location, time, type, arrest status,
etc). Since that's too many to visualize directly, we used **PCA** to compress
them into fewer components, and **t-SNE** to create a 2D visualization that groups
similar crimes together.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("data/crime_data_app.csv")
    tsne_df = pd.read_csv("data/tsne_app.csv")
    with open("data/model_results.json") as f:
        results = json.load(f)
    return df, tsne_df, results

df, tsne_df, results = load_data()

st.markdown("### PCA Results")
col1, col2 = st.columns(2)
col1.metric("Components Used", results["dimensionality_reduction"]["PCA"]["n_components"])
col2.metric("Total Variance Explained", f"{results['dimensionality_reduction']['PCA']['total_variance_explained']*100:.1f}%")

st.markdown("""
**Note:** Using just 2-3 components only explained about 37-49% of the variance,
since our features cover fairly independent aspects of a crime (location, time,
crime type). To reach 70%+ variance, we needed 6 components. For visualization
below, we use the first 2 components (PC1 and PC2) since that's what can be plotted.
""")

color_by = st.selectbox("Color PCA plot by:", ["geo_cluster_kmeans", "primary_type", "is_weekend"])

fig1, ax1 = plt.subplots(figsize=(9, 6))
if color_by == "primary_type":
    top_types = df["primary_type"].value_counts().head(8).index
    plot_df = df[df["primary_type"].isin(top_types)]
    for crime_type in top_types:
        subset = plot_df[plot_df["primary_type"] == crime_type]
        ax1.scatter(subset["pca_1"], subset["pca_2"], label=crime_type, s=4, alpha=0.5)
    ax1.legend(markerscale=3, fontsize=8)
else:
    scatter = ax1.scatter(df["pca_1"], df["pca_2"], c=df[color_by], cmap="tab10", s=4, alpha=0.5)
    plt.colorbar(scatter, ax=ax1, label=color_by)

ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.set_title(f"PCA 2D Visualization (colored by {color_by})")
st.pyplot(fig1)

st.markdown("### t-SNE Visualization")
st.markdown("""
t-SNE was run on a random sample of 5,000 records (running it on all 100,000 rows
is very slow), just for visualization purposes.
""")

tsne_color_by = st.selectbox("Color t-SNE plot by:", ["geo_cluster_kmeans", "primary_type", "is_weekend"], key="tsne_color")

fig2, ax2 = plt.subplots(figsize=(9, 6))
if tsne_color_by == "primary_type":
    top_types = tsne_df["primary_type"].value_counts().head(8).index
    plot_df = tsne_df[tsne_df["primary_type"].isin(top_types)]
    for crime_type in top_types:
        subset = plot_df[plot_df["primary_type"] == crime_type]
        ax2.scatter(subset["tsne_1"], subset["tsne_2"], label=crime_type, s=8, alpha=0.6)
    ax2.legend(markerscale=2, fontsize=8)
else:
    scatter2 = ax2.scatter(tsne_df["tsne_1"], tsne_df["tsne_2"], c=tsne_df[tsne_color_by], cmap="tab10", s=8, alpha=0.6)
    plt.colorbar(scatter2, ax=ax2, label=tsne_color_by)

ax2.set_xlabel("t-SNE Dimension 1")
ax2.set_ylabel("t-SNE Dimension 2")
ax2.set_title(f"t-SNE 2D Visualization (colored by {tsne_color_by})")
st.pyplot(fig2)
