import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Geographic Hotspots", layout="wide")
st.title("Geographic Crime Hotspot Clustering")

st.markdown("""
We used **K-Means Clustering** on latitude and longitude to group crimes into
geographic hotspot zones. We also compared this against **DBSCAN** and
**Hierarchical Clustering** (see the Model Performance page for the comparison).
K-Means was selected for this map because it gave the best silhouette score
and produces clean, easy-to-interpret zones for patrol planning.
""")

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(APP_DIR, "data", "crime_data_app.csv"))

df = load_data()

algo_choice = st.radio("Choose clustering algorithm to view:", ["K-Means", "DBSCAN"], horizontal=True)
cluster_col = "geo_cluster_kmeans" if algo_choice == "K-Means" else "geo_cluster_dbscan"

st.markdown("### Crime Hotspot Map")
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(df["longitude"], df["latitude"], c=df[cluster_col], cmap="tab10", s=3, alpha=0.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Crime Hotspot Zones - {algo_choice}")
plt.colorbar(scatter, ax=ax, label="Cluster")
st.pyplot(fig)

st.markdown("### Cluster Summary")
cluster_summary = df.groupby(cluster_col).agg(
    total_crimes=("primary_type", "count"),
    dominant_crime_type=("primary_type", lambda x: x.mode()[0]),
    arrest_rate=("arrest", "mean")
).reset_index()
cluster_summary["arrest_rate"] = (cluster_summary["arrest_rate"] * 100).round(1)
cluster_summary.columns = ["Cluster", "Total Crimes", "Dominant Crime Type", "Arrest Rate (%)"]
st.dataframe(cluster_summary, use_container_width=True)

st.markdown("### Crime Risk Heatmap by District")
district_counts = df.groupby("district").size().reset_index(name="crime_count")
district_counts["risk_level"] = pd.cut(
    district_counts["crime_count"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

fig2, ax2 = plt.subplots(figsize=(10, 4))
colors = district_counts["risk_level"].map({"Low": "green", "Medium": "yellow", "High": "red"})
ax2.bar(district_counts["district"].astype(str), district_counts["crime_count"], color=colors)
ax2.set_xlabel("Police District")
ax2.set_ylabel("Number of Crimes")
ax2.set_title("Crime Count per District (Red = High Risk, Yellow = Medium, Green = Low)")
st.pyplot(fig2)

from pathlib import Path

import pandas as pd

ASSETS = Path(__file__).resolve().parent.parent / "assets"
NEEDED = ["elbow_plot.png", "silhouette_by_k.png", "dendrogram.png", "kmeans_k_sweep.csv", "dbscan_grid.csv"]

st.divider()
st.header("How the clustering settings were checked")

if not all((ASSETS / f).exists() for f in NEEDED):
    st.warning("The extra evaluation files were not found in the app's assets folder.")
else:
    # --- K-Means: elbow and silhouette by k ---
    st.subheader("K-Means: choosing k")
    sweep = pd.read_csv(ASSETS / "kmeans_k_sweep.csv")
    best = sweep.loc[sweep["silhouette"].idxmax()]
    deployed = sweep[sweep["k"] == 7].iloc[0]
    left, right = st.columns(2)
    left.image(str(ASSETS / "elbow_plot.png"), caption="Elbow plot (inertia by k)")
    right.image(str(ASSETS / "silhouette_by_k.png"), caption="Silhouette score by k")
    summary = (
        f"Silhouette is highest at k = {int(best['k'])} ({best['silhouette']:.3f}); "
        f"the deployed k = 7 scores {deployed['silhouette']:.3f} "
        f"(Davies-Bouldin {deployed['davies_bouldin']:.3f}). "
    )
    if (sweep["silhouette"] < 0.5).all():
        summary += "No value of k reaches the brief's 0.5 target."
    else:
        summary += "At least one value of k reaches the brief's 0.5 target."
    st.write(summary)
    with st.expander("Full table (k = 2 to 10)"):
        st.dataframe(sweep.round(3))

    # --- Hierarchical: dendrogram ---
    st.subheader("Hierarchical clustering: dendrogram")
    st.image(
        str(ASSETS / "dendrogram.png"),
        caption="Ward linkage on a 1,500-point sample (last 30 merges shown)",
    )

    # --- DBSCAN: parameter grid ---
    st.subheader("DBSCAN: parameter grid")
    grid = pd.read_csv(ASSETS / "dbscan_grid.csv")
    one_cluster = int((grid["largest_cluster_%_of_clustered"] > 95).sum())
    top = (
        grid.dropna(subset=["silhouette_excl_noise"])
        .sort_values("silhouette_excl_noise", ascending=False)
        .iloc[0]
    )
    st.write(
        f"{len(grid)} settings were tried. In {one_cluster} of them more than 95% of the clustered crimes "
        f"fall into a single cluster. The best silhouette ({top['silhouette_excl_noise']:.3f}, "
        f"eps = {top['eps']}, min_samples = {int(top['min_samples'])}) labels {top['noise_%']}% of crimes "
        f"as noise, so it describes only a small part of the city."
    )
    good = grid[grid["silhouette_excl_noise"] > 0.5]
    if len(good):
        degenerate = good[(good["noise_%"] > 90) | (good["largest_cluster_%_of_clustered"] > 95)]
        if len(degenerate) == len(good):
            st.write(
                f"All {len(good)} settings that score above 0.5 are degenerate: they either label more than 90% "
                f"of crimes as noise or put more than 95% of the clustered crimes into one cluster, so none of "
                f"them gives a usable zoning of the city."
            )
        else:
            st.write(
                f"{len(good)} settings score above 0.5; check the table to see which of them give usable zones."
            )
    st.dataframe(grid.round(3))