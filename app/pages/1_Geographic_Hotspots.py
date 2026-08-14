import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    return pd.read_csv("data/crime_data_app.csv")

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
