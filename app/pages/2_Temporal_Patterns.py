import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Temporal Patterns", layout="wide")
st.title("Temporal Pattern Analysis")

st.markdown("""
We used **K-Means Clustering** on hour, day of week, and month to find distinct
time-based crime patterns. This page shows both the raw temporal trends and the
clusters found by the model.
""")

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(APP_DIR, "data", "crime_data_app.csv"))

df = load_data()

st.markdown("### Crimes by Hour of Day")
hourly_counts = df.groupby("hour").size()
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(hourly_counts.index, hourly_counts.values, color="steelblue")
ax1.set_xlabel("Hour of Day (0-23)")
ax1.set_ylabel("Number of Crimes")
ax1.set_title("Crime Frequency by Hour")
st.pyplot(fig1)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Crimes by Day of Week")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = df["day_of_week"].value_counts().reindex(day_order)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(day_counts.index, day_counts.values, color="coral")
    ax2.set_xticklabels(day_counts.index, rotation=45)
    ax2.set_ylabel("Number of Crimes")
    st.pyplot(fig2)

with col2:
    st.markdown("### Crimes by Season")
    season_counts = df["season"].value_counts()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(season_counts.index, season_counts.values, color="seagreen")
    ax3.set_ylabel("Number of Crimes")
    st.pyplot(fig3)

st.markdown("### Hourly Crime Heatmap (Hour vs Day of Week)")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0).reindex(day_order)
fig4, ax4 = plt.subplots(figsize=(12, 5))
sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax4)
ax4.set_xlabel("Hour of Day")
ax4.set_ylabel("Day of Week")
st.pyplot(fig4)

st.markdown("### Temporal Cluster Profiles")
st.markdown("Each cluster below represents a distinct time-based crime pattern found by K-Means.")

cluster_profile = df.groupby("temporal_cluster").agg(
    avg_hour=("hour", "mean"),
    total_crimes=("primary_type", "count"),
    weekend_pct=("is_weekend", "mean")
).reset_index()
cluster_profile["avg_hour"] = cluster_profile["avg_hour"].round(1)
cluster_profile["weekend_pct"] = (cluster_profile["weekend_pct"] * 100).round(1)
cluster_profile.columns = ["Temporal Cluster", "Average Hour", "Total Crimes", "% Weekend Crimes"]
st.dataframe(cluster_profile, use_container_width=True)