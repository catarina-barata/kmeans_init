import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="K-Means vs K-Means++",
    page_icon="📊",
    layout="wide"
)

st.title("📊 K-Means: Random vs K-Means++ Initialization")
st.markdown("""
This demo compares standard **Random Initialization** against **K-Means++ Initialization** using 
**Sepal Length** and **Sepal Width** from the Iris dataset.
""")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data[:, :2]  # Sepal length & Sepal width
    feature_names = ["Sepal Length (cm)", "Sepal Width (cm)"]
    return X, feature_names

X, feature_names = load_data()

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=6, value=3)
seed = st.sidebar.number_input("Random Seed:", min_value=0, max_value=999, value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Legend")
st.sidebar.markdown("⭐ **Yellow Star:** Initial Centroid")
st.sidebar.markdown("❌ **Red Cross:** Final Centroid")

# -----------------------------------------------------------------------------
# Helper Functions for True Centroid Extraction
# -----------------------------------------------------------------------------
def get_random_kmeans(X, k, seed):
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], k, replace=False)
    initial_centroids = X[idx]
    
    km = KMeans(n_clusters=k, init=initial_centroids, n_init=1, random_state=seed)
    km.fit(X)
    return km, initial_centroids

def get_kmeans_pp(X, k, seed):
    # Step 1: Run 1 step to capture exact K-Means++ initial centroids
    km_init = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=seed)
    km_init.fit(X)
    initial_centroids = km_init.cluster_centers_

    # Step 2: Fit to full convergence using those initial centroids
    km = KMeans(n_clusters=k, init=initial_centroids, n_init=1, random_state=seed)
    km.fit(X)
    return km, initial_centroids

km_random, init_random = get_random_kmeans(X, k, seed)
km_pp, init_pp = get_kmeans_pp(X, k, seed)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_clusters(ax, km, initial_centroids, title):
    # Voronoi Decision Boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='Set2')
    
    # Data points
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='Set2', edgecolor='k', s=50, alpha=0.8)
    
    # Initial Centroids (Yellow Stars)
    ax.scatter(
        initial_centroids[:, 0], initial_centroids[:, 1],
        c='yellow', marker='*', s=250, edgecolor='black', linewidth=1.5,
        label='Initial Centroids', zorder=10
    )
    
    # Final Centroids (Red Crosses)
    ax.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        c='red', marker='x', s=150, linewidth=3,
        label='Final Centroids', zorder=10
    )
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(loc='upper right')

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_clusters(ax1, km_random, init_random, "Random Initialization")
    st.pyplot(fig1)
    st.metric("Final Inertia", f"{km_random.inertia_:.2f}")
    st.metric("Iterations to Converge", f"{km_random.n_iter_}")

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_clusters(ax2, km_pp, init_pp, "K-Means++ Initialization")
    st.pyplot(fig2)
    st.metric("Final Inertia", f"{km_pp.inertia_:.2f}")
    st.metric("Iterations to Converge", f"{km_pp.n_iter_}")

# -----------------------------------------------------------------------------
# Monte Carlo Simulation
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧪 100-Run Simulation (Inertia Distribution)")

if st.button("Run 100 Random Trials"):
    inertias_random = []
    inertias_pp = []
    
    for s in range(100):
        km_r, _ = get_random_kmeans(X, k, s)
        inertias_random.append(km_r.inertia_)
        
        km_p, _ = get_kmeans_pp(X, k, s)
        inertias_pp.append(km_p.inertia_)
        
    fig_sim, ax_sim = plt.subplots(figsize=(8, 4))
    ax_sim.boxplot([inertias_random, inertias_pp], tick_labels=['Random', 'K-Means++'])
    ax_sim.set_ylabel("Final Inertia (Lower is Better)")
    ax_sim.set_title("Inertia Across 100 Random Seeds")
    st.pyplot(fig_sim)
    
    st.success(f"""
    **Results Summary:**
    - **Random Mean Inertia:** {np.mean(inertias_random):.2f} (Variance: {np.var(inertias_random):.2f})
    - **K-Means++ Mean Inertia:** {np.mean(inertias_pp):.2f} (Variance: {np.var(inertias_pp):.2f})
    """)
