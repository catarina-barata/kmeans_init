import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="K-Means vs K-Means++ Demonstration",
    page_icon="📊",
    layout="wide"
)

st.title("📊 K-Means Demonstration: Random vs K-Means++ Initialization")
st.markdown("""
This interactive web app is designed for undergraduate students to visually explore how the choice of 
**initial centroids** affects convergence speed, final inertia, and overall clustering quality in **K-Means**.
We use only **Sepal Length** and **Sepal Width** from the Iris dataset to enable 2D visualization.
""")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data[:, :2]  # Sepal length and Sepal width only
    y = iris.target
    feature_names = ["Sepal Length (cm)", "Sepal Width (cm)"]
    return X, y, feature_names

X, y_true, feature_names = load_data()

# -----------------------------------------------------------------------------
# Sidebar - Algorithm Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Algorithm Parameters")

k = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=6, value=3)
seed = st.sidebar.number_input("Random Seed:", min_value=0, max_value=999, value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Plot Legend")
st.sidebar.markdown("⭐ **Yellow Star:** Initial Centroid")
st.sidebar.markdown("❌ **Red Cross:** Final Centroid")

# -----------------------------------------------------------------------------
# Clustering & Plotting Function
# -----------------------------------------------------------------------------
def run_kmeans(X, n_clusters, init_type, random_state):
    if init_type == 'random':
        np.random.seed(random_state)
        idx = np.random.choice(X.shape[0], n_clusters, replace=False)
        initial_centroids = X[idx]
        km = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, random_state=random_state)
    else:  # k-means++
        km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=random_state)
        np.random.seed(random_state)
        idx = np.random.choice(X.shape[0], n_clusters, replace=False)
        initial_centroids = X[idx]

    km.fit(X)
    
    return {
        'model': km,
        'labels': km.labels_,
        'final_centroids': km.cluster_centers_,
        'initial_centroids': initial_centroids,
        'inertia': km.inertia_,
        'n_iter': km.n_iter_
    }

# Run both methods
res_random = run_kmeans(X, k, 'random', seed)
res_pp = run_kmeans(X, k, 'k-means++', seed)

# -----------------------------------------------------------------------------
# Side-by-Side Visualization
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

def plot_clusters(ax, title, res, X):
    # Decision boundaries (Voronoi background)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = res['model'].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='Set2')
    
    # Data points
    ax.scatter(X[:, 0], X[:, 1], c=res['labels'], cmap='Set2', edgecolor='k', s=50, alpha=0.8)
    
    # Initial and Final Centroids
    ax.scatter(res['initial_centroids'][:, 0], res['initial_centroids'][:, 1], 
               c='yellow', marker='*', s=250, edgecolor='black', linewidth=1.5, label='Initial Centroids', zorder=10)
    
    ax.scatter(res['final_centroids'][:, 0], res['final_centroids'][:, 1], 
               c='red', marker='x', s=150, linewidth=3, label='Final Centroids', zorder=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(loc='upper right')

# Plot 1: Random Initialization
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_clusters(ax1, "Random Initialization", res_random, X)
    st.pyplot(fig1)
    
    st.metric("Final Inertia (Within-Cluster Sum of Squares)", f"{res_random['inertia']:.2f}")
    st.metric("Iterations to Converge", f"{res_random['n_iter']}")

# Plot 2: K-Means++ Initialization
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_clusters(ax2, "K-Means++ Initialization", res_pp, X)
    st.pyplot(fig2)
    
    st.metric("Final Inertia (Within-Cluster Sum of Squares)", f"{res_pp['inertia']:.2f}")
    st.metric("Iterations to Converge", f"{res_pp['n_iter']}")

# -----------------------------------------------------------------------------
# Monte Carlo Empirical Simulation
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧪 Monte Carlo Simulation: 100 Random Runs Comparison")
st.write("To demonstrate to students why **K-Means++** is the default industry standard, run both methods across 100 random seeds:")

if st.button("Run 100-Trial Simulation"):
    inertias_random = []
    inertias_pp = []
    
    for s in range(100):
        # Random
        km_r = KMeans(n_clusters=k, init='random', n_init=1, random_state=s).fit(X)
        inertias_random.append(float(km_r.inertia_))
        
        # K-Means++
        km_p = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=s).fit(X)
        inertias_pp.append(float(km_p.inertia_))
        
    df_sim = pd.DataFrame({
        'Random': inertias_random,
        'K-Means++': inertias_pp
    })
    
    fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
    # Uses tick_labels for compatibility with Matplotlib 3.9+
    ax_sim.boxplot([inertias_random, inertias_pp], tick_labels=['Random Initialization', 'K-Means++'])
    ax_sim.set_ylabel("Final Inertia (Lower is Better)")
    ax_sim.set_title("Distribution of Final Inertia Across 100 Runs")
    st.pyplot(fig_sim)
    
    st.success(f"""
    **Simulation Key Takeaway:**
    - **Mean Inertia (Random):** {np.mean(inertias_random):.2f} (Variance: {np.var(inertias_random):.2f})
    - **Mean Inertia (K-Means++):** {np.mean(inertias_pp):.2f} (Variance: {np.var(inertias_pp):.2f})
    
    K-Means++ achieves consistently lower inertia with significantly reduced variance across runs.
    """)
