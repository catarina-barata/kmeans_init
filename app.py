import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Clustering Demo: K-Means vs GMM",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Clustering Comparison: K-Means & Gaussian Mixture Models (GMM)")
st.markdown("""
This educational web application compares **K-Means** (Random vs. K-Means++ initialization) and **Gaussian Mixture Models (GMM)** 
using different covariance structures (`spherical`, `diagonal`, and `full`). We use only **Sepal Length** and **Sepal Width** 
from the Iris dataset for clear 2D visualization.
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
# Tabs for Models
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["1. K-Means (Random vs K-Means++)", "2. GMM (Covariance Types)"])

# =============================================================================
# TAB 1: K-MEANS
# =============================================================================
with tab1:
    st.header("K-Means Initialization Comparison")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        k_km = st.slider("Number of Clusters (K):", min_value=2, max_value=6, value=3, key="k_km")
    with col_ctrl2:
        seed_km = st.number_input("Random Seed:", min_value=0, max_value=999, value=42, key="seed_km")

    def run_kmeans(init_type):
        if init_type == 'random':
            # Uniformly random choice of initial centroids
            np.random.seed(seed_km)
            idx = np.random.choice(X.shape[0], k_km, replace=False)
            initial_centroids = X[idx]
            km = KMeans(n_clusters=k_km, init=initial_centroids, n_init=1, random_state=seed_km)
            km.fit(X)
        else:  # k-means++
            # Run 0 iterations (max_iter=1) to extract exact K-Means++ initial centroids
            km_init = KMeans(n_clusters=k_km, init='k-means++', n_init=1, max_iter=1, random_state=seed_km)
            km_init.fit(X)
            initial_centroids = km_init.cluster_centers_

            # Run full convergence starting from those exact K-Means++ centroids
            km = KMeans(n_clusters=k_km, init=initial_centroids, n_init=1, random_state=seed_km)
            km.fit(X)

        return km, initial_centroids

    km_rand, init_rand = run_kmeans('random')
    km_pp, init_pp = run_kmeans('k-means++')

    col1, col2 = st.columns(2)

    def plot_kmeans(km, initial_centroids, title):
        fig, ax = plt.subplots(figsize=(6, 5))
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='Set2')
        
        ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='Set2', edgecolor='k', s=50, alpha=0.8)
        ax.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='yellow', marker='*', s=250, edgecolor='black', linewidth=1.5, label='Initial Centroids', zorder=10)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='x', s=150, linewidth=3, label='Final Centroids', zorder=10)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend(loc='upper right')
        return fig, km.inertia_, km.n_iter_

    with col1:
        fig_r, inertia_r, n_iter_r = plot_kmeans(km_rand, init_rand, "Random Initialization")
        st.pyplot(fig_r)
        st.metric("Final Inertia", f"{inertia_r:.2f}")
        st.metric("Iterations to Converge", f"{n_iter_r}")

    with col2:
        fig_p, inertia_p, n_iter_p = plot_kmeans(km_pp, init_pp, "K-Means++ Initialization")
        st.pyplot(fig_p)
        st.metric("Final Inertia", f"{inertia_p:.2f}")
        st.metric("Iterations to Converge", f"{n_iter_p}")

# =============================================================================
# TAB 2: GMM COVARIANCE TYPES
# =============================================================================
with tab2:
    st.header("Gaussian Mixture Models (GMM) & Covariance Types")
    st.markdown("""
    Unlike K-Means (which hard-assigns points to spherical boundaries), GMM performs **soft clustering** using probability densities 
    and supports flexible ellipse shapes via covariance matrix constraints.
    """)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        k_gmm = st.slider("Number of Components (K):", min_value=2, max_value=6, value=3, key="k_gmm")
    with col_g2:
        cov_type = st.selectbox("Covariance Type:", ["spherical", "diagonal", "full"], key="cov_type")

    # Robust ellipse drawing function handling 1D (spherical/diag) and 2D (full)
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        ax = ax or plt.gca()
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            covariance = np.atleast_1d(covariance)
            if len(covariance) == 1:
                width = height = 2 * np.sqrt(covariance[0])
            else:
                width, height = 2 * np.sqrt(covariance[:2])
            angle = 0

        for k_ell in range(1, 3):
            ellipse = Ellipse(xy=position, width=k_ell*width, height=k_ell*height, angle=angle, **kwargs)
            ax.add_patch(ellipse)

    gmm = GaussianMixture(n_components=k_gmm, covariance_type=cov_type, random_state=42)
    gmm.fit(X)
    labels_gmm = gmm.predict(X)

    fig_g, ax_g = plt.subplots(figsize=(8, 6))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z_gmm = gmm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax_g.contourf(xx, yy, Z_gmm, alpha=0.2, cmap='viridis')
    
    ax_g.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis', s=50, edgecolor='k', alpha=0.8)

    for i in range(gmm.n_components):
        draw_ellipse(gmm.means_[i], gmm.covariances_[i], ax=ax_g, alpha=0.2, color='black')

    ax_g.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=200, edgecolor='black', label='Component Means', zorder=10)
    ax_g.set_title(f"GMM Clustering (covariance_type = '{cov_type}', K = {k_gmm})", fontweight='bold')
    ax_g.set_xlabel(feature_names[0])
    ax_g.set_ylabel(feature_names[1])
    ax_g.legend(loc='upper right')
    
    st.pyplot(fig_g)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("AIC (Akaike Info Criterion)", f"{gmm.aic(X):.2f}")
    col_m2.metric("BIC (Bayesian Info Criterion)", f"{gmm.bic(X):.2f}")
    col_m3.metric("Converged", str(gmm.converged_))
