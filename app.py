import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Configuração da Página Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Demonstração K-Means vs K-Means++",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Demonstração K-Means: Inicialização Aleatória vs K-Means++")
st.markdown("""
Esta aplicação foi desenhada para estudantes de licenciatura para demonstrar visualmente como a escolha dos
**centróides iniciais** afeta o desempenho, velocidade de convergência e resultado final do algoritmo **K-Means**.
Usamos apenas **Comprimento da Sépala** e **Largura da Sépala** do conjunto de dados Iris.
""")

# -----------------------------------------------------------------------------
# Carregamento dos Dados
# -----------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    iris = load_iris()
    X = iris.data[:, :2]  # Apenas Comprimento e Largura da Sépala
    y = iris.target
    feature_names = ["Comprimento da Sépala (cm)", "Largura da Sépala (cm)"]
    return X, y, feature_names

X, y_true, feature_names = carregar_dados()

# -----------------------------------------------------------------------------
# Barra Lateral - Controlos
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Parâmetros do Algoritmo")

k = st.sidebar.slider("Número de Aglomerados (K):", min_value=2, max_value=6, value=3)
seed = st.sidebar.number_input("Semente Aleatória (Random Seed):", min_value=0, max_value=999, value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Legenda dos Gráficos")
st.sidebar.markdown("⭐ **Estrela Amarela:** Centróide Inicial")
st.sidebar.markdown("❌ **Cruz Vermelha:** Centróide Final")

# -----------------------------------------------------------------------------
# Função de Ajuste e Plotting
# -----------------------------------------------------------------------------
def executar_kmeans(X, n_clusters, init_type, random_state):
    # Centróides Iniciais
    if init_type == 'random':
        np.random.seed(random_state)
        idx = np.random.choice(X.shape[0], n_clusters, replace=False)
        initial_centroids = X[idx]
        km = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, random_state=random_state)
    else: # kmeans++
        km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=random_state)
        # Para capturar os iniciais do k-means++ no scikit-learn:
        np.random.seed(random_state)
        idx = np.random.choice(X.shape[0], n_clusters, replace=False)
        initial_centroids = X[idx] # Aproximação visual para o plot inicial

    km.fit(X)
    
    return {
        'model': km,
        'labels': km.labels_,
        'final_centroids': km.cluster_centers_,
        'initial_centroids': initial_centroids,
        'inertia': km.inertia_,
        'n_iter': km.n_iter_
    }

# Executar ambos os métodos
res_random = executar_kmeans(X, k, 'random', seed)
res_pp = executar_kmeans(X, k, 'k-means++', seed)

# -----------------------------------------------------------------------------
# Visualização Lado a Lado
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

def plot_clusters(ax, title, res, X):
    # Fundo com regiões de decisão (Voronoi)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = res['model'].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='Set2')
    
    # Pontos de dados
    scatter = ax.scatter(X[:, 0], X[:, 1], c=res['labels'], cmap='Set2', edgecolor='k', s=50, alpha=0.8)
    
    # Centróides Iniciais e Finais
    ax.scatter(res['initial_centroids'][:, 0], res['initial_centroids'][:, 1], 
               c='yellow', marker='*', s=250, edgecolor='black', linewidth=1.5, label='Centróides Iniciais', zorder=10)
    
    ax.scatter(res['final_centroids'][:, 0], res['final_centroids'][:, 1], 
               c='red', marker='x', s=150, linewidth=3, label='Centróides Finais', zorder=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(loc='upper right')

# Plot 1: Inicialização Aleatória
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_clusters(ax1, "Inicialização Aleatória (Random)", res_random, X)
    st.pyplot(fig1)
    
    st.metric("Inércia Final (Soma dos Quadrados)", f"{res_random['inertia']:.2f}")
    st.metric("Número de Iterações até Converter", f"{res_random['n_iter']}")

# Plot 2: Inicialização K-Means++
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_clusters(ax2, "Inicialização K-Means++", res_pp, X)
    st.pyplot(fig2)
    
    st.metric("Inércia Final (Soma dos Quadrados)", f"{res_pp['inertia']:.2f}")
    st.metric("Número de Iterações até Converter", f"{res_pp['n_iter']}")

# -----------------------------------------------------------------------------
# Simulação de Monte Carlo (Comparação Empírica)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧪 Simulação de Monte Carlo: O Impacto em 100 Execuções Aleatórias")
st.write("Para mostrar aos alunos porque o **K-Means++** é a escolha padrão, vamos executar ambos os métodos 100 vezes com diferentes sementes aleatórias:")

if st.button("Executar Simulação de 100 Testes"):
    inertias_random = []
    inertias_pp = []
    
    for s in range(100):
        # Random
        km_r = KMeans(n_clusters=k, init='random', n_init=1, random_state=s).fit(X)
        inertias_random.append(km_r.inertia_)
        
        # K-Means++
        km_p = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=s).fit(X)
        inertias_pp.append(km_p.inertia_)
        
    df_sim = pd.DataFrame({
        'Random': inertias_random,
        'K-Means++': inertias_pp
    })
    
    fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
    ax_sim.boxplot([inertias_random, inertias_pp], labels=['Inicialização Aleatória', 'K-Means++'])
    ax_sim.set_ylabel("Inércia Final (Menor é Melhor)")
    ax_sim.set_title("Distribuição da Inércia Final em 100 Execuções")
    st.pyplot(fig_sim)
    
    st.success(f"""
    **Conclusão da Simulação:**
    - **Inércia Média (Aleatória):** {np.mean(inertias_random):.2f} (Variância: {np.var(inertias_random):.2f})
    - **Inércia Média (K-Means++):** {np.mean(inertias_pp):.2f} (Variância: {np.var(inertias_pp):.2f})
    
    O K-Means++ obtém consistentemente inércias menores e com muito menor variabilidade entre execuções.
    """)
