import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import io

st.title('Кластеризация клиентов ретейла')
st.write('Пожалуйста, загрузите файл в формате .xlsx для анализа клиентов.')
st.write('Используйте заголовки: CustomerID, TotalSpend, NumTransactions, Frequency, UniqueCategories.')

uploaded_file = st.file_uploader("Выберите файл", type=["xlsx"])

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, **kwargs)
    plt.title('Дендрограмма')
    plt.xlabel("Индекс образца")
    plt.ylabel("Расстояние")
    st.pyplot(fig)
    plt.close(fig)

if uploaded_file is not None:
    try:
        Retail_df = load_data(uploaded_file)
        
        required_columns = ['CustomerID', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        if not all(col in Retail_df.columns for col in required_columns):
            st.error("Ошибка: Недостаточно необходимых колонок в файле")
            st.stop()

        Retail_df = Retail_df.dropna()
        if Retail_df.empty:
            st.error("Ошибка: Нет данных после очистки")
            st.stop()

        numeric_cols = required_columns[1:]
        for col in numeric_cols:
            Retail_df[col] = pd.to_numeric(Retail_df[col], errors='coerce')
        
        if Retail_df[numeric_cols].isnull().any().any():
            st.error("Ошибка: Некорректные значения в данных")
            st.stop()

        X = Retail_df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.sidebar.header("Параметры анализа")
        method = st.sidebar.selectbox(
            "Метод кластеризации",
            ["K-Means", "DBSCAN", "Иерархическая", "GMM"],
            index=0
        )

        if method == "K-Means":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
            
            cluster_range = range(2, 11)
            inertias = []
            silhouettes = []
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                inertias.append(kmeans.inertia_)
                if len(set(labels)) > 1:
                    silhouettes.append(silhouette_score(X_scaled, labels))
                else:
                    silhouettes.append(0)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(cluster_range, inertias, marker='o')
            ax1.set_title('Метод локтя')
            ax1.set_xlabel('Количество кластеров')
            
            ax2.plot(cluster_range, silhouettes, marker='o')
            ax2.set_title('Silhouette Score')
            ax2.set_xlabel('Количество кластеров')
            
            st.pyplot(fig)
            plt.close(fig)

        elif method == "DBSCAN":
            eps = st.sidebar.slider("Радиус (eps)", 0.1, 1.0, 0.5, 0.05)
            min_samples = st.sidebar.slider("Минимальное количество", 1, 10, 5)

        elif method == "Иерархическая":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
            linkage_method = st.sidebar.selectbox("Тип связи", ["ward", "complete", "average", "single"])
            
            # Создание модели с вычислением расстояний
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                compute_distances=True,  # Ключевой параметр
                metric='euclidean' if linkage_method == 'ward' else 'precomputed'
            )
            model.fit(X_scaled)
            
            # Построение дендрограммы с проверкой атрибутов
            try:
                if hasattr(model, 'distances_') and model.distances_ is not None:
                    plot_dendrogram(model, truncate_mode='level', p=3)
                else:
                    st.warning("Дендрограмма недоступна для текущих параметров")
            except Exception as e:
                st.error(f"Ошибка при построении дендрограммы: {str(e)}")
                st.warning("Попробуйте изменить параметры кластеризации")

        elif method == "GMM":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)

        if method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == "Иерархическая":
            model = model
        elif method == "GMM":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        
        labels = model.fit_predict(X_scaled) if method != "GMM" else model.fit(X_scaled).predict(X_scaled)
        Retail_df['Cluster'] = labels

        if method != "DBSCAN":
            try:
                silhouette = silhouette_score(X_scaled, labels)
                st.metric("Silhouette Score", value=f"{silhouette:.2f}")
            except:
                st.warning("Невозможно рассчитать силуэттный коэффициент")

        st.subheader("Результаты кластеризации")
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                Retail_df['TotalSpend'], 
                Retail_df['Frequency'], 
                c=Retail_df['Cluster'], 
                cmap='tab10',
                alpha=0.7
            )
            ax.set_title('Сумма покупок vs Частота')
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                X_pca[:, 0], 
                X_pca[:, 1], 
                c=Retail_df['Cluster'], 
                cmap='tab10',
                alpha=0.7
            )
            ax.set_title('PCA проекция')
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Характеристики кластеров")
        
        cluster_stats = Retail_df.groupby('Cluster')[numeric_cols].median()
        global_median = Retail_df[numeric_cols].median()
        total_clients = len(Retail_df)

        for cluster in sorted(cluster_stats.index):
            cluster_size = len(Retail_df[Retail_df['Cluster'] == cluster])
            cluster_percent = (cluster_size / total_clients) * 100
            cluster_data = cluster_stats.loc[cluster]

            st.markdown(f"#### Кластер {cluster} ({cluster_percent:.1f}%)")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Медианные значения:**")
                for feature in numeric_cols:
                    value = cluster_data[feature]
                    st.markdown(f"""
                    <div style="padding:5px; border-bottom:1px solid #eee">
                        {feature}: <strong>{value:.1f}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Отклонение от среднего:**")
                deviations = (cluster_data - global_median) / global_median * 100
                
                for feature, value in deviations.items():
                    arrow = ""
                    color = "black"
                    if value > 0:
                        arrow = "▲"
                        color = "#2ecc71"
                    elif value < 0:
                        arrow = "▼" 
                        color = "#e74c3c"
                    
                    st.markdown(f"""
                    <div style="padding:5px; border-bottom:1px solid #eee">
                        <span style="color:{color}; font-weight:bold">{arrow} {feature}:</span> 
                        {value:+.1f}%
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            Retail_df.to_excel(writer, index=False)
        output.seek(0)
        
        st.download_button(
            label="Скачать результаты",
            data=output,
            file_name='clusters.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        st.error(f"Ошибка выполнения: {str(e)}")
else:
    st.info("Загрузите файл для начала анализа")