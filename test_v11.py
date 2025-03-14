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

# Заголовок приложения
st.title('Кластеризация клиентов ретейла')
st.write('Пожалуйста, загрузите файл в формате .xlsx для анализа клиентов.')
st.write('Используйте заголовки: CustomerID, TotalSpend, NumTransactions, Frequency, UniqueCategories.')

# Загрузка файла
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
        # Загрузка данных
        Retail_df = load_data(uploaded_file)
        
        # Проверка наличия необходимых колонок
        required_columns = ['CustomerID', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        if not all(col in Retail_df.columns for col in required_columns):
            st.error("Ошибка: Недостаточно необходимых колонок в файле")
            st.stop()

        # Очистка данных
        Retail_df = Retail_df.dropna()
        if Retail_df.empty:
            st.error("Ошибка: Нет данных после очистки")
            st.stop()

        # Преобразование числовых колонок
        numeric_cols = required_columns[1:]
        for col in numeric_cols:
            Retail_df[col] = pd.to_numeric(Retail_df[col], errors='coerce')
        
        if Retail_df[numeric_cols].isnull().any().any():
            st.error("Ошибка: Некорректные значения в данных")
            st.stop()

        # Масштабирование данных
        X = Retail_df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Настройки кластеризации
        st.sidebar.header("Параметры анализа")
        method = st.sidebar.selectbox(
            "Метод кластеризации",
            ["K-Means", "DBSCAN", "Иерархическая", "GMM"],
            index=0
        )

        # Параметры методов
        if method == "K-Means":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
        elif method == "DBSCAN":
            eps = st.sidebar.slider("Радиус (eps)", 0.1, 1.0, 0.5, 0.05)
            min_samples = st.sidebar.slider("Минимальное количество", 1, 10, 5)
        elif method == "Иерархическая":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
            linkage_method = st.sidebar.selectbox("Тип связи", ["ward", "complete", "average", "single"])
        elif method == "GMM":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)

        # Оптимизация числа кластеров для K-Means
        if method == "K-Means":
            st.subheader("Оптимизация числа кластеров")
            
            cluster_range = range(2, 11)
            inertias = []  # Для метода локтя
            silhouettes = []  # Для силуэтного коэффициента
            
            for k in cluster_range:
                model_temp = KMeans(n_clusters=k, random_state=42)
                labels_temp = model_temp.fit_predict(X_scaled)
                inertias.append(model_temp.inertia_)
                
                # Расчет силуэтного коэффициента
                if len(np.unique(labels_temp)) > 1:
                    silhouettes.append(silhouette_score(X_scaled, labels_temp))
                else:
                    silhouettes.append(0)
            
            # Создаем две колонки для графиков
            col1, col2 = st.columns(2)
            
            # График метода локтя
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(cluster_range, inertias, marker='o')
                ax1.set_title('Метод локтя')
                ax1.set_xlabel('Количество кластеров')
                ax1.set_ylabel('Инерция')
                st.pyplot(fig1)
                plt.close(fig1)
            
            # График силуэтного коэффициента
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(cluster_range, silhouettes, marker='o', color='orange')
                ax2.set_title('Силуэтный коэффициент')
                ax2.set_xlabel('Количество кластеров')
                ax2.set_ylabel('Силуэтный коэффициент')
                st.pyplot(fig2)
                plt.close(fig2)

        # Оптимизация числа кластеров для иерархической кластеризации и GMM
        elif method in ["Иерархическая", "GMM"]:
            st.subheader("Оптимизация числа кластеров")
            
            cluster_range = range(2, 11)
            silhouettes = []  # Для силуэтного коэффициента
            
            for k in cluster_range:
                if method == "Иерархическая":
                    model_temp = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
                    labels_temp = model_temp.fit_predict(X_scaled)
                elif method == "GMM":
                    model_temp = GaussianMixture(n_components=k, random_state=42)
                    labels_temp = model_temp.fit_predict(X_scaled)
                
                # Расчет силуэтного коэффициента
                if len(np.unique(labels_temp)) > 1:
                    silhouettes.append(silhouette_score(X_scaled, labels_temp))
                else:
                    silhouettes.append(0)
            
            # График силуэтного коэффициента
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cluster_range, silhouettes, marker='o', color='orange')
            ax.set_title('Силуэтный коэффициент')
            ax.set_xlabel('Количество кластеров')
            ax.set_ylabel('Силуэтный коэффициент')
            st.pyplot(fig)
            plt.close(fig)

        # Применение модели
        if method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == "Иерархическая":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        elif method == "GMM":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        
        labels = model.fit_predict(X_scaled) if method != "GMM" else model.fit(X_scaled).predict(X_scaled)
        Retail_df['Cluster'] = labels

        # Оценка качества кластеризации
        if method != "DBSCAN" and len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X_scaled, labels)
            st.metric("Силуэтный коэффициент", value=f"{silhouette:.2f}")
        elif method == "DBSCAN":
            noise_percent = (labels == -1).sum() / len(labels) * 100
            st.metric("% шума", f"{noise_percent:.1f}%")

        # Визуализация
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
            ax.set_xlabel('Сумма покупок (TotalSpend)')  # Подпись оси X
            ax.set_ylabel('Частота (Frequency)')         # Подпись оси Y
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
            ax.set_xlabel('Первая главная компонента (PC1)')  # Подпись оси X
            ax.set_ylabel('Вторая главная компонента (PC2)')  # Подпись оси Y
            st.pyplot(fig)
            plt.close(fig)

        # Анализ кластеров
        st.subheader("Характеристики кластеров")

        # Глобальные средние значения
        global_mean = Retail_df[numeric_cols].mean()
        #st.markdown("#### Глобальные средние значения:")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Глобальные средние значения:**")
            for feature in numeric_cols:
                value = global_mean[feature]
                st.markdown(f"""
                <div style="padding:5px; border-bottom:1px solid #eee">
                    {feature}: <strong>{value:.1f}</strong>
                </div>
                """, unsafe_allow_html=True)


        st.markdown("---")

        # Характеристики каждого кластера
        cluster_stats = Retail_df.groupby('Cluster')[numeric_cols].median()
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
                deviations = (cluster_data - global_mean) / global_mean * 100
                
                for feature, value in deviations.items():
                    arrow = ""
                    color = "black"
                    if value > 0:
                        arrow = "▲"
                        color = "#2ecc71"  # Зеленый
                    elif value < 0:
                        arrow = "▼" 
                        color = "#e74c3c"  # Красный
                    
                    st.markdown(f"""
                    <div style="padding:5px; border-bottom:1px solid #eee">
                        <span style="color:{color}; font-weight:bold">{arrow} {feature}:</span> 
                        {value:+.1f}%
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")

        # Экспорт результатов
        st.subheader("Экспорт данных")
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