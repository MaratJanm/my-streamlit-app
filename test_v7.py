import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Настройка страницы
st.set_page_config(page_title="Анализ клиентов", layout="wide")
st.title('🎯 Интеллектуальная кластеризация клиентов')
st.write("Загрузите файл с данными клиентов в формате XLSX")

# Загрузка файла
uploaded_file = st.file_uploader("Выберите файл", type=["xlsx"])

# Кэширование данных
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

if uploaded_file is not None:
    try:
        # Загрузка и проверка данных
        Retail_df = load_data(uploaded_file)
        
        required_columns = ['CustomerID', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        if not all(col in Retail_df.columns for col in required_columns):
            st.error("❌ Ошибка: В файле отсутствуют необходимые колонки")
            st.stop()

        Retail_df = Retail_df.dropna()
        if Retail_df.empty:
            st.error("❌ Ошибка: Нет данных после очистки")
            st.stop()

        numeric_cols = ['TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        for col in numeric_cols:
            Retail_df[col] = pd.to_numeric(Retail_df[col], errors='coerce')
        
        if Retail_df[numeric_cols].isnull().any().any():
            st.error("❌ Ошибка: Обнаружены некорректные числовые значения")
            st.stop()

        # Масштабирование данных
        features = numeric_cols
        X = Retail_df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Выбор метода кластеризации
        st.sidebar.header("Настройки кластеризации")
        method = st.sidebar.selectbox(
            "Метод кластеризации",
            ["K-Means", "DBSCAN", "Иерархическая", "Gaussian Mixture"],
            index=0
        )

        # Параметры методов
        if method == "K-Means":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
        elif method == "DBSCAN":
            eps = st.sidebar.slider("EPS (радиус)", 0.1, 1.0, 0.5, 0.05)
            min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)
        elif method == "Иерархическая":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
            linkage = st.sidebar.selectbox("Связь", ["ward", "complete", "average", "single"])
        elif method == "Gaussian Mixture":
            n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
            covariance_type = st.sidebar.selectbox(
                "Тип ковариации", 
                ["full", "tied", "diag", "spherical"]
            )

        # Кластеризация
        if method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == "Иерархическая":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        elif method == "Gaussian Mixture":
            model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42)
        
        if method == "Gaussian Mixture":
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
        
        Retail_df['cluster_label'] = labels

        # Обработка шума для DBSCAN
        if method == "DBSCAN" and -1 in labels:
            noise_count = list(labels).count(-1)
            st.warning(f"Обнаружен шум: {noise_count} объектов")

        # Визуализация
        st.header("📊 Визуализация результатов")
        
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    Retail_df['TotalSpend'], 
                    Retail_df['Frequency'], 
                    c=Retail_df['cluster_label'], 
                    cmap='viridis',
                    alpha=0.7
                )
                ax.set_title('Total Spend vs Frequency')
                ax.set_xlabel('Общая сумма покупок')
                ax.set_ylabel('Частота покупок')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    X_pca[:, 0], 
                    X_pca[:, 1], 
                    c=Retail_df['cluster_label'], 
                    cmap='viridis',
                    alpha=0.7
                )
                ax.set_title('PCA проекция')
                ax.set_xlabel('Главная компонента 1')
                ax.set_ylabel('Главная компонента 2')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)
                plt.close(fig)

        # Анализ кластеров
        st.header("📌 Характеристики кластеров")
        
        feature_descriptions = {
            'TotalSpend': 'Общая сумма покупок',
            'NumTransactions': 'Количество транзакций',
            'Frequency': 'Частота покупок',
            'UniqueCategories': 'Разнообразие категорий'
        }

        global_medians = Retail_df[features].median()

        def generate_cluster_description(cluster_data):
            desc = []
            for feature in features:
                cluster_median = cluster_data[feature].median()
                global_median = global_medians[feature]
                
                diff = cluster_median - global_median
                diff_percent = abs(diff) / global_median * 100
                
                if diff > 0:
                    trend = "▲ Выше среднего"
                    emoji = "🔺"
                else:
                    trend = "▼ Ниже среднего"
                    emoji = "🔻"
                
                if diff_percent > 30:
                    intensity = "значительно"
                    emoji = "🔥 " + emoji
                elif diff_percent > 15:
                    intensity = "умеренно"
                else:
                    intensity = "незначительно"
                    emoji = "➖"
                
                desc.append(
                    f"{emoji} **{feature_descriptions[feature]}**: "
                    f"{trend} ({intensity}, {diff_percent:.1f}%)"
                )
            return "\n\n".join(desc)

        for cluster_id in sorted(Retail_df['cluster_label'].unique()):
            if cluster_id == -1 and method == "DBSCAN":
                continue  # Пропускаем шум
            
            cluster_data = Retail_df[Retail_df['cluster_label'] == cluster_id]
            
            with st.expander(f"Кластер {cluster_id} ({len(cluster_data)} клиентов)", expanded=True):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.metric(
                        label="Доля клиентов",
                        value=f"{len(cluster_data)/len(Retail_df):.1%}"
                    )
                    st.write("**Медианные значения:**")
                    st.dataframe(
                        cluster_data[features].median().to_frame().T.style.format("{:.1f}"),
                        hide_index=True
                    )
                
                with col2:
                    st.write("**Особенности кластера:**")
                    st.markdown(generate_cluster_description(cluster_data))
                
                st.markdown("---")

        # Скачивание результатов
        st.header("📥 Экспорт результатов")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            Retail_df.to_excel(writer, index=False)
        output.seek(0)
        
        st.download_button(
            label="Скачать результаты",
            data=output,
            file_name='clustered_customers.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        st.error(f"⚠️ Ошибка: {str(e)}")
else:
    st.info("ℹ️ Пожалуйста, загрузите файл для начала анализа")