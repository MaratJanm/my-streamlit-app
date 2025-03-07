import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Настройка страницы
st.set_page_config(page_title="RFM-анализ", layout="wide")
st.title('📊 Кластеризация клиентов ретейла')
st.write("Загрузите файл с данными клиентов в формате XLSX")

# Загрузка файла
uploaded_file = st.file_uploader("Выберите файл", type=["xlsx"], key="file_uploader")

# Кэширование данных
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

if uploaded_file is not None:
    try:
        # Загрузка и проверка данных
        Retail_df = load_data(uploaded_file)
        
        # Проверка обязательных колонок
        required_columns = ['CustomerID', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        if not all(col in Retail_df.columns for col in required_columns):
            st.error("❌ Ошибка: В файле отсутствуют необходимые колонки")
            st.stop()

        # Очистка данных
        Retail_df = Retail_df.dropna()
        if Retail_df.empty:
            st.error("❌ Ошибка: Нет данных после очистки")
            st.stop()

        # Проверка числовых колонок
        numeric_cols = ['TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        for col in numeric_cols:
            Retail_df[col] = pd.to_numeric(Retail_df[col], errors='coerce')
        
        if Retail_df[numeric_cols].isnull().any().any():
            st.error("❌ Ошибка: Обнаружены некорректные числовые значения")
            st.stop()

        # Выбор признаков
        features = numeric_cols
        X = Retail_df[features]

        # Масштабирование данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Подбор оптимального числа кластеров
        with st.expander("🔍 Оптимальное число кластеров"):
            cluster_range = range(2, 11)
            cluster_errors = []
            silhouette_scores = []

            for num_clusters in cluster_range:
                clusters = KMeans(n_clusters=num_clusters, random_state=42)
                clusters.fit(X_scaled)
                cluster_errors.append(clusters.inertia_)
                if len(np.unique(clusters.labels_)) > 1:
                    silhouette_scores.append(silhouette_score(X_scaled, clusters.labels_))
                else:
                    silhouette_scores.append(0)

            # Визуализация
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            
            ax1.plot(cluster_range, cluster_errors, marker='o')
            ax1.set_title('Метод локтя')
            ax1.set_xlabel('Количество кластеров')
            ax1.set_ylabel('Inertia')
            
            ax2.plot(cluster_range, silhouette_scores, marker='o')
            ax2.set_title('Silhouette Score')
            ax2.set_xlabel('Количество кластеров')
            ax2.set_ylabel('Score')
            
            st.pyplot(fig)
            plt.close(fig)

        # Выбор числа кластеров
        n_clusters = st.slider("Выберите количество кластеров", 2, 10, 3, key='n_clusters')

        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        Retail_df['cluster_label'] = kmeans.fit_predict(X_scaled)

        # Визуализация
        st.header("Визуализация кластеров")
        
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                # TotalSpend vs Frequency
                fig1, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    Retail_df['TotalSpend'], 
                    Retail_df['Frequency'], 
                    c=Retail_df['cluster_label'], 
                    cmap='viridis'
                )
                ax.set_title('Total Spend vs Frequency')
                ax.set_xlabel('Общая сумма покупок')
                ax.set_ylabel('Частота покупок')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                # PCA визуализация
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig2, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    X_pca[:, 0], 
                    X_pca[:, 1], 
                    c=Retail_df['cluster_label'], 
                    cmap='viridis'
                )
                ax.set_title('PCA проекция')
                ax.set_xlabel('Главная компонента 1')
                ax.set_ylabel('Главная компонента 2')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig2)
                plt.close(fig2)

        # Анализ кластеров
        st.header("📌 Характеристики кластеров")
        
        # Описания признаков
        feature_descriptions = {
            'TotalSpend': 'Общая сумма покупок',
            'NumTransactions': 'Количество транзакций',
            'Frequency': 'Частота покупок',
            'UniqueCategories': 'Разнообразие категорий'
        }

        # Глобальные медианы
        global_medians = Retail_df[features].median(numeric_only=True)

        # Функция генерации описания
        def generate_cluster_description(cluster_data):
            description = []
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
                
                if abs(diff_percent) > 20:
                    intensity = "значительно"
                    emoji = "❗" + emoji
                elif abs(diff_percent) > 10:
                    intensity = "умеренно"
                else:
                    intensity = "незначительно"
                    emoji = "➖"
                
                description.append(
                    f"{emoji} **{feature_descriptions[feature]}**: "
                    f"{trend} ({intensity}, {diff_percent:.1f}%)"
                )
            return "\n\n".join(description)

        # Отображение кластеров
        for cluster_id in sorted(Retail_df['cluster_label'].unique()):
            cluster_data = Retail_df[Retail_df['cluster_label'] == cluster_id]
            
            with st.expander(f"## Кластер {cluster_id} ({len(cluster_data)} клиентов)", expanded=True):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.metric(
                        label="Доля от всех клиентов",
                        value=f"{len(cluster_data)/len(Retail_df):.1%}"
                    )
                    st.write("**Медианные значения:**")
                    st.dataframe(
                        cluster_data[features].median().to_frame().T.style.format("{:.1f}"),
                        hide_index=True
                    )
                
                with col2:
                    st.write("**Характеристики:**")
                    st.markdown(generate_cluster_description(cluster_data))
                
                st.markdown("---")

        # Скачивание результатов
        st.header("📥 Экспорт результатов")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            Retail_df.to_excel(writer, index=False)
        output.seek(0)
        
        st.download_button(
            label="Скачать данные с кластерами",
            data=output,
            file_name='clustered_customers.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        st.error(f"⚠️ Критическая ошибка: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ Пожалуйста, загрузите файл для начала анализа")