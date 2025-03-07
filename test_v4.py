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

# Заголовок приложения
st.title('Кластеризация клиентов ретейла')
st.write('Пожалуйста, загрузите файл в формате .xlsx для анализа клиентов.')
st.write('Используйте заголовки: CustomerID, TotalSpend, NumTransactions, Frequency, UniqueCategories.')

# Загрузка файла
uploaded_file = st.file_uploader("Выберите файл", type=["xlsx"])

# Кэширование данных
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

if uploaded_file is not None:
    try:
        # Чтение данных
        Retail_df = load_data(uploaded_file)

        # Проверка наличия необходимых столбцов
        required_columns = ['CustomerID', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        if not all(col in Retail_df.columns for col in required_columns):
            st.error("Ошибка: Недостаточно заголовков в загруженном файле. Пожалуйста, убедитесь, что файл содержит все необходимые заголовки.")
            st.stop()

        # Удаление строк с пропущенными значениями
        Retail_df = Retail_df.dropna()
        if Retail_df.empty:
            st.error("Ошибка: После удаления пропусков данные отсутствуют.")
            st.stop()

        # Проверка типов данных
        numeric_cols = ['TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
        for col in numeric_cols:
            Retail_df[col] = pd.to_numeric(Retail_df[col], errors='coerce')
        
        if Retail_df[numeric_cols].isnull().any().any():
            st.error("Ошибка: Некорректные значения в числовых колонках.")
            st.stop()

        # Вывод первых строк данных
        st.write("Первые 10 строк данных:")
        st.write(Retail_df.head(10))

        # Выбор признаков для кластеризации
        features = numeric_cols
        X = Retail_df[features]

        # Масштабирование данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Метод локтя для определения оптимального числа кластеров
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

        clusters_df = pd.DataFrame({
            "num_clusters": cluster_range,
            "cluster_errors": cluster_errors,
            "silhouette_scores": silhouette_scores
        })

        # Визуализация метода локтя
        st.write("Метод локтя для определения числа кластеров:")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
        ax1.set_title('Метод локтя')
        ax1.set_xlabel('Количество кластеров')
        ax1.set_ylabel('Ошибка кластера (inertia)')
        ax1.grid(True)
        st.pyplot(fig1)
        plt.close(fig1)

        # Визуализация silhouette score
        st.write("Silhouette Score для определения числа кластеров:")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(clusters_df.num_clusters, clusters_df.silhouette_scores, marker="o")
        ax2.set_title('Silhouette Score')
        ax2.set_xlabel('Количество кластеров')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True)
        st.pyplot(fig2)
        plt.close(fig2)

        # Кластеризация KMeans
        n_clusters = st.slider("Выберите количество кластеров", min_value=2, max_value=10, value=3)
        clusters = KMeans(n_clusters=n_clusters, random_state=42)
        Retail_df["cluster_label"] = clusters.fit_predict(X_scaled)

        # Визуализация кластеров (2D)
        st.write("График кластеризации клиентов по KMeans (TotalSpend vs Frequency):")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        scatter = ax3.scatter(Retail_df['TotalSpend'], Retail_df['Frequency'],
                    c=Retail_df['cluster_label'], cmap='viridis', marker='o')
        ax3.set_title('Кластеры клиентов по KMeans')
        ax3.set_xlabel('Total Spend (общая сумма покупок)')
        ax3.set_ylabel('Frequency (частота покупок)')
        plt.colorbar(scatter, ax=ax3, label='Cluster Label')
        ax3.grid(True)
        st.pyplot(fig3)
        plt.close(fig3)

        # Визуализация всех признаков (Pairplot)
        st.write("Pairplot для всех признаков:")
        pairplot_fig = sns.pairplot(Retail_df, hue='cluster_label', vars=features, palette='viridis')
        st.pyplot(pairplot_fig.fig)
        plt.close(pairplot_fig.fig)

        # Визуализация PCA
        st.write("Кластеры клиентов (PCA):")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        scatter_pca = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=Retail_df['cluster_label'], cmap='viridis', marker='o')
        ax4.set_title('Кластеры клиентов (PCA)')
        ax4.set_xlabel('Principal Component 1')
        ax4.set_ylabel('Principal Component 2')
        plt.colorbar(scatter_pca, ax=ax4, label='Cluster Label')
        ax4.grid(True)
        st.pyplot(fig4)
        plt.close(fig4)

        # Описание кластеров
        st.write("Медианные значения по кластерам:")
        cluster_medians = Retail_df.groupby('cluster_label').median(numeric_only=True)[features]
        st.write(cluster_medians)

        # Отображение клиентов по кластерам
        st.write("### Клиенты по кластерам")
        for i in range(n_clusters):
            cluster_data = Retail_df[Retail_df['cluster_label'] == i]
            st.write(f"#### Кластер {i}:")
            st.write(f"Количество клиентов: {len(cluster_data)}")
            st.write(cluster_data[['CustomerID', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']])

        # Фильтрация по кластерам
        st.write("### Фильтрация по кластерам")
        selected_cluster = st.selectbox("Выберите кластер для просмотра", range(n_clusters))
        st.write(f"#### Клиенты в кластере {selected_cluster}:")
        st.write(Retail_df[Retail_df['cluster_label'] == selected_cluster][['CustomerID', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']])

        # Скачивание результатов
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            Retail_df.to_excel(writer, sheet_name='Кластеризация', index=False)
        output.seek(0)

        st.success("Кластеризация завершена успешно!")

        st.download_button(
            label="Скачать файл",
            data=output,
            file_name='clients_clusters.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")