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
st.write('Используйте заголовки: CustomerID, PurchaseDate, TotalSpend, NumTransactions, Frequency, UniqueCategories, LoyaltyProgram.')

# Загрузка файла
uploaded_file = st.file_uploader("Выберите файл", type=["xlsx"])

# Кэширование данных
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

if uploaded_file is not None:
    # Чтение данных
    Retail_df = load_data(uploaded_file)

    # Проверка наличия необходимых столбцов
    required_columns = ['CustomerID', 'PurchaseDate', 'TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories', 'LoyaltyProgram']
    if not all(col in Retail_df.columns for col in required_columns):
        st.error("Ошибка: Недостаточно заголовков в загруженном файле. Пожалуйста, убедитесь, что файл содержит все необходимые заголовки.")
    else:
        # Преобразование даты
        Retail_df['PurchaseDate'] = pd.to_datetime(Retail_df['PurchaseDate'], format='%Y-%m-%d %H:%M:%S')

        # Удаление строк с пропущенными значениями
        Retail_df = Retail_df.dropna()

        # Вывод первых строк данных
        st.write("Первые 10 строк данных:")
        st.write(Retail_df.head(10))

        # Выбор признаков для кластеризации
        features = ['TotalSpend', 'NumTransactions', 'Frequency', 'UniqueCategories']
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
            silhouette_scores.append(silhouette_score(X_scaled, clusters.labels_))

        clusters_df = pd.DataFrame({
            "num_clusters": cluster_range,
            "cluster_errors": cluster_errors,
            "silhouette_scores": silhouette_scores
        })

        # Визуализация метода локтя
        st.write("Метод локтя для определения числа кластеров:")
        plt.figure(figsize=(12, 6))
        plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
        plt.title('Метод локтя')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Ошибка кластера (inertia)')
        plt.grid(True)
        st.pyplot(plt)

        # Визуализация silhouette score
        st.write("Silhouette Score для определения числа кластеров:")
        plt.figure(figsize=(12, 6))
        plt.plot(clusters_df.num_clusters, clusters_df.silhouette_scores, marker="o")
        plt.title('Silhouette Score')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        st.pyplot(plt)

        # Кластеризация KMeans
        n_clusters = st.slider("Выберите количество кластеров", min_value=2, max_value=10, value=3)
        clusters = KMeans(n_clusters=n_clusters, random_state=42)
        Retail_df["cluster_label"] = clusters.fit_predict(X_scaled)

        # Визуализация кластеров (2D)
        st.write("График кластеризации клиентов по KMeans (TotalSpend vs Frequency):")
        plt.figure(figsize=(10, 6))
        plt.scatter(Retail_df['TotalSpend'], Retail_df['Frequency'],
                    c=Retail_df['cluster_label'], cmap='viridis', marker='o')
        plt.title('Кластеры клиентов по KMeans')
        plt.xlabel('Total Spend (общая сумма покупок)')
        plt.ylabel('Frequency (частота покупок)')
        plt.colorbar(label='Cluster Label')
        plt.grid(True)
        st.pyplot(plt)

        # Визуализация всех признаков (Pairplot)
        st.write("Pairplot для всех признаков:")
        sns.pairplot(Retail_df, hue='cluster_label', vars=features, palette='viridis')
        st.pyplot(plt)

        # Визуализация PCA
        st.write("Кластеры клиентов (PCA):")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Retail_df['cluster_label'], cmap='viridis', marker='o')
        plt.title('Кластеры клиентов (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster Label')
        plt.grid(True)
        st.pyplot(plt)

        # Описание кластеров
        st.write("Медианные значения по кластерам:")
        cluster_medians = Retail_df.groupby('cluster_label').median()[features]
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