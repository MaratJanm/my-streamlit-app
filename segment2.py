import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.display import display
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

# Загрузка данных
Retail_df = pd.read_excel("online_retail.xlsx")

# Конвертация колонны 'InvoiceDate' в формат даты и сохранение в 'Date' колонне
Retail_df['Date'] = pd.to_datetime(Retail_df['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')

# Функция для подсчета уникальных значений в каждой колонне
def unique_counts(dataframe):
    for i in dataframe.columns:
        count = dataframe[i].nunique()
        print(f"{i}: {count}")

# Вывод уникальных значений
unique_counts(Retail_df)

# Вывод общей стоимости
Retail_df['Total_Price'] = Retail_df['Quantity'] * Retail_df['UnitPrice']
print("Первые 10 строк после добавления 'Total_Price':")
print(Retail_df.head(10))

# Удаление строк с NaN в 'CustomerID'
Online_retail_df = Retail_df[np.isfinite(Retail_df['CustomerID'])]

# Удаление строк с отрицательными значениями Quantity
final_retail = Online_retail_df[Online_retail_df['Quantity'] > 0]

# Работа с датами
NOW = dt.datetime(2011, 12, 10)

# Создание RFM таблицы
rfmTable = final_retail.groupby('CustomerID').agg({
    'Date': lambda x: (NOW - x.max()).days,
    'InvoiceNo': lambda x: len(x),
    'Total_Price': lambda x: x.sum()
})
rfmTable.rename(columns={'Date': 'recency', 'InvoiceNo': 'frequency', 'Total_Price': 'monetary_value'}, inplace=True)

# Вывод размеров и первых 10 строк RFM таблицы
print("Размеры RFM таблицы:", rfmTable.shape)
print("Первые 10 строк RFM таблицы:")
display(rfmTable.head(10))

# Сортировка таблицы RFM
rfmTable.sort_values(['frequency', 'monetary_value'], ascending=[False, False], inplace=True)

# Кластеризация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfmTable)

clusters = KMeans(3)  # 3 кластера
rfmTable["cluster_label"] = clusters.fit_predict(X_scaled)

# Вывод средних значений по каждому кластеру
print("\nСредние значения по меткам кластеров:")
print(rfmTable.groupby('cluster_label').mean())

# Визуализация кластеров KMeans
plt.figure(figsize=(10, 6))
plt.scatter(rfmTable['recency'], rfmTable['monetary_value'],
            c=rfmTable['cluster_label'], cmap='viridis', marker='o')
plt.title('Кластеры клиентов по KMeans')
plt.xlabel('Recency (дни с последней покупки)')
plt.ylabel('Monetary Value (сумма покупок)')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()

# Пример случайной выборки
random.seed(9008)
X_sample = np.array(random.sample(X_scaled.tolist(), 20))

# Добавление меток кластеров в rfmTable
rfmTable["cluster_new"] = clusters.labels_

# Вывод средних значений по каждому кластеру
print("Средние значения по кластерам:")
print(rfmTable.groupby('cluster_new').mean())

# Дендограмма, построенная с использованием случайных выборок из X_scaled
rfmTable.drop('cluster_new', axis=1, inplace=True)

# Построение кластерной карты
cmap = sn.cubehelix_palette(as_cmap=True, rot=-0.3, light=1)
g = sn.clustermap(X_sample, cmap=cmap, linewidths=0.5)


# Метод Elbow для проверки сегментации кластера
cluster_range = range(1, 10)
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans(num_clusters)
    clusters.fit(X_scaled)
    cluster_errors.append(clusters.inertia_)

clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})

plt.figure(figsize=(12, 6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
plt.title('Метод Elbow для определения числа кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('Ошибка кластера (inertia)')
plt.grid(True)
plt.show()

# Сегментация на 3 кластера и добавление меток
clusters = KMeans(3)  # 3 кластера
clusters.fit(X_scaled)
rfmTable["cluster_label"] = clusters.labels_

# Вывод средних значений по каждому кластеру
print("Средние значения по меткам кластеров:")
print(rfmTable.groupby('cluster_label').mean())

#Клиенты с высокой новизной, низкой частотой и низкой денежной стоимостью сегментируются в этот кластер. Это наименее прибыльные клиенты для компании.
rfmTable_0 = rfmTable[rfmTable['cluster_label'] == 0]
print("\nКлиенты с высокой новизной, низкой частотой и низкой денежной стоимостью сегментируются в этот кластер. Это наименее прибыльные клиенты для компании:")
print(rfmTable_0.head(5))

#Клиенты, которые являются потенциальными клиентами с хорошей частотой и денежной ценностью. Компания должна работать с ними, чтобы превратить их в наиболее прибыльных клиентов.
rfmTable_1 = rfmTable[rfmTable['cluster_label'] == 1]
print("\nКлиенты, которые являются потенциальными клиентами с хорошей частотой и денежной ценностью. Компания должна работать с ними, чтобы превратить их в наиболее прибыльных клиентов:")
print(rfmTable_1.head(5))

#Клиенты с низкой новизной, высокой частотой и денежной ценностью сегментируются в этом кластере. Это самые прибыльные и высокоценные клиенты, на которых должна обратить внимание компания.
rfmTable_2 = rfmTable[rfmTable['cluster_label'] == 2]
print("\nКлиенты с низкой новизной, высокой частотой и денежной ценностью сегментируются в этом кластере. Это самые прибыльные и высокоценные клиенты, на которых должна обратить внимание компания:")
print(rfmTable_2.head(5))

#Вывод всех клиентов с определенным кластером
clusters = KMeans(3)
clusters.fit( X_scaled )
rfmTable.head(10)
rfmTable.to_excel('clients_clusters.xlsx', index=False)

