import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn import mixture
from sklearn import manifold
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import cluster
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import ensemble
import warnings

from IPython.display import display, HTML

warnings.filterwarnings("ignore")

plt.rcParams["patch.force_edgecolor"] = True
# Загрузка данных с указанными параметрами
data = pd.read_csv(
    "customer_segmentation_project.csv",
    encoding="ISO-8859-1",
    dtype={'CustomerID': str, 'InvoiceNo': str}
)
# Проверка формы данных
print('Data shape: {}'.format(data.shape))

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

data.dropna(subset=['Description', 'CustomerID'], inplace=True)
data.reset_index(drop=True, inplace=True)

data.drop_duplicates(inplace=True)

negative_quantity = data[(data['Quantity'] < 0)]
negative_quantity.head()

negative_quantity['refund'] = negative_quantity['InvoiceNo'].apply(lambda x: 1 if str(x)[0] != 'C' else 0)

temp = data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Number of products'})

nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x: 1 if str(x)[0] == 'C' else 0)
proc_ref = nb_products_per_basket['order_canceled'].sum() / nb_products_per_basket.shape[0] * 100

nb_products_per_basket[nb_products_per_basket['order_canceled']==1].head()

def get_quantity_canceled(data):
    # Инициализируем Series той же длины, что и столбцы таблицы, нулями
    quantity_canceled = pd.Series(np.zeros(data.shape[0]), index=data.index)
    negative_quantity = data[(data['Quantity'] < 0)].copy()
    for index, col in negative_quantity.iterrows():
        # Создаем DataFrame из всех контрагентов
        df_test = data[(data['CustomerID'] == col['CustomerID']) &
                       (data['StockCode']  == col['StockCode']) &
                       (data['InvoiceDate'] < col['InvoiceDate']) &
                       (data['Quantity'] > 0)].copy()
        # Транзация-возврат не имеет контрагента - ничего не делаем
        if (df_test.shape[0] == 0):
            continue
        # Транзакция-возврат имеет ровно одного контрагента
        # Добавляем количество отмененного в столбец QuantityCanceled
        elif (df_test.shape[0] == 1):
            index_order = df_test.index[0]
            quantity_canceled.loc[index_order] = -col['Quantity']
        # Транзакция-возврат имеет несколько контрагентов
        # Задаем количество отмененного товара в столбец QuantityCanceled для той транзакции на покупку,
        # в которой количество товара > -(количество товаров в транзакции-возврате)
        elif (df_test.shape[0] > 1):
            df_test.sort_index(axis=0 ,ascending=False, inplace = True)
            for ind, val in df_test.iterrows():
                if val['Quantity'] < -col['Quantity']:
                    continue
                quantity_canceled.loc[ind] = -col['Quantity']
                break
    return quantity_canceled

data['QuantityCanceled'] = get_quantity_canceled(data)

data = data[data['Quantity'] > 0]

n_unique_code = data['StockCode'][data['StockCode'].str.contains('^[a-zA-Z]+', regex=True)].nunique()
data = data[~data['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]

data = data[data['UnitPrice'] != 0]

country_customer = data.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)
country_customer = pd.DataFrame(country_customer).reset_index().rename(mapper={'CustomerID': 'Value'}, axis=1)

country_order = data.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)
country_order = pd.DataFrame(country_order).reset_index().rename(mapper={'InvoiceNo': 'Value'}, axis=1)

data['TotalPrice'] = data['UnitPrice']*(data['Quantity'] - data['QuantityCanceled'])

country_revenue = data.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
country_revenue = pd.DataFrame(country_revenue).reset_index().rename(mapper={'TotalPrice': 'Revenue'}, axis=1)

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

data['month'] = data['InvoiceDate'].dt.month
data['day_of_week'] = data['InvoiceDate'].dt.day_name ()
data['hour'] = data['InvoiceDate'].dt.hour

month_revenue = data.groupby('month')['TotalPrice'].sum().sort_values(ascending=False)
month_revenue = pd.DataFrame(month_revenue).reset_index().rename(mapper={'TotalPrice': 'Revenue'}, axis=1)

day_of_week_order = data.groupby('day_of_week')['InvoiceNo'].nunique().sort_values(ascending=False)
day_of_week_order = pd.DataFrame(day_of_week_order).reset_index().rename(mapper={'InvoiceNo': 'Value'}, axis=1)


weekdays = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
weekdays_data = set(day_of_week_order['day_of_week'].tolist())


df_dt = data[['InvoiceDate', 'InvoiceNo', 'hour']]
df_dt['date'] = df_dt['InvoiceDate'].dt.date

data_dt = df_dt.groupby(['date', 'hour'])['InvoiceNo'].nunique()
data_dt = pd.DataFrame(data_dt).reset_index().rename(mapper={'InvoiceNo': 'value'}, axis=1)

hour_dt = data_dt.groupby('hour')['value'].mean()
hour_dt = pd.DataFrame(hour_dt).reset_index()

#важно
t0 = pd.to_datetime('2011-12-10 00:00:00')
recency = (t0 - data.groupby(by=['CustomerID'])['InvoiceDate'].max()).dt.days
frequency = data.groupby(by=['CustomerID'])['InvoiceNo'].nunique()
monetary_value = data.groupby(by=['CustomerID'])['TotalPrice'].sum()

rfm_table = pd.DataFrame({
                          'Recency': recency,
                          'Frequency': frequency,
                          'Monetary': monetary_value
                          })

frequency_limit = np.quantile(rfm_table['Frequency'], 0.95)
monetary_limit = np.quantile(rfm_table['Monetary'], 0.95)

rfm_table_cleaned = rfm_table[(rfm_table['Frequency'] <= frequency_limit) & (rfm_table['Monetary'] <= monetary_limit)]

# создадим трёхмерный объект
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)
# добавим дополнительную ось в объект картинки
fig.add_axes(ax)
ax.azim = 20
ax.elev = 30

# визуализируем данные, передав значения x, y, z, а также информацию о группировке данных по цветам
ax.scatter(
    rfm_table_cleaned['Recency'].to_list(),
    rfm_table_cleaned['Frequency'].to_list(),
    rfm_table_cleaned['Monetary'].to_list()
)
# добавим оси
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary');

pipe = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=2, random_state=42))
    ])

rfm_table_processed  = pd.DataFrame(pipe.fit_transform(rfm_table_cleaned))
rfm_table_processed.columns = ['axis-1','axis-2']

fig = plt.figure(figsize=(12, 5))
sns.scatterplot(data=rfm_table_processed, x='axis-1', y='axis-2');

silhouette = []

silhouet = [0, 0]

for n_clusters in range(2, 11):
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(rfm_table_processed)
    value = metrics.silhouette_score(rfm_table_processed, kmeans.labels_, random_state=42)
    silhouette.append(value)
    if silhouet[1] < value:
        silhouet[0] = n_clusters
        silhouet[1] = value

plot = sns.lineplot(x=range(2, 11), y=silhouette);
plot.set_title('Зависимость коэффициента силуета от числа кластеров');
plot.set(xlabel='Число кластеров', ylabel='Коэффициент силуета');

print(f'Наибольший коэффициент силуета: {round(silhouet[1], 2)} при количество кластеров: {silhouet[0]}')


kmeans = cluster.KMeans(n_clusters=3, n_init=10, random_state=42).fit(rfm_table_processed)

cluster_values = [np.count_nonzero (kmeans.labels_ == i) for i in range(3)]

print(f'В самом большом кластере {max(cluster_values)} элементов')


fig = plt.figure(figsize=(12, 5))
sns.scatterplot(
    data=rfm_table_processed,
    x='axis-1',
    y='axis-2',
    hue=kmeans.labels_,
    palette=['green','orange','red']
);

rfm_table_copy = rfm_table_cleaned.copy()
rfm_table_copy['label'] = kmeans.labels_
rfm_table_mean = rfm_table_copy.groupby('label').mean()
display(rfm_table_mean)

pipe = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('tsne', manifold.TSNE(n_components=2, perplexity=50, random_state=100, n_jobs=-1))
    ])

rfm_table_processed  = pd.DataFrame(pipe.fit_transform(rfm_table_cleaned))
rfm_table_processed.columns = ['axis-1','axis-2']

silhouette = []

silhouet = [0, 0]

for n_clusters in range(3, 9):
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(rfm_table_processed)
    value = metrics.silhouette_score(rfm_table_processed, kmeans.labels_, random_state=42)
    silhouette.append(value)
    if silhouet[1] < value:
        silhouet[0] = n_clusters
        silhouet[1] = value

plot = sns.lineplot(x=range(3, 9), y=silhouette);
plot.set_title('Зависимость коэффициента силуета от числа кластеров');
plot.set(xlabel='Число кластеров', ylabel='Коэффициент силуета');


kmeans = cluster.KMeans(n_clusters=7, n_init=10, random_state=42).fit(rfm_table_processed)

cluster_values = [np.count_nonzero (kmeans.labels_ == i) for i in range(7)]

fig = plt.figure(figsize=(12, 5))
sns.scatterplot(data=rfm_table_processed, x='axis-1', y='axis-2', hue=kmeans.labels_.astype('str'));

rfm_table_copy = rfm_table_cleaned.copy()
rfm_table_copy['label'] = kmeans.labels_
rfm_table_mean = rfm_table_copy.groupby('label').mean().round()
display(rfm_table_mean)
print(f'Максимальное (среди всех кластеров) среднее значение признака Recency: {round(rfm_table_mean["Recency"].max())}')
