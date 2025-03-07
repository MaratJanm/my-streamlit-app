import pandas as pd

# Загрузка данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)

# Расчет TotalSpend
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# Группировка по CustomerID
customer_data = df.groupby('CustomerID').agg(
    TotalSpend=('TotalSpend', 'sum'),
    NumTransactions=('InvoiceNo', 'nunique'),
    Frequency=('InvoiceDate', lambda x: (x.max() - x.min()).days / 30),  # Средняя частота в месяц
    UniqueCategories=('StockCode', 'nunique')
).reset_index()

# Сохранение в Excel
customer_data.to_excel('retail_customers.xlsx', index=False)