import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Генерация синтетических данных
def generate_synthetic_data(n_customers=1000):
    data = {
        'CustomerID': np.arange(1, n_customers + 1),
        'PurchaseDate': [datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(n_customers)],
        'TotalSpend': np.round(np.random.uniform(10, 1000, n_customers), 2),
        'NumTransactions': np.random.randint(1, 50, n_customers),
        'Frequency': np.random.randint(1, 30, n_customers),
        'UniqueCategories': np.random.randint(1, 10, n_customers),
        'LoyaltyProgram': np.random.choice([0, 1], n_customers)
    }
    return pd.DataFrame(data)

# Генерация датасета
synthetic_data = generate_synthetic_data()

# Сохранение в Excel
synthetic_data.to_excel('synthetic_retail_data.xlsx', index=False)