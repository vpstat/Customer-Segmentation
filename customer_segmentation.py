import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)
data = {
    'CustomerID': range(1, 201),
    'Age': np.random.randint(18, 70, 200),
    'AnnualIncome': np.random.randint(20000, 150000, 200),
    'SpendingScore': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)

# Feature selection and scaling
features = ['Age', 'AnnualIncome', 'SpendingScore']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(X_scaled)

# Save the segmented data
df.to_csv('segmented_customers.csv', index=False)
