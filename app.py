import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Generate normally distributed data
np.random.seed(42)
n_samples = 200
data = {
    'CustomerID': range(1, n_samples + 1),
    'Age': np.random.normal(loc=40, scale=12, size=n_samples).astype(int),
    'AnnualIncome': np.random.normal(loc=75000, scale=20000, size=n_samples).astype(int),
    'SpendingScore': np.random.normal(loc=50, scale=20, size=n_samples).astype(int)
}
df = pd.DataFrame(data)
df['Age'] = df['Age'].clip(18, 70)
df['AnnualIncome'] = df['AnnualIncome'].clip(20000, 150000)
df['SpendingScore'] = df['SpendingScore'].clip(1, 100)

# Feature selection and scaling
features = ['Age', 'AnnualIncome', 'SpendingScore']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# KMeans clustering (3 segments)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(X_scaled)

# Streamlit UI
st.title("Customer Segmentation with Normal Distribution Data")

st.write("""
This app generates normally distributed customer data, clusters it, and provides interactive prediction and analysis.
""")

# Show statistical info
st.subheader("Statistical Information")
stats = df[features].describe().T
stats['median'] = df[features].median()
st.table(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'median']])

# User input for prediction
st.sidebar.header("Predict Your Segment")
age = st.sidebar.slider('Age', 18, 70, 40)
income = st.sidebar.slider('Annual Income', 20000, 150000, 75000, step=1000)
score = st.sidebar.slider('Spending Score', 1, 100, 50)

# Predict segment for user input
user_data = np.array([[age, income, score]])
user_data_scaled = scaler.transform(user_data)
user_segment = kmeans.predict(user_data_scaled)[0]

st.sidebar.success(f"Predicted Customer Segment: {user_segment}")

# Interactive Plotly visualization with 3 custom colors
st.subheader("Customer Segments Visualization (Interactive)")
custom_colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red

fig = px.scatter(
    df,
    x='AnnualIncome',
    y='SpendingScore',
    color='Segment',
    color_discrete_sequence=custom_colors,
    hover_data=['CustomerID', 'Age'],
    title='Customer Segments (interactive)'
)

# Add user input as a separate trace
fig.add_trace(go.Scatter(
    x=[income],
    y=[score],
    mode='markers',
    marker=dict(
        color='gold',
        size=18,
        symbol='star'
    ),
    name='Your Input'
))

fig.update_layout(
    legend=dict(
        x=0.8, y=0.95,
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='Black',
        borderwidth=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Segment profile
st.subheader("Segment Profiles")
profile = df.groupby('Segment')[features].mean().round(1)
st.table(profile)

if st.checkbox("Show raw data"):
    st.dataframe(df)
