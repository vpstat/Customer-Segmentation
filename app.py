import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Generate sample data
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

# Streamlit UI
st.title("Customer Segmentation Predictor")

st.write("""
Input values for Age, Annual Income, and Spending Score to predict the customer segment.
""")

# User input
age = st.slider('Age', 18, 70, 30)
income = st.slider('Annual Income', 20000, 150000, 50000, step=1000)
score = st.slider('Spending Score', 1, 100, 50)

# Prepare user input for prediction
user_data = np.array([[age, income, score]])
user_data_scaled = scaler.transform(user_data)
user_segment = kmeans.predict(user_data_scaled)[0]

st.success(f"Predicted Customer Segment: {user_segment}")

# Plotly interactive scatter plot with separate traces for user input
st.subheader("Customer Segments Visualization (Interactive)")

# Plot existing customers
fig = px.scatter(
    df,
    x='AnnualIncome',
    y='SpendingScore',
    color='Segment',
    hover_data=['CustomerID', 'Age'],
    title='Customer Segments (interactive)'
)

# Add user input as a separate trace
fig.add_trace(go.Scatter(
    x=[income],
    y=[score],
    mode='markers',
    marker=dict(
        color='red',
        size=16,
        symbol='star'
    ),
    name='Your Input'
))

# Update layout to move legend and avoid overlap
fig.update_layout(
    legend=dict(
        x=1.3, y=0.95,
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='Black',
        borderwidth=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Segment profile
st.subheader("Segment Profiles")
profile = df.groupby('Segment')[['Age', 'AnnualIncome', 'SpendingScore']].mean().round(1)
st.table(profile)

if st.checkbox("Show raw data"):
    st.dataframe(df)
