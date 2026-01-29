import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Gas Production Efficiency", layout="wide")

st.title("🛢️ Gas Production Efficiency Tracker")
st.markdown("Dashboard for monitoring US gas production efficiency and detecting operational anomalies (Flaring/Venting).")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('processed_gas_efficiency.csv')
        df['Production Date'] = pd.to_datetime(df['Production Date'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Error: 'processed_gas_efficiency.csv' not found. Please run the preprocessing script first.")
    st.stop()

# --- 3. SIDEBAR (FILTERS) ---
st.sidebar.header("Region Filter")
# Get top 10 regions by volume
valid_states = df.groupby('State')['Total_Production'].sum().sort_values(ascending=False).index[:10]
selected_state = st.sidebar.selectbox("Select Region:", valid_states)

# Filter data
df_filtered = df[df['State'] == selected_state].sort_values('Production Date')

# --- 4. KEY METRICS ---
# Get latest data points
last_data = df_filtered.iloc[-1]
prev_data = df_filtered.iloc[-2] if len(df_filtered) > 1 else last_data

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Gas Production (Last Month)",
        value=f"{last_data['Total_Production']:,.0f} Mcf",
        delta=f"{last_data['Total_Production'] - prev_data['Total_Production']:,.0f} Mcf"
    )

with col2:
    current_waste = last_data['Waste_Ratio'] * 100
    prev_waste = prev_data['Waste_Ratio'] * 100
    st.metric(
        label="Waste Ratio (Inefficiency)",
        value=f"{current_waste:.2f}%",
        delta=f"{current_waste - prev_waste:.2f}%",
        delta_color="inverse" # Red if waste increases, Green if it decreases
    )

with col3:
    st.metric(
        label="Wasted Volume (Flared/Vented)",
        value=f"{last_data['Waste']:,.0f} Mcf",
        delta_color="inverse"
    )

# --- 5. TIME SERIES & ANOMALY VISUALIZATION ---
st.subheader(f"Efficiency Trend: {selected_state}")

# Run Isolation Forest for the selected region
iso_model = IsolationForest(contamination=0.05, random_state=42)
X_anomaly = df_filtered[['Waste_Ratio']].fillna(0)
df_filtered['Anomaly'] = iso_model.fit_predict(X_anomaly)
anomalies = df_filtered[df_filtered['Anomaly'] == -1]

# Interactive Plot using Plotly
fig = go.Figure()

# Line Chart: Waste Ratio
fig.add_trace(go.Scatter(
    x=df_filtered['Production Date'], 
    y=df_filtered['Waste_Ratio'],
    mode='lines',
    name='Waste Ratio',
    line=dict(color='blue')
))

# Markers: Anomalies
fig.add_trace(go.Scatter(
    x=anomalies['Production Date'],
    y=anomalies['Waste_Ratio'],
    mode='markers',
    name='ANOMALY (High Waste)',
    marker=dict(color='red', size=10, symbol='x')
))

fig.update_layout(
    title="Gas Flaring/Venting Anomaly Detection", 
    xaxis_title="Year", 
    yaxis_title="Waste Ratio (0-1)"
)
st.plotly_chart(fig, use_container_width=True)

# --- 6. FORECASTING SECTION ---
st.markdown("---")
st.subheader("🤖 AI Forecasting (Next Month Prediction)")

if st.button("Run Prediction Model"):
    with st.spinner('Training Random Forest Model...'):
        # Prepare Lag Features
        df_ml = df_filtered.copy()
        df_ml['Waste_Ratio_Lag1'] = df_ml['Waste_Ratio'].shift(1)
        df_ml['Waste_Ratio_Lag3'] = df_ml['Waste_Ratio'].shift(3)
        df_ml['Total_Prod_Lag1'] = df_ml['Total_Production'].shift(1)
        df_ml = df_ml.dropna()

        if len(df_ml) > 10: # Ensure enough data exists
            features = ['Waste_Ratio_Lag1', 'Waste_Ratio_Lag3', 'Total_Prod_Lag1']
            X = df_ml[features]
            y = df_ml['Waste_Ratio']
            
            # Train Model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            
            # Predict Next Month
            last_row = df_filtered.iloc[[-1]].copy()
            last_row['Waste_Ratio_Lag1'] = df_filtered.iloc[-1]['Waste_Ratio']
            last_row['Waste_Ratio_Lag3'] = df_filtered.iloc[-3]['Waste_Ratio'] if len(df_filtered) >= 3 else 0
            last_row['Total_Prod_Lag1'] = df_filtered.iloc[-1]['Total_Production']
            
            next_pred = rf_model.predict(last_row[features])[0]
            
            # Display Results
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.info(f"Predicted Waste Ratio (Next Month): **{next_pred*100:.2f}%**")
            
            with col_pred2:
                diff = next_pred - last_data['Waste_Ratio']
                status = "INCREASE ⚠️" if diff > 0 else "DECREASE ✅"
                st.write(f"The trend is expected to **{status}** by {abs(diff)*100:.2f}% compared to this month.")
                
        else:
            st.warning("Insufficient data to perform forecasting for this region.")

# Footer
st.markdown("---")
st.caption("Developed for Oil & Gas IT Portfolio Project. Data Source: US DOI ONRR.")