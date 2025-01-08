# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Paths
data_path = "./data/processed/updated_processed_data.csv"
model_path = "./models/"
output_path = "./outputs/"
os.makedirs(output_path, exist_ok=True)

# Load data and models
@st.cache_data  # Updated caching mechanism
def load_data():
    data = pd.read_csv(data_path)
    with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return data, scaler

data, scaler = load_data()

# Available models
model_options = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Linear Regression": "linear_regression.pkl",
    "Lasso Regression": "lasso_regression.pkl",
}

# Load a selected model
def load_model(model_name):
    with open(os.path.join(model_path, model_name), "rb") as f:
        return pickle.load(f)

# App Title and Description
st.title("Real Estate Price Prediction Dashboard")
st.write("""
This dashboard estimates future prices affected by new house constructions.
You can select a city, choose a prediction model, and specify additional housing units to see the impact on prices.
""")

# Model Selection
selected_model_name = st.selectbox("Select a Prediction Model", list(model_options.keys()))
selected_model_file = model_options[selected_model_name]
model = load_model(selected_model_file)

# City Selection
cities = data["RegionName"].unique()
selected_city = st.selectbox("Select a City", sorted(cities))

# Enter additional housing units
extra_housing_units = st.number_input(
    "Enter the number of additional housing units (City_Housing_Starts):",
    min_value=0,
    step=1,
    value=0
)

# Filter data for the selected city
city_data = data[data["RegionName"] == selected_city]
if city_data.empty:
    st.error(f"No data available for the selected city: {selected_city}")
    st.stop()

# Ensure interaction features are present
city_data["Housing_Market_Interaction"] = city_data["City_Housing_Starts"] * city_data["market_heat_index"]
city_data["Housing_Sales_Ratio"] = city_data["City_Housing_Starts"] / (city_data["sales_count_nowcast"] + 1)

# Predict future prices
def predict_prices(city_data, extra_units, model, scaler):
    features = [
        "City_Housing_Starts",
        "new_construction_sales_all_homes",
        "market_heat_index",
        "percent_sold_above_list_all_homes",
        "percent_sold_below_list_all_homes",
        "sales_count_nowcast",
        "total_transaction_value_all_homes",
        "zhvi_all_homes_smoothed",
        "Housing_Market_Interaction",  # Ensure these features are included
        "Housing_Sales_Ratio",
    ]

    # Adjust `City_Housing_Starts` for prediction
    city_data["City_Housing_Starts"] += extra_units
    city_data["Housing_Market_Interaction"] = city_data["City_Housing_Starts"] * city_data["market_heat_index"]
    city_data["Housing_Sales_Ratio"] = city_data["City_Housing_Starts"] / (city_data["sales_count_nowcast"] + 1)

    X = city_data[features]
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)
    return predictions

# Generate predictions
base_predictions = predict_prices(city_data.copy(), 0, model, scaler)
adjusted_predictions = predict_prices(city_data.copy(), extra_housing_units, model, scaler)

# Display predictions
st.subheader(f"Predicted Prices for {selected_city} Using {selected_model_name}")
st.write(f"Average Predicted Price (No Additional Housing): ${np.mean(base_predictions):,.2f}")
st.write(f"Average Predicted Price (With {extra_housing_units} Additional Units): ${np.mean(adjusted_predictions):,.2f}")

# Interactive visualization with Plotly
st.subheader("Price Impact Visualization")
fig = go.Figure()

# Add traces for predictions
fig.add_trace(go.Scatter(
    x=city_data["Date"], y=base_predictions,
    mode='lines', name='No Additional Housing',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=city_data["Date"], y=adjusted_predictions,
    mode='lines', name=f'With {extra_housing_units} Units',
    line=dict(color='orange')
))

# Layout customization
fig.update_layout(
    title=f"Impact of {extra_housing_units} Additional Units on Prices in {selected_city}",
    xaxis_title="Date",
    yaxis_title="Predicted Price",
    legend_title="Scenario",
    hovermode="x unified"
)

# Render the interactive plot
st.plotly_chart(fig)
