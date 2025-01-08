import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Paths
data_path = "./data/processed/updated_processed_data.csv"
model_path = "./models/"
output_path = "./outputs/"

# Load Data
data = pd.read_csv(data_path)

# Define Features and Target
features = [
    "City_Housing_Starts",
    "new_construction_sales_all_homes",
    "market_heat_index",
    "percent_sold_above_list_all_homes",
    "percent_sold_below_list_all_homes",
    "sales_count_nowcast",
    "total_transaction_value_all_homes",
    "zhvi_all_homes_smoothed"
]
target = "median_sale_price_all_homes"

# Preprocess Data
data = data.dropna(subset=features + [target])  # Drop rows with missing values
X = data[features]
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load Model
with open(model_path + "random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# Diagnostic 1: Plot Predictions for Increasing City_Housing_Starts
print("Generating predictions for increasing City_Housing_Starts...")
selected_city = "Example City"  # Replace with an actual city name
city_data = data[data["RegionName"] == selected_city]
if city_data.empty:
    print(f"No data available for {selected_city}")
    city_data = data.iloc[0:1]  # Use the first row if no city is found
else:
    city_data = city_data.iloc[0:1].copy()

city_housing_values = range(0, 10000, 100)  # Simulate up to 10,000 units
predictions = []
for value in city_housing_values:
    city_data["City_Housing_Starts"] = value
    X_sim = scaler.transform(city_data[features])
    predictions.append(model.predict(X_sim)[0])

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(city_housing_values, predictions, label="Predicted Prices")
plt.xlabel("City Housing Starts")
plt.ylabel("Predicted Price")
plt.title(f"Impact of City Housing Starts on Predicted Prices for {selected_city}")
plt.legend()
plt.grid(True)
plt.show()

# Diagnostic 2: Feature Importance (SHAP)
print("Generating SHAP feature importance...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_scaled)

shap.summary_plot(shap_values, X_train, feature_names=features)

# Diagnostic 3: Add Interaction Features and Retrain Model
print("Adding interaction features and retraining model...")
data["Housing_Market_Interaction"] = data["City_Housing_Starts"] * data["market_heat_index"]
data["Housing_Sales_Ratio"] = data["City_Housing_Starts"] / (data["sales_count_nowcast"] + 1)

updated_features = features + ["Housing_Market_Interaction", "Housing_Sales_Ratio"]
X_updated = data[updated_features]
y_updated = data[target]

X_train_updated, X_test_updated, y_train_updated, y_test_updated = train_test_split(
    X_updated, y_updated, test_size=0.2, random_state=42
)
X_train_updated_scaled = scaler.fit_transform(X_train_updated)
X_test_updated_scaled = scaler.transform(X_test_updated)

updated_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
updated_model.fit(X_train_updated_scaled, y_train_updated)

# Save Updated Model
with open(model_path + "updated_random_forest.pkl", "wb") as f:
    pickle.dump(updated_model, f)

print("Updated model retrained and saved.")

# Diagnostic 4: Hyperparameter Tuning
print("Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train_updated_scaled, y_train_updated)
best_model = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")

# Save Best Model
with open(model_path + "best_random_forest.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model saved.")

# Diagnostic 5: Plot Predictions with Updated Model
print("Generating predictions with updated model...")
predictions_updated = []
for value in city_housing_values:
    city_data["City_Housing_Starts"] = value
    X_sim_updated = scaler.transform(city_data[updated_features])
    predictions_updated.append(updated_model.predict(X_sim_updated)[0])

# Plot Updated Results
plt.figure(figsize=(10, 6))
plt.plot(city_housing_values, predictions_updated, label="Predicted Prices (Updated Model)", color="orange")
plt.xlabel("City Housing Starts")
plt.ylabel("Predicted Price")
plt.title(f"Impact of City Housing Starts on Predicted Prices (Updated Model) for {selected_city}")
plt.legend()
plt.grid(True)
plt.show()
