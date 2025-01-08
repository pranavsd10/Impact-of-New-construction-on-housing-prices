import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Paths
data_path = "./data/processed/updated_processed_data.csv"
model_path = "./models/"
results_path = "./models/model_comparison.csv"
os.makedirs(model_path, exist_ok=True)

# Load the processed dataset
print("Loading processed dataset...")
data = pd.read_csv(data_path)

# Define Features and Target
print("Defining features and target...")
data["Housing_Market_Interaction"] = data["City_Housing_Starts"] * data["market_heat_index"]
data["Housing_Sales_Ratio"] = data["City_Housing_Starts"] / (data["sales_count_nowcast"] + 1)

features = [
    "City_Housing_Starts",
    "new_construction_sales_all_homes",
    "market_heat_index",
    "percent_sold_above_list_all_homes",
    "percent_sold_below_list_all_homes",
    "sales_count_nowcast",
    "total_transaction_value_all_homes",
    "zhvi_all_homes_smoothed",
    "Housing_Market_Interaction",
    "Housing_Sales_Ratio",
]
target = "median_sale_price_all_homes"

# Ensure all required columns are present
for col in features + [target]:
    if col not in data.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

X = data[features].fillna(0)  # Fill missing feature values with 0
y = data[target].fillna(0)   # Fill missing target values with 0

# Train-Test Split
print("Performing train-test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
scaler_path = os.path.join(model_path, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# Define Models
print("Defining models...")
base_models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
}

# Train and Evaluate Models
print("Training and evaluating models...")
results = []
for name, model in base_models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": mae, "R²": r2})
    
    # Save the trained model
    model_file = os.path.join(model_path, f"{name.replace(' ', '_').lower()}.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"{name} saved to {model_file}")

# Hyperparameter Tuning for Best Model (Example with XGBoost)
print("Performing hyperparameter tuning for XGBoost...")
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters for XGBoost: {grid_search.best_params_}")

# Save the tuned model
best_model_path = os.path.join(model_path, "best_xgboost.pkl")
with open(best_model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Tuned model saved to {best_model_path}")

# Diagnostics: Impact of City Housing Starts
print("Generating diagnostics plot...")
city_data = data.iloc[0:1].copy()  # Select a sample row
city_housing_values = range(0, 10000, 100)
predictions = []

for value in city_housing_values:
    city_data["City_Housing_Starts"] = value
    city_data["Housing_Market_Interaction"] = city_data["City_Housing_Starts"] * city_data["market_heat_index"]
    city_data["Housing_Sales_Ratio"] = city_data["City_Housing_Starts"] / (city_data["sales_count_nowcast"] + 1)
    X_sim = scaler.transform(city_data[features])
    predictions.append(best_model.predict(X_sim)[0])

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(city_housing_values, predictions, label="Predicted Prices")
plt.xlabel("City Housing Starts")
plt.ylabel("Predicted Price")
plt.title("Impact of City Housing Starts on Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()
