# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Paths
data_path = "./data/processed/updated_processed_data.csv"
output_path = "./eda_outputs/"
os.makedirs(output_path, exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv(data_path)
data["Date"] = pd.to_datetime(data["Date"])

# General Overview
print("Generating dataset overview...")
overview = data.describe(include="all")
overview.to_csv(os.path.join(output_path, "dataset_overview.csv"))

# Missing Values
print("Analyzing missing values...")
missing_values = data.isnull().sum().sort_values(ascending=False)
missing_values.to_csv(os.path.join(output_path, "missing_values.csv"))

# Time Series Overview
print("Generating time-series plot...")
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Date", y="median_sale_price_all_homes", errorbar=None)  # Updated
plt.title("Median Sale Price Over Time")
plt.xlabel("Date")
plt.ylabel("Median Sale Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "median_sale_price_over_time.png"))
plt.close()

# Correlation Heatmap
print("Generating correlation heatmap...")
numeric_data = data.select_dtypes(include=["number"])  # Only numeric columns
correlation = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "correlation_heatmap.png"))
plt.close()

# Feature Distributions
print("Generating feature distributions...")
numeric_features = numeric_data.columns
for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{feature}_distribution.png"))
    plt.close()

# Scatter Plot for Key Features
print("Generating scatter plot for key features...")
key_features = [
    "City_Housing_Starts",
    "market_heat_index",
    "percent_sold_above_list_all_homes",
    "percent_sold_below_list_all_homes",
    "sales_count_nowcast",
    "zhvi_all_homes_smoothed",
]

for feature in key_features:
    if feature in data.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=feature, y="median_sale_price_all_homes")
        plt.title(f"Median Sale Price vs {feature}")
        plt.xlabel(feature)
        plt.ylabel("Median Sale Price")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"median_sale_price_vs_{feature}.png"))
        plt.close()

# Interactive Visualizations with Plotly
print("Generating interactive visualizations...")
interactive_plot = px.scatter(
    data_frame=data,
    x="City_Housing_Starts",
    y="median_sale_price_all_homes",
    color="market_heat_index" if "market_heat_index" in data.columns else None,
    size="sales_count_nowcast" if "sales_count_nowcast" in data.columns else None,
    hover_data=["RegionName", "StateName"] if "RegionName" in data.columns and "StateName" in data.columns else None,
    title="Median Sale Price vs City Housing Starts (Interactive)",
)
interactive_plot.write_html(os.path.join(output_path, "interactive_plot.html"))

# Summary
print("EDA completed. Outputs saved in the 'eda_outputs/' directory.")
