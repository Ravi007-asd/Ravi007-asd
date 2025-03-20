import pandas as pd

# Load the cleaned data to check its structure
file_path = "C:\Projects\DL\stock\cleaned_data.csv"
df = pd.read_csv(file_path)

# Display basic information and first few rows
df.info(), df.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

# Convert Date to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort by Date to maintain chronological order
df = df.sort_values(by="Date")

# Create feature matrix using past 7 closing prices
window_size = 7
X, y = [], []

for i in range(len(df) - window_size):
    X.append(df["Close"].iloc[i:i+window_size].values)
    y.append(df["Close"].iloc[i+window_size])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
model_path = "stock_price_model_new.pkl"
scaler_path = "stock_price_scaler_new.pkl"
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

model_path, scaler_path
