import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Dataset Load
df = pd.read_csv("dataset/data.csv")

# Important columns only
df = df[['total_sqft', 'bath', 'size', 'price']]

# Remove missing values
df = df.dropna()

# Convert BHK from "2 BHK" -> 2
df['bhk'] = df['size'].str.extract('(\d+)').astype(int)

# Convert total_sqft to number
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')

# Remove invalid rows
df = df.dropna()

# Inputs
X = df[['total_sqft', 'bath', 'bhk']]

# Output
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model Saved Successfully")