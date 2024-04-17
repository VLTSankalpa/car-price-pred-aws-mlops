import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Read the dataset from a CSV file
df = pd.read_csv("./train.csv")

print(df.columns)

# Drop the 'Unnamed: 0' column if it exists
df.drop('Unnamed: 0', inplace=True, axis=1)

# Convert Fuel_Type to numerical values: Petrol (1), Diesel (0)
df['Fuel_Type'] = df['Fuel_Type'].apply(lambda x: 1 if x == "Petrol" else 0)

# Separate target variable (SellingPrice)
y_train = df['SellingPrice']

# Drop 'SellingPrice' column from features
X_train = df.drop('SellingPrice', axis=1)

# Create a linear regression model
lin_reg = LinearRegression()

# Train the linear regression model with the features (X_train) and target variable (y_train)
lin_reg.fit(X_train.values, y_train)

# Create a new data point for prediction
X_new = np.array([26000, 0, 3, 1, 86, 0, 1300, 1015, 5]).reshape(1, -1)

# Predict the SellingPrice for the new data point using the trained model
prediction = lin_reg.predict(X_new)
print("Predicted SellingPrice:", prediction)