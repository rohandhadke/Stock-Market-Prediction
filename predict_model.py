import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('linear_regression_model.pkl')

# Example input (replace with actual lagged return value)
lagged_return_value = 0.01  # Example value

# Create a DataFrame with the appropriate feature name
input_data = pd.DataFrame([[lagged_return_value]], columns=['Lagged Return'])

# Make prediction
prediction = model.predict(input_data)

print("Predicted Daily Return:", prediction[0])
