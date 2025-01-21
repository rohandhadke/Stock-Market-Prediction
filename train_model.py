import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import pickle

# Directory containing the stock data CSV files
data_dir = 'data'
combined_df = pd.DataFrame()

# Load each company's stock data
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        company_df = pd.read_csv(file_path)

        # Assuming the file contains columns: Date, Close
        company_df['Date'] = pd.to_datetime(company_df['Date'])
        company_df.set_index('Date', inplace=True)

        # Calculate Daily Returns
        company_df['Daily Return'] = company_df['Close'].pct_change(fill_method=None)
        company_df['Company'] = filename.split('.')[0]  # Use filename as company name
        combined_df = pd.concat([combined_df, company_df])

# Drop NaN values
combined_df.dropna(inplace=True)

# Prepare features and target variable
X = combined_df[['Close']].shift(1).dropna()
y = combined_df['Daily Return'].dropna().iloc[1:]  # Align with X

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Show percentage ups and downs
up_percentage = combined_df[combined_df['Daily Return'] > 0].groupby('Company').size() / combined_df.groupby('Company').size() * 100
down_percentage = combined_df[combined_df['Daily Return'] < 0].groupby('Company').size() / combined_df.groupby('Company').size() * 100

print("Percentage Ups:\n", up_percentage)
print("Percentage Downs:\n", down_percentage)

print("Model saved as 'linear_regression_model.pkl'")
