from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model
model_filename = 'linear_regression_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load company data
data_folder = 'data/'
companies = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.csv')]

# Function to calculate ups and downs
def calculate_percentage_up_down(company):
    df = pd.read_csv(f"{data_folder}{company}.csv")
    df['Daily Return'] = df['Close'].pct_change()
    percentage_up = (df[df['Daily Return'] > 0].count()['Daily Return'] / df['Daily Return'].count()) * 100
    percentage_down = (df[df['Daily Return'] < 0].count()['Daily Return'] / df['Daily Return'].count()) * 100
    return percentage_up, percentage_down

@app.route('/')
def index():
    return render_template('index.html', companies=companies)

@app.route('/predict', methods=['POST'])
def predict():
    selected_company = request.form['company']
    percentage_up, percentage_down = calculate_percentage_up_down(selected_company)
    return render_template('result.html', company=selected_company, percentage_up=percentage_up, percentage_down=percentage_down)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query')
    suggestions = [company for company in companies if query.lower() in company.lower()]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)
