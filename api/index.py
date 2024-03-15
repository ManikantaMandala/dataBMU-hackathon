from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.compose import ColumnTransformer as preprocessor
import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app) 

census_income = fetch_ucirepo(id=20)

X = census_income.data.features
y = census_income.data.targets
X_df = pd.DataFrame(X)

decision_tree_model = joblib.load('pickles/decision_tree_model.pkl')
k_neighbours_model = joblib.load('pickles/k_neighbours_model.pkl')
logistic_regression_model = joblib.load('pickles/logistic_regression_model.pkl')
random_forest_model = joblib.load('pickles/random_forest_model.pkl')
svc_model = joblib.load('pickles/SVC_model.pkl')
xgb_model = joblib.load('pickles/XGB_model.pkl')
numeric_features = ['age','hours-per-week']
categorical_features = ['education','marital-status','occupation','relationship']

numeric_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer,numeric_features),
        ('cat', categorical_transformer,categorical_features)
    ]
)

def calculate_income(req_data):
    model_name = req_data.get('model', 'random_forest_model')
    data = req_data.get('data', {})
    
    # Preprocess input data
    input_processed = preprocessor.transform(pd.DataFrame(data, index=[0]))
    
    # Load the specified model
    switcher = {
            'decision_tree': decision_tree_model,
            'k_neighbours_model': k_neighbours_model,
            'logistic_regression_model': logistic_regression_model,
            'random_forest_model': random_forest_model,
            'svc_model': svc_model,
            'xgb_model': xgb_model
            }
    model = switcher.get(model_name, random_forest_model)
    
    # Predict income level
    prediction = model.predict(input_processed)[0]
    
    return prediction

@app.post('/calculateIncome')
def handle_calculate_income_request():
    if request.is_json:
        data = request.json
        prediction = calculate_income(data)
        if prediction == 1:
            result = "Income is predicted to exceed $50K/yr."
        else:
            result = "Income is predicted to be $50K/yr or less."
        return jsonify({'prediction': result})
    else:
        return jsonify({"error": "Invalid request"}), 400

if __name__ == '__main__':
    preprocessor.fit(X_df)
    app.run(debug=True)
