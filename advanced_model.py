import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib

# Load the data
df = pd.read_csv("Covid Dataset.csv")
df = df.replace({'Yes':1,'No':0})

# Select features
features = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Abroad travel', 
            'Contact with COVID Patient', 'Attended Large Gathering', 
            'Visited Public Exposed Places', 'Family working in Public Exposed Places']

X = df[features]
y = df['COVID-19']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipelines for each model
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

gb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

nn_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('nn', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
])

# Train and evaluate models
models = {
    'Random Forest': rf_pipeline,
    'Gradient Boosting': gb_pipeline,
    'Neural Network': nn_pipeline
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} Cross-validation score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    model.fit(X_train, y_train)
    print(f"{name} Test score: {model.score(X_test, y_test):.3f}")

# Save models for A/B testing
joblib.dump(models['Random Forest'], 'covid_model_a.joblib')
joblib.dump(models['Gradient Boosting'], 'covid_model_b.joblib')

print("Models saved as 'covid_model_a.joblib' and 'covid_model_b.joblib'")