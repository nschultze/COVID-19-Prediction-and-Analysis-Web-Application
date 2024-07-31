import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load the data from the local file
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

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'covid_model.joblib')

print("Model trained and saved as 'covid_model.joblib'")
