import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def generate_visualizations():
    # Create 'static' directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    df = pd.read_csv("Covid Dataset.csv")
    df = df.replace({'Yes':1,'No':0})

    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('static/correlation_heatmap.png')
    plt.close()

    # Distribution of COVID-19 cases
    plt.figure(figsize=(8, 6))
    df['COVID-19'].value_counts().plot(kind='bar')
    plt.title('Distribution of COVID-19 Cases')
    plt.xlabel('COVID-19')
    plt.ylabel('Count')
    plt.savefig('static/covid_distribution.png')
    plt.close()

    # Feature Importance
    features = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Abroad travel', 
                'Contact with COVID Patient', 'Attended Large Gathering', 
                'Visited Public Exposed Places', 'Family working in Public Exposed Places']
    X = df[features]
    y = df['COVID-19']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()

def perform_eda():
    df = pd.read_csv("Covid Dataset.csv")
    df = df.replace({'Yes':1,'No':0})
    basic_stats = df.describe()
    corr_matrix = df.corr()
    
    features = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Abroad travel', 
                'Contact with COVID Patient', 'Attended Large Gathering', 
                'Visited Public Exposed Places', 'Family working in Public Exposed Places']
    X = df[features]
    y = df['COVID-19']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    
    return basic_stats, corr_matrix, feature_importance