import pandas as pd
import numpy as np

# Create a mock dataset
np.random.seed(42)
n_samples = 1000

data = {
    'Breathing Problem': np.random.choice([0, 1], n_samples),
    'Fever': np.random.choice([0, 1], n_samples),
    'Dry Cough': np.random.choice([0, 1], n_samples),
    'Sore throat': np.random.choice([0, 1], n_samples),
    'Abroad travel': np.random.choice([0, 1], n_samples),
    'Contact with COVID Patient': np.random.choice([0, 1], n_samples),
    'Attended Large Gathering': np.random.choice([0, 1], n_samples),
    'Visited Public Exposed Places': np.random.choice([0, 1], n_samples),
    'Family working in Public Exposed Places': np.random.choice([0, 1], n_samples),
    'COVID-19': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)

# Save the mock dataset
df.to_csv('Covid Dataset.csv', index=False)
print("Mock dataset created and saved as 'Covid Dataset.csv'")