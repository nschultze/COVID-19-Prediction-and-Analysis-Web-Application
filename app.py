from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from eda_visualization import perform_eda, generate_visualizations
import random
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load the trained models
try:
    model_a = joblib.load('covid_model_a.joblib')
    model_b = joblib.load('covid_model_b.joblib')
    app.logger.info("Models loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading models: {str(e)}")

# Generate visualizations when the app starts
generate_visualizations()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get user inputs
            inputs = [
                'breathing_problem', 'fever', 'dry_cough', 'sore_throat', 'abroad_travel',
                'contact_with_covid', 'attended_large_gathering', 'visited_public_exposed_places',
                'family_working_in_public_exposed_places'
            ]
            user_input = [int(request.form.get(feature, 0)) for feature in inputs]
            app.logger.debug(f"User input: {user_input}")

            # Create input array for prediction
            input_data = np.array([user_input])

            # A/B testing: randomly select a model
            selected_model = random.choice([model_a, model_b])
            app.logger.debug(f"Selected model: {selected_model.__class__.__name__}")

            # Make prediction
            prediction = selected_model.predict(input_data)
            probability = selected_model.predict_proba(input_data)[0][1]
            result = "You might have COVID-19" if prediction[0] > 0.5 else "You might not have COVID-19"
            
            app.logger.debug(f"Prediction: {prediction}, Probability: {probability}, Result: {result}")
            
            return render_template('result.html', result=result, probability=probability, model=selected_model.__class__.__name__)
        except Exception as e:
            app.logger.error(f"Error in prediction: {str(e)}")
            return "An error occurred during prediction. Please try again."

    return render_template('index.html')

@app.route('/understand', methods=['GET'])
def understand():
    try:
        basic_stats, corr_matrix, feature_importance = perform_eda()
        return render_template('understand.html', 
                               basic_stats=basic_stats.to_html(), 
                               corr_matrix=corr_matrix.to_html(), 
                               feature_importance=feature_importance.to_dict())
    except Exception as e:
        app.logger.error(f"Error in understand route: {str(e)}")
        return "An error occurred while generating the understanding page. Please try again."

if __name__ == '__main__':
    app.run(debug=True)