# COVID-19 Prediction and Analysis Web Application

## Project Overview
This project is a web-based application that predicts the likelihood of COVID-19 infection based on user-input symptoms and risk factors. It uses machine learning models to make predictions and provides visualizations for data analysis.

## Key Features
- Interactive web interface for inputting symptoms and risk factors
- Machine learning-based prediction of COVID-19 likelihood
- Data visualizations including correlation heatmaps and feature importance charts
- A/B testing of different machine learning models
- Exploratory Data Analysis (EDA) functionality

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

## Project Structure
- `app.py`: Main Flask application
- `advanced_model.py`: Script for training and saving multiple ML models
- `train_model.py`: Script for training and saving a decision tree model
- `eda_visualization.py`: Functions for EDA and generating visualizations
- `create_mock_dataset.py`: Script to generate a mock COVID-19 dataset
- HTML templates:
  - `index.html`: Main page for symptom input
  - `result.html`: Page displaying prediction results
  - `understand.html`: Page explaining the prediction model and showing visualizations

## Setup and Installation
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run `python create_mock_dataset.py` to generate the mock dataset
4. Run `python advanced_model.py` to train and save the ML models
5. Start the Flask app: `python app.py`
6. Access the application at `http://localhost:5000`

## Future Enhancements
- Implement user authentication and data storage
- Deploy the application to a cloud platform
- Integrate real-time COVID-19 data streams
- Enhance the front-end with a JavaScript framework

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](link-to-issues-page) if you want to contribute.

## Author
[Your Name]

## License
This project is licensed under the [MIT License](link-to-license).