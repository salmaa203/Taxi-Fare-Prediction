# Taxi Fare Prediction System

**Associated with Cellula Technologies**

## Overview

Taxi Fare Prediction System is an end-to-end Machine Learning deployment project designed to predict taxi fares based on trip details, temporal features, passenger count, and geographical information.

The project covers the complete Machine Learning workflow, from data preprocessing and feature engineering to model training, saving, and deployment as an interactive web application.

## Machine Learning Pipeline

The trained model uses custom feature engineering to extract meaningful information from the trip data, including:

- Trip distance using the Haversine formula
- Bearing between pickup and dropoff locations
- Distance to major New York City landmarks and airports
- Hour, day, month, weekday, and year
- Weekend, night, and rush-hour indicators
- Passenger count
- Car condition
- Weather
- Traffic conditions

The numerical features are scaled using a trained `StandardScaler` before being passed to the Machine Learning model.

## Deployment

The application was initially developed using **Flask** and later deployed as an interactive **Streamlit** web application on Streamlit Community Cloud.

Users can:

1. Select pickup and dropoff locations directly from an interactive Folium map.
2. Enter the number of passengers.
3. Select the car condition.
4. The application automatically determines the current date and time based on the **New York timezone**, matching the timezone used by the training dataset.
5. The application calculates the required geographical features automatically.
6. The trained Machine Learning model predicts the estimated taxi fare instantly.

## Key Features

- End-to-end Machine Learning deployment
- Interactive NYC map for route selection
- Automatic trip distance calculation
- Automatic bearing calculation
- NYC landmark and airport distance features
- Automatic New York local date and time
- Feature scaling
- Real-time taxi fare prediction
- Interactive Streamlit interface
- Cloud deployment using Streamlit Community Cloud

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- Streamlit
- Folium
- Streamlit-Folium
- Geopy
- Timezone handling with `tzdata`
  
## Live Demo

[![Open App](https://img.shields.io/badge/Live%20Demo-Open%20App-blue?style=for-the-badge)](https://taxi-fare-prediction-g9tm7nawkeq8jc7seimyxo.streamlit.app/)
