from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load Model and Scaler
model = joblib.load("TaxiFarePredictionModel.pkl")
scaler = joblib.load("TaxiFareScaler.pkl")

# Helper functions for calculations
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    return np.degrees(initial_bearing)

# Landmarks
JFK_COORD = (40.6413, -73.7781)
EWR_COORD = (40.6895, -74.1745)
LGA_COORD = (40.7769, -73.8740)
SOL_COORD = (40.6892, -74.0445)
NYC_COORD = (40.7128, -74.0060)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        passenger_count = int(request.form["passenger_count"])
        hour = int(request.form["hour"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        weekday = int(request.form["weekday"])
        year = int(request.form["year"])

        pickup_latitude = float(request.form.get("pickup_latitude", 0))
        pickup_longitude = float(request.form.get("pickup_longitude", 0))
        dropoff_latitude = float(request.form.get("dropoff_latitude", 0))
        dropoff_longitude = float(request.form.get("dropoff_longitude", 0))

        car_condition = int(request.form["car_condition"])
        weather = int(request.form["weather"])
        traffic = int(request.form["traffic"])

        if pickup_latitude == 0 or dropoff_latitude == 0:
            return render_template(
                "index.html",
                prediction_text="Please select both Pickup and Dropoff locations on the map."
            )

        # Distance & Feature Engineering
        distance = haversine_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
        bearing = calculate_bearing(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

        jfk_dist = haversine_distance(pickup_latitude, pickup_longitude, JFK_COORD[0], JFK_COORD[1])
        ewr_dist = haversine_distance(pickup_latitude, pickup_longitude, EWR_COORD[0], EWR_COORD[1])
        lga_dist = haversine_distance(pickup_latitude, pickup_longitude, LGA_COORD[0], LGA_COORD[1])
        sol_dist = haversine_distance(pickup_latitude, pickup_longitude, SOL_COORD[0], SOL_COORD[1])
        nyc_dist = haversine_distance(pickup_latitude, pickup_longitude, NYC_COORD[0], NYC_COORD[1])

        is_weekend = 1 if weekday in [5, 6] else 0
        is_night = 1 if hour >= 22 or hour <= 5 else 0
        is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0

        data = pd.DataFrame({
            "Car Condition": [car_condition],
            "Weather": [weather],
            "Traffic Condition": [traffic],
            "pickup_longitude": [pickup_longitude],
            "pickup_latitude": [pickup_latitude],
            "dropoff_longitude": [dropoff_longitude],
            "dropoff_latitude": [dropoff_latitude],
            "passenger_count": [passenger_count],
            "hour": [hour],
            "day": [day],
            "month": [month],
            "weekday": [weekday],
            "year": [year],
            "jfk_dist": [jfk_dist],
            "ewr_dist": [ewr_dist],
            "lga_dist": [lga_dist],
            "sol_dist": [sol_dist],
            "nyc_dist": [nyc_dist],
            "distance": [distance],
            "bearing": [bearing],
            "is_weekend": [is_weekend],
            "is_night": [is_night],
            "is_rush_hour": [is_rush_hour]
        })

        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)

        return render_template(
            "index.html",
            prediction_text=f"Estimated Taxi Fare: ${prediction[0]:.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )