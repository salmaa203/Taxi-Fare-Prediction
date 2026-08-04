import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import Geocoder
from streamlit_folium import st_folium
from datetime import datetime

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Taxi Fare Prediction",
    layout="wide"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #172554);
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    color: #ffffff;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 35px;
}

.section-title {
    color: #60a5fa;
    font-size: 25px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 15px;
}

label {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

input {
    color: #111827 !important;
}

div[data-baseweb="select"] {
    color: #111827 !important;
}

div[data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    border-radius: 10px !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #f8fafc !important;
    border-radius: 10px !important;
}

.stButton > button {
    width: 100%;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-size: 17px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

.result-card {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 30px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.result-title {
    color: white;
    font-size: 22px;
    font-weight: 500;
    margin-bottom: 10px;
}

.result-price {
    color: white;
    font-size: 42px;
    font-weight: 700;
}

.info-card {
    background: rgba(30, 58, 95, 0.8);
    padding: 18px;
    border-radius: 12px;
    margin-top: 12px;
    color: #dbeafe;
    font-size: 17px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# Load Model and Scaler
# =========================================================

@st.cache_resource
def load_models():
    model = joblib.load("TaxiFarePredictionModel.pkl")
    scaler = joblib.load("TaxiFareScaler.pkl")
    return model, scaler


try:
    model, scaler = load_models()
except Exception as e:
    st.error("Could not load the model or scaler. Check if .pkl files are uploaded to GitHub.")
    st.error(str(e))
    st.stop()


# =========================================================
# Helper Functions
# =========================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )
    initial_bearing = np.arctan2(x, y)
    return np.degrees(initial_bearing)


# =========================================================
# Landmarks
# =========================================================

JFK_COORD = (40.6413, -73.7781)
EWR_COORD = (40.6895, -74.1745)
LGA_COORD = (40.7769, -73.8740)
SOL_COORD = (40.6892, -74.0445)
NYC_COORD = (40.7128, -74.0060)


# =========================================================
# Title
# =========================================================

st.markdown(
    '<div class="main-title">Taxi Fare Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Select your pickup and dropoff locations on the map '
    'and enter the trip details to estimate the taxi fare.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# Trip Information
# =========================================================

st.markdown(
    '<div class="section-title">Trip Information</div>',
    unsafe_allow_html=True
)

# Safe DateTime handling for Streamlit Cloud
try:
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Africa/Cairo"))
except Exception:
    now = datetime.now()

hour = now.hour
day = now.day
month = now.month
year = now.year
weekday = now.weekday()

weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_name = weekday_names[weekday]

col1, col2 = st.columns(2)

with col1:
    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=8,
        value=1,
        step=1
    )

with col2:
    st.markdown(
        f"""
        <div class="info-card">
            Current Date & Time<br>
            <strong>
                {weekday_name}, {day}/{month}/{year}
                — {hour:02d}:{now.minute:02d}
            </strong>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# Trip Conditions
# =========================================================

st.markdown(
    '<div class="section-title">Trip Conditions</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    car_condition_name = st.selectbox(
        "Car Condition",
        [
            "Bad",
            "Excellent",
            "Good",
            "Very Good"
        ]
    )
    car_condition_map = {
        "Bad": 0,
        "Excellent": 1,
        "Good": 2,
        "Very Good": 3
    }
    car_condition = car_condition_map[car_condition_name]

with col2:
    weather_name = st.selectbox(
        "Weather",
        [
            "Cloudy",
            "Rainy",
            "Stormy",
            "Sunny",
            "Windy"
        ]
    )
    weather_map = {
        "Cloudy": 0,
        "Rainy": 1,
        "Stormy": 2,
        "Sunny": 3,
        "Windy": 4
    }
    weather = weather_map[weather_name]

with col3:
    traffic_name = st.selectbox(
        "Traffic Condition",
        [
            "Congested Traffic",
            "Dense Traffic",
            "Flow Traffic"
        ]
    )
    traffic_map = {
        "Congested Traffic": 0,
        "Dense Traffic": 1,
        "Flow Traffic": 2
    }
    traffic = traffic_map[traffic_name]


# =========================================================
# Map
# =========================================================

st.markdown(
    '<div class="section-title">Select Route</div>',
    unsafe_allow_html=True
)

st.write(
    "Click once on the map to select Pickup, "
    "then click again to select Dropoff. You can also use the search icon on the map to find locations."
)

if "pickup" not in st.session_state:
    st.session_state.pickup = None

if "dropoff" not in st.session_state:
    st.session_state.dropoff = None

m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11,
    tiles="OpenStreetMap"
)

# 🔍 إضافة زر البحث
Geocoder().add_to(m)

# Pickup Marker
if st.session_state.pickup is not None:
    folium.Marker(
        st.session_state.pickup,
        tooltip="Pickup",
        popup="Pickup Location",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(m)

# Dropoff Marker
if st.session_state.dropoff is not None:
    folium.Marker(
        st.session_state.dropoff,
        tooltip="Dropoff",
        popup="Dropoff Location",
        icon=folium.Icon(
            color="red",
            icon="stop"
        )
    ).add_to(m)

map_data = st_folium(
    m,
    width=None,
    height=500,
    key="taxi_map"
)


# =========================================================
# Handle Map Click
# =========================================================

if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    clicked_location = (clicked_lat, clicked_lon)

    if st.session_state.pickup is None:
        st.session_state.pickup = clicked_location
        st.rerun()
    elif st.session_state.dropoff is None and clicked_location != st.session_state.pickup:
        st.session_state.dropoff = clicked_location
        st.rerun()


# =========================================================
# Reset Locations
# =========================================================

if st.button("Reset Locations"):
    st.session_state.pickup = None
    st.session_state.dropoff = None
    st.rerun()


# =========================================================
# Prediction
# =========================================================

st.markdown(
    '<div class="section-title">Prediction</div>',
    unsafe_allow_html=True
)

if st.button("Predict Taxi Fare"):
    if (
        st.session_state.pickup is None
        or st.session_state.dropoff is None
    ):
        st.warning("Please select both Pickup and Dropoff locations on the map.")
    else:
        try:
            pickup_latitude = st.session_state.pickup[0]
            pickup_longitude = st.session_state.pickup[1]
            dropoff_latitude = st.session_state.dropoff[0]
            dropoff_longitude = st.session_state.dropoff[1]

            distance = haversine_distance(
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude
            )
            bearing = calculate_bearing(
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude
            )

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
            fare = float(prediction[0])

            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-title">Estimated Taxi Fare</div>'
                f'<div class="result-price">${fare:.2f}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="info-card">Trip Distance: {distance:.2f} km</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="info-card">Bearing: {bearing:.2f} degrees</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")


# =========================================================
# Footer
# =========================================================

st.markdown(
    '<div class="footer">Machine Learning Deployment Project</div>',
    unsafe_allow_html=True
)
