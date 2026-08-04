import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Taxi Fare Prediction",
    page_icon="🚕",
    layout="centered"
)


# =========================
# Load Model and Scaler
# =========================

@st.cache_resource
def load_models():

    model = joblib.load("TaxiFarePredictionModel.pkl")
    scaler = joblib.load("TaxiFareScaler.pkl")

    return model, scaler


model, scaler = load_models()


# =========================
# Helper Functions
# =========================

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


# =========================
# Landmarks
# =========================

JFK_COORD = (40.6413, -73.7781)
EWR_COORD = (40.6895, -74.1745)
LGA_COORD = (40.7769, -73.8740)
SOL_COORD = (40.6892, -74.0445)
NYC_COORD = (40.7128, -74.0060)


# =========================
# Title
# =========================

st.title("🚕 Taxi Fare Prediction")

st.write(
    "Select your pickup and dropoff locations on the map "
    "and enter the trip details to estimate the taxi fare."
)


# =========================
# Trip Information
# =========================

st.subheader("Trip Information")

col1, col2 = st.columns(2)

with col1:

    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1,
        step=1
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )


with col2:

    weekday = st.number_input(
        "Weekday",
        min_value=0,
        max_value=6,
        value=0,
        step=1,
        help="0 = Monday, 6 = Sunday"
    )

    year = st.number_input(
        "Year",
        min_value=2009,
        max_value=2030,
        value=2026,
        step=1
    )

    car_condition = st.number_input(
        "Car Condition",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

    weather = st.number_input(
        "Weather",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

    traffic = st.number_input(
        "Traffic Condition",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )


# =========================
# Map
# =========================

st.subheader("📍 Select Pickup & Dropoff Locations")

st.write(
    "Click once on the map for Pickup location, "
    "then click again for Dropoff location."
)


# Initialize session state

if "pickup" not in st.session_state:
    st.session_state.pickup = None

if "dropoff" not in st.session_state:
    st.session_state.dropoff = None


# Create map centered on NYC

m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11
)


# Add existing pickup marker

if st.session_state.pickup is not None:

    folium.Marker(
        st.session_state.pickup,
        popup="Pickup Location",
        tooltip="Pickup",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(m)


# Add existing dropoff marker

if st.session_state.dropoff is not None:

    folium.Marker(
        st.session_state.dropoff,
        popup="Dropoff Location",
        tooltip="Dropoff",
        icon=folium.Icon(
            color="red",
            icon="stop"
        )
    ).add_to(m)


# Display map

map_data = st_folium(
    m,
    width=700,
    height=500,
    returned_objects=["last_clicked"]
)


# =========================
# Handle Map Click
# =========================

if map_data["last_clicked"] is not None:

    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    # First click = Pickup

    if st.session_state.pickup is None:

        st.session_state.pickup = (
            clicked_lat,
            clicked_lon
        )

        st.rerun()


    # Second click = Dropoff

    elif st.session_state.dropoff is None:

        st.session_state.dropoff = (
            clicked_lat,
            clicked_lon
        )

        st.rerun()


# =========================
# Display Selected Locations
# =========================

if st.session_state.pickup is not None:

    st.success(
        f"Pickup selected: "
        f"{st.session_state.pickup[0]:.5f}, "
        f"{st.session_state.pickup[1]:.5f}"
    )


if st.session_state.dropoff is not None:

    st.error(
        f"Dropoff selected: "
        f"{st.session_state.dropoff[0]:.5f}, "
        f"{st.session_state.dropoff[1]:.5f}"
    )


# Reset locations button

if st.button("🔄 Reset Locations"):

    st.session_state.pickup = None
    st.session_state.dropoff = None

    st.rerun()


# =========================
# Prediction
# =========================

if st.button(
    "🚕 Predict Taxi Fare",
    use_container_width=True
):

    try:

        # Check locations

        if (
            st.session_state.pickup is None
            or st.session_state.dropoff is None
        ):

            st.warning(
                "Please select both Pickup and Dropoff "
                "locations on the map."
            )

            st.stop()


        # Get coordinates

        pickup_latitude = st.session_state.pickup[0]
        pickup_longitude = st.session_state.pickup[1]

        dropoff_latitude = st.session_state.dropoff[0]
        dropoff_longitude = st.session_state.dropoff[1]


        # =========================
        # Distance
        # =========================

        distance = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude
        )


        # =========================
        # Bearing
        # =========================

        bearing = calculate_bearing(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude
        )


        # =========================
        # Landmark Distances
        # =========================

        jfk_dist = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            JFK_COORD[0],
            JFK_COORD[1]
        )

        ewr_dist = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            EWR_COORD[0],
            EWR_COORD[1]
        )

        lga_dist = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            LGA_COORD[0],
            LGA_COORD[1]
        )

        sol_dist = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            SOL_COORD[0],
            SOL_COORD[1]
        )

        nyc_dist = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            NYC_COORD[0],
            NYC_COORD[1]
        )


        # =========================
        # Feature Engineering
        # =========================

        is_weekend = (
            1 if weekday in [5, 6]
            else 0
        )

        is_night = (
            1 if hour >= 22 or hour <= 5
            else 0
        )

        is_rush_hour = (
            1 if hour in [7, 8, 9, 16, 17, 18]
            else 0
        )


        # =========================
        # Create DataFrame
        # =========================

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


        # =========================
        # Scaling
        # =========================

        data_scaled = scaler.transform(data)


        # =========================
        # Prediction
        # =========================

        prediction = model.predict(data_scaled)

        fare = float(prediction[0])


        # =========================
        # Results
        # =========================

        st.success(
            f"💰 Estimated Taxi Fare: ${fare:.2f}"
        )

        st.info(
            f"🚕 Trip Distance: {distance:.2f} km"
        )

        st.info(
            f"🧭 Bearing: {bearing:.2f}°"
        )


    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
