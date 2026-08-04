import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Taxi Fare Prediction",
    page_icon="Taxi",
    layout="centered"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 50%,
            #0f172a 100%
        );
    }

    /* Main content */
    .block-container {
        max-width: 1000px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #f8fafc;
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

    /* Section titles */
    .section-title {
        color: #e2e8f0;
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 18px;
        border-left: 4px solid #3b82f6;
        padding-left: 12px;
    }

    /* Labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    /* Inputs */
    div[data-baseweb="input"] {
        background-color: #1e293b;
        border-radius: 10px;
    }

    div[data-baseweb="select"] {
        background-color: #1e293b;
        border-radius: 10px;
    }

    input {
        color: #f8fafc !important;
    }

    /* Select text */
    div[data-baseweb="select"] * {
        color: #f8fafc !important;
    }

    /* Map card */
    .map-card {
        background: #172033;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .map-description {
        color: #cbd5e1;
        font-size: 15px;
        margin-bottom: 15px;
    }

    /* Location status */
    .location-box {
        background: #172033;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px 16px;
        color: #cbd5e1;
        margin-top: 8px;
    }

    /* Prediction result */
    .result-box {
        background: linear-gradient(
            135deg,
            #1d4ed8,
            #2563eb
        );
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
    }

    .result-title {
        color: #bfdbfe;
        font-size: 16px;
        margin-bottom: 5px;
    }

    .result-price {
        color: white;
        font-size: 36px;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px;
        font-size: 17px;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        border: none;
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #334155;
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


model, scaler = load_models()


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
# Header
# =========================================================

st.markdown(
    '<div class="main-title">Taxi Fare Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Select your route and trip details to estimate the taxi fare.'
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

col1, col2 = st.columns(2)


with col1:

    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=8,
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

    weekday_name = st.selectbox(
        "Weekday",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    weekday = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }[weekday_name]


    year = st.number_input(
        "Year",
        min_value=2009,
        max_value=2030,
        value=2026,
        step=1
    )

    car_condition_name = st.selectbox(
        "Car Condition",
        [
            "Bad",
            "Excellent",
            "Good",
            "Very Good"
        ]
    )

    car_condition = {
        "Bad": 0,
        "Excellent": 1,
        "Good": 2,
        "Very Good": 3
    }[car_condition_name]


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

    weather = {
        "Cloudy": 0,
        "Rainy": 1,
        "Stormy": 2,
        "Sunny": 3,
        "Windy": 4
    }[weather_name]


    traffic_name = st.selectbox(
        "Traffic Condition",
        [
            "Congested Traffic",
            "Dense Traffic",
            "Flow Traffic"
        ]
    )

    traffic = {
        "Congested Traffic": 0,
        "Dense Traffic": 1,
        "Flow Traffic": 2
    }[traffic_name]


# =========================================================
# Map
# =========================================================

st.markdown(
    '<div class="section-title">Select Route</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="map-card">'
    '<div class="map-description">'
    'Click once on the map to select your Pickup location, '
    'then click again to select your Dropoff location.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# Session State
# =========================================================

if "pickup" not in st.session_state:
    st.session_state.pickup = None

if "dropoff" not in st.session_state:
    st.session_state.dropoff = None


# =========================================================
# Create Map
# =========================================================

m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11,
    tiles="OpenStreetMap"
)


# Pickup Marker

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


# Dropoff Marker

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


# Display Map

map_data = st_folium(
    m,
    width=900,
    height=500,
    returned_objects=["last_clicked"]
)


st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# Handle Map Click
# =========================================================

if map_data["last_clicked"] is not None:

    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    if st.session_state.pickup is None:

        st.session_state.pickup = (
            clicked_lat,
            clicked_lon
        )

        st.rerun()


    elif st.session_state.dropoff is None:

        st.session_state.dropoff = (
            clicked_lat,
            clicked_lon
        )

        st.rerun()


# =========================================================
# Selected Locations
# =========================================================

if st.session_state.pickup is not None:

    st.markdown(
        f"""
        <div class="location-box">
        <b>Pickup:</b>
        {st.session_state.pickup[0]:.5f},
        {st.session_state.pickup[1]:.5f}
        </div>
        """,
        unsafe_allow_html=True
    )


if st.session_state.dropoff is not None:

    st.markdown(
        f"""
        <div class="location-box">
        <b>Dropoff:</b>
        {st.session_state.dropoff[0]:.5f},
        {st.session_state.dropoff[1]:.5f}
        </div>
        """,
        unsafe_allow_html=True
    )


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
    '<div class="section-title">Fare Prediction</div>',
    unsafe_allow_html=True
)


if st.button(
    "Predict Taxi Fare",
    use_container_width=True
):

    try:

        if (
            st.session_state.pickup is None
            or st.session_state.dropoff is None
        ):

            st.warning(
                "Please select both Pickup and Dropoff locations."
            )

            st.stop()


        # =================================================
        # Coordinates
        # =================================================

        pickup_latitude = st.session_state.pickup[0]
        pickup_longitude = st.session_state.pickup[1]

        dropoff_latitude = st.session_state.dropoff[0]
        dropoff_longitude = st.session_state.dropoff[1]


        # =================================================
        # Distance
        # =================================================

        distance = haversine_distance(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude
        )


        # =================================================
        # Bearing
        # =================================================

        bearing = calculate_bearing(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude
        )


        # =================================================
        # Landmark Distances
        # =================================================

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


        # =================================================
        # Feature Engineering
        # =================================================

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


        # =================================================
        # Create DataFrame
        # =================================================

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


        # =================================================
        # Scaling
        # =================================================

        data_scaled = scaler.transform(data)


        # =================================================
        # Prediction
        # =================================================

        prediction = model.predict(data_scaled)

        fare = float(prediction[0])


        # =================================================
        # Result
        # =================================================

        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-title">
                    Estimated Taxi Fare
                </div>

                <div class="result-price">
                    ${fare:.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.info(
            f"Trip Distance: {distance:.2f} km"
        )


        st.info(
            f"Bearing: {bearing:.2f} degrees"
        )


    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer">
        Taxi Fare Prediction System<br>
        Machine Learning Deployment Project
    </div>
    """,
    unsafe_allow_html=True
)
