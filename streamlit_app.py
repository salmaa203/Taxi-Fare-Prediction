import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Taxi Fare Prediction",
    page_icon="Taxi",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1733 0%,
            #122653 50%,
            #0b1733 100%
        );
    }

    /* Main content width */
    .block-container {
        max-width: 1200px;
        padding-top: 40px;
        padding-bottom: 50px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #c9d6ee;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Section titles */
    h2 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Labels */
    label {
        color: #e8eef9 !important;
        font-weight: 500 !important;
    }

    /* Input containers */
    div[data-baseweb="input"] {
        background-color: #f4f6fa !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] {
        background-color: #f4f6fa !important;
        border-radius: 10px !important;
    }

    /* Input text */
    input {
        color: #17233f !important;
    }

    /* Select text */
    div[data-baseweb="select"] div {
        color: #17233f !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 17px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .prediction-title {
        color: white;
        font-size: 20px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .prediction-price {
        color: white;
        font-size: 42px;
        font-weight: 700;
    }

    /* Information boxes */
    .info-box {
        background: #122653;
        border: 1px solid #244274;
        border-radius: 12px;
        padding: 18px;
        color: #dbe7fb;
        margin-top: 12px;
        font-size: 17px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9fb1cf;
        margin-top: 45px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
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
# LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("TaxiFarePredictionModel.pkl")

    scaler = joblib.load("TaxiFareScaler.pkl")

    return model, scaler


model, scaler = load_models()


# =========================================================
# SESSION STATE
# =========================================================

if "pickup" not in st.session_state:
    st.session_state.pickup = None

if "dropoff" not in st.session_state:
    st.session_state.dropoff = None

if "last_click" not in st.session_state:
    st.session_state.last_click = None


# =========================================================
# HELPER FUNCTIONS
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
        +
        np.cos(lat1)
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
        -
        np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    initial_bearing = np.arctan2(x, y)

    bearing = np.degrees(initial_bearing)

    return bearing


# =========================================================
# LANDMARK COORDINATES
# =========================================================

JFK_COORD = (40.6413, -73.7781)

EWR_COORD = (40.6895, -74.1745)

LGA_COORD = (40.7769, -73.8740)

SOL_COORD = (40.6892, -74.0445)

NYC_COORD = (40.7128, -74.0060)


# =========================================================
# TRIP INFORMATION
# =========================================================

st.header("Trip Information")

col1, col2, col3 = st.columns(3)


with col1:

    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=8,
        value=1,
        step=1
    )


with col2:

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )


with col3:

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15,
        step=1
    )


col1, col2, col3 = st.columns(3)


with col1:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6,
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


with col3:

    year = st.number_input(
        "Year",
        min_value=2009,
        max_value=2030,
        value=2009,
        step=1
    )


# Convert weekday name to number
weekday_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

weekday = weekday_mapping[weekday_name]


# =========================================================
# TRIP CONDITIONS
# =========================================================

st.header("Trip Conditions")

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


with col3:

    traffic_name = st.selectbox(
        "Traffic Condition",
        [
            "Congested Traffic",
            "Dense Traffic",
            "Flow Traffic"
        ]
    )


# Convert categorical values to the same encoding
# used during model training

car_condition_mapping = {
    "Bad": 0,
    "Excellent": 1,
    "Good": 2,
    "Very Good": 3
}

weather_mapping = {
    "Cloudy": 0,
    "Rainy": 1,
    "Stormy": 2,
    "Sunny": 3,
    "Windy": 4
}

traffic_mapping = {
    "Congested Traffic": 0,
    "Dense Traffic": 1,
    "Flow Traffic": 2
}

car_condition = car_condition_mapping[car_condition_name]

weather = weather_mapping[weather_name]

traffic = traffic_mapping[traffic_name]


# =========================================================
# MAP
# =========================================================

st.header("Select Pickup and Dropoff Locations")

st.write(
    "Click once on the map for Pickup, then click again for Dropoff."
)


# Create map
m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11,
    control_scale=True
)


# Add pickup marker
if st.session_state.pickup is not None:

    folium.Marker(
        location=st.session_state.pickup,
        popup="Pickup Location",
        tooltip="Pickup",
        icon=folium.Icon(
            color="blue",
            icon="info-sign"
        )
    ).add_to(m)


# Add dropoff marker
if st.session_state.dropoff is not None:

    folium.Marker(
        location=st.session_state.dropoff,
        popup="Dropoff Location",
        tooltip="Dropoff",
        icon=folium.Icon(
            color="red",
            icon="info-sign"
        )
    ).add_to(m)


# Add route line
if (
    st.session_state.pickup is not None
    and
    st.session_state.dropoff is not None
):

    folium.PolyLine(
        [
            st.session_state.pickup,
            st.session_state.dropoff
        ],
        color="#2563eb",
        weight=4,
        opacity=0.8
    ).add_to(m)


map_data = st_folium(
    m,
    width=None,
    height=500,
    returned_objects=["last_clicked"]
)


# =========================================================
# HANDLE MAP CLICK
# =========================================================

if map_data and map_data.get("last_clicked"):

    clicked_lat = map_data["last_clicked"]["lat"]

    clicked_lon = map_data["last_clicked"]["lng"]

    current_click = (
        round(clicked_lat, 6),
        round(clicked_lon, 6)
    )


    # Prevent the same click from being processed twice
    if current_click != st.session_state.last_click:

        st.session_state.last_click = current_click

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

        else:

            # If both already exist,
            # start a new route
            st.session_state.pickup = (
                clicked_lat,
                clicked_lon
            )

            st.session_state.dropoff = None

            st.rerun()


# =========================================================
# DISPLAY SELECTED LOCATIONS
# =========================================================

col1, col2 = st.columns(2)


with col1:

    if st.session_state.pickup:

        st.markdown(
            f"""
            <div class="info-box">
            <strong>Pickup Location</strong><br>
            Latitude: {st.session_state.pickup[0]:.6f}<br>
            Longitude: {st.session_state.pickup[1]:.6f}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info("Pickup location not selected yet.")


with col2:

    if st.session_state.dropoff:

        st.markdown(
            f"""
            <div class="info-box">
            <strong>Dropoff Location</strong><br>
            Latitude: {st.session_state.dropoff[0]:.6f}<br>
            Longitude: {st.session_state.dropoff[1]:.6f}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info("Dropoff location not selected yet.")


# =========================================================
# CLEAR LOCATIONS BUTTON
# =========================================================

if st.button("Clear Selected Locations"):

    st.session_state.pickup = None

    st.session_state.dropoff = None

    st.session_state.last_click = None

    st.rerun()


# =========================================================
# PREDICTION
# =========================================================

st.header("Fare Prediction")


if st.button("Predict Taxi Fare"):

    if (
        st.session_state.pickup is None
        or
        st.session_state.dropoff is None
    ):

        st.error(
            "Please select both Pickup and Dropoff locations on the map."
        )

    else:

        try:

            # -------------------------------------------------
            # Get coordinates
            # -------------------------------------------------

            pickup_latitude = st.session_state.pickup[0]

            pickup_longitude = st.session_state.pickup[1]

            dropoff_latitude = st.session_state.dropoff[0]

            dropoff_longitude = st.session_state.dropoff[1]


            # -------------------------------------------------
            # Calculate Trip Distance
            # -------------------------------------------------

            distance = haversine_distance(
                pickup_latitude,
                pickup_longitude,
                dropoff_latitude,
                dropoff_longitude
            )


            # -------------------------------------------------
            # Calculate Bearing
            # -------------------------------------------------

            bearing = calculate_bearing(
                pickup_latitude,
                pickup_longitude,
                dropoff_latitude,
                dropoff_longitude
            )


            # -------------------------------------------------
            # Landmark Distances
            # -------------------------------------------------

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


            # -------------------------------------------------
            # Feature Engineering
            # -------------------------------------------------

            is_weekend = (
                1 if weekday in [5, 6]
                else 0
            )

            is_night = (
                1 if hour >= 22 or hour <= 5
                else 0
            )

            is_rush_hour = (
                1
                if hour in [7, 8, 9, 16, 17, 18]
                else 0
            )


            # -------------------------------------------------
            # Create DataFrame
            # -------------------------------------------------

            data = pd.DataFrame({

                "Car Condition": [car_condition],

                "Weather": [weather],

                "Traffic Condition": [traffic],

                "pickup_longitude": [
                    pickup_longitude
                ],

                "pickup_latitude": [
                    pickup_latitude
                ],

                "dropoff_longitude": [
                    dropoff_longitude
                ],

                "dropoff_latitude": [
                    dropoff_latitude
                ],

                "passenger_count": [
                    passenger_count
                ],

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


            # -------------------------------------------------
            # Scaling
            # -------------------------------------------------

            data_scaled = scaler.transform(data)


            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            prediction = model.predict(data_scaled)

            fare = float(prediction[0])


            # -------------------------------------------------
            # Display Result
            # -------------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-box">

                    <div class="prediction-title">
                        Estimated Taxi Fare
                    </div>

                    <div class="prediction-price">
                        ${fare:.2f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # -------------------------------------------------
            # Trip Information
            # -------------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.markdown(
                    f"""
                    <div class="info-box">
                    <strong>Trip Distance</strong><br>
                    {distance:.2f} km
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                st.markdown(
                    f"""
                    <div class="info-box">
                    <strong>Bearing</strong><br>
                    {bearing:.2f} degrees
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        except Exception as e:

            st.error(
                f"Prediction Error: {str(e)}"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Machine Learning Deployment Project
        <br>
        Built with Python, Scikit-Learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
