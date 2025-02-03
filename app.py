import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import pandas as pd
import requests
from shapely.geometry import Polygon
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Title
st.title("🌤️ Weather Prediction Dashboard")

# Initialize Folium map with drawing tools
m = folium.Map(location=[0, 0], zoom_start=2)
Draw(export=True, draw_options={"rectangle": True, "polygon": True}).add_to(m)

# Display the map and capture drawn data
output = st_folium(m, width=700, height=500, key="map")

# Extract coordinates from the drawn shape
coordinates = []
if "all_drawings" in output and output["all_drawings"]:
    for drawing in output["all_drawings"]:
        if drawing["geometry"]["type"] == "Polygon":
            # Get the first polygon's coordinates (simplified)
            coords = drawing["geometry"]["coordinates"][0]
            coordinates = [(lat, lon) for lon, lat in coords]  # Folium uses (lat, lon)

# Show extracted coordinates (optional)
if coordinates:
    st.write("Selected Area Coordinates:", coordinates)

    # Convert to a bounding box (min/max lat/lon)
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    bbox = {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons)
    }
    st.write("Bounding Box:", bbox)
else:
    st.warning("Draw a polygon/rectangle on the map to select an area!")

# Function to fetch historical weather data
@st.cache_data
def fetch_historical_weather(latitude, longitude, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["daily"])
    return df

# Date input for historical data
start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Fetch weather data for the centroid of the drawn area
if coordinates:
    # Create a polygon from the coordinates
    polygon = Polygon(coordinates)
    centroid = polygon.centroid
    latitude, longitude = centroid.y, centroid.x  # Shapely uses (x=lon, y=lat)

    st.write(f"Centroid Coordinates: Latitude = {latitude}, Longitude = {longitude}")

    if st.button("Fetch Weather for Centroid"):
        df = fetch_historical_weather(latitude, longitude, start_date, end_date)
        st.subheader("Historical Weather Data")
        st.dataframe(df)

        # Plot temperature
        st.line_chart(df.set_index("time")["temperature_2m_max"])

        # Prepare data for Prophet
        df_prophet = df[["time", "temperature_2m_max"]].rename(columns={"time": "ds", "temperature_2m_max": "y"})

        # Train Prophet model
        model = Prophet()
        model.fit(df_prophet.dropna())

        # Predict for the next 90 days
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        # Show forecast
        st.subheader("Temperature Forecast for the Next 90 Days")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

        # Plot forecast
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)

# Fetch weather data for multiple points in the bounding box (optional)
if coordinates and st.button("Fetch Weather for Entire Area"):
    # Generate grid points within the bounding box
    lats = np.linspace(bbox["min_lat"], bbox["max_lat"], num=3)  # 3x3 grid
    lons = np.linspace(bbox["min_lon"], bbox["max_lon"], num=3)
    
    # Fetch data for all points and average
    all_temps = []
    for lat in lats:
        for lon in lons:
            df = fetch_historical_weather(lat, lon, start_date, end_date)
            all_temps.append(df["temperature_2m_max"].mean())
    
    st.write("Average Temperature in Area:", np.mean(all_temps))