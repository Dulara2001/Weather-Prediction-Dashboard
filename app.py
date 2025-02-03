import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
import plotly.express as px
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from datetime import datetime
from geopy.geocoders import Nominatim
import time

# Initialize session state
if "historical_data" not in st.session_state:
    st.session_state.historical_data = None
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None

# Page configuration
st.set_page_config(layout="wide")
st.title("🌦️ Smart Weather Analysis Dashboard")

# ================= Layout Structure =================
left_col, right_col = st.columns([1, 1])

# ================= Left Column (Map + Historical Data) =================
with left_col:
    st.subheader("🗺️ Select Area")
    m = folium.Map(location=[0, 0], zoom_start=2)
    Draw(export=True, draw_options={"rectangle": True}).add_to(m)
    map_data = st_folium(m, width=500, height=400)
    
    # Historical Data Section
    if st.session_state.historical_data is not None:
        st.subheader("📜 Historical Weather Data")
        
        # Rename columns for better understanding
        historical_df = st.session_state.historical_data.rename(columns={
            "time": "Date",
            "temperature_2m_max": "Max Temperature (°C)",
            "precipitation_sum": "Total Precipitation (mm)"
        })
        
        st.dataframe(historical_df, use_container_width=True)
        
        # Temperature Chart
        fig_temp = px.line(
            historical_df,
            x="Date",
            y="Max Temperature (°C)",
            title="Historical Temperature Trend"
        )
        st.plotly_chart(fig_temp, use_container_width=True)

# ================= Right Column (Controls + Forecast) =================
with right_col:
    # Show help message until area is selected
    if not map_data.get("last_active_drawing"):
        st.subheader("ℹ️ Instructions")
        st.info("""
        1. Draw a rectangle on the map
        2. Select date range below
        3. Click 'Fetch Historical Data'
        4. Click 'Predict Next 30 Days'
        """)
    
    # Date Selection
    st.subheader("📅 Date Range Selection")
    start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
    
    # Location Details Section
    if map_data.get("last_active_drawing"):
        st.subheader("📍 Selected Location Details")
        
        # Calculate centroid
        geometry = map_data["last_active_drawing"]["geometry"]
        coordinates = geometry["coordinates"][0]
        lats = [coord[1] for coord in coordinates]
        lons = [coord[0] for coord in coordinates]
        latitude = sum(lats) / len(lats)
        longitude = sum(lons) / len(lons)
        
        # Display coordinates
        st.write(f"**Latitude:** {latitude:.4f}")
        st.write(f"**Longitude:** {longitude:.4f}")
        
        # Reverse geocoding for address details
        try:
            geolocator = Nominatim(user_agent="weather_dashboard")
            location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
            
            if location:
                address = location.raw.get('address', {})
                st.write("**Address Details:**")
                st.write(f"📍 {address.get('road', '')} {address.get('house_number', '')}")
                st.write(f"🏙️ {address.get('city', address.get('town', ''))}")
                st.write(f"🗺️ {address.get('state', '')}, {address.get('country', '')}")
                st.write(f"🌐 {address.get('postcode', '')}")
            else:
                st.warning("Could not retrieve address details for this location")
        
        except Exception as e:
            st.error(f"Error fetching location details: {str(e)}")
            st.info("Note: Location details might not be available for remote areas")
        
        # Add delay to prevent rate limiting
        time.sleep(1)
    
    # Prediction Section
    if st.session_state.forecast_data is not None:
        st.subheader("🔮 Future Weather Forecast")
        
        # Rename forecast columns
        forecast_df = st.session_state.forecast_data.rename(columns={
            "ds": "Date",
            "yhat": "Predicted Temperature (°C)",
            "yhat_lower": "Minimum Estimate",
            "yhat_upper": "Maximum Estimate"
        })
        
        # Filter only future dates
        last_historical_date = pd.to_datetime(st.session_state.historical_data["time"].max())
        future_forecast = forecast_df[forecast_df["Date"] > last_historical_date]
        
        st.dataframe(future_forecast[["Date", "Predicted Temperature (°C)", 
                                    "Minimum Estimate", "Maximum Estimate"]], 
                    use_container_width=True)
        
        # Forecast Visualization
        fig_forecast = px.line(
            future_forecast,
            x="Date",
            y="Predicted Temperature (°C)",
            title="30-Day Temperature Forecast",
            labels={"value": "Temperature (°C)"}
        ).update_layout(showlegend=False)
        
        st.plotly_chart(fig_forecast, use_container_width=True)

# ================= Common Functions =================
def fetch_weather_data(latitude, longitude):
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
    return pd.DataFrame(response.json()["daily"])

# ================= Button Controls =================
if map_data.get("last_active_drawing"):
    # Calculate centroid
    geometry = map_data["last_active_drawing"]["geometry"]
    coordinates = geometry["coordinates"][0]
    lats = [coord[1] for coord in coordinates]
    lons = [coord[0] for coord in coordinates]
    latitude = sum(lats) / len(lats)
    longitude = sum(lons) / len(lons)
    
    # Buttons at bottom
    left_col.button("🗓️ Fetch Historical Data", 
                  on_click=lambda: st.session_state.update({
                      "historical_data": fetch_weather_data(latitude, longitude),
                      "forecast_data": None
                  }))
    
    # Modified prediction button
    if right_col.button("🔮 Predict Next 30 Days"):
        if st.session_state.historical_data is not None:
            try:
                # Prepare data
                df = (st.session_state.historical_data
                      .rename(columns={"time": "ds", "temperature_2m_max": "y"})
                      [["ds", "y"]]
                     )
                
                # Create and fit model
                model = Prophet()
                model.fit(df)
                
                # Generate future dates
                future = model.make_future_dataframe(periods=30)
                
                # Make predictions
                forecast = model.predict(future)
                st.session_state.forecast_data = forecast
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            st.warning("Please fetch historical data first!")