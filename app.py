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
from transformers import pipeline
import re


# Initialize chatbot pipeline 
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Initialize session state
if "historical_data" not in st.session_state:
    st.session_state.historical_data = None
if "forecast_temp" not in st.session_state:
    st.session_state.forecast_temp = None
if "forecast_precip" not in st.session_state:
    st.session_state.forecast_precip = None

# Page configuration
st.set_page_config(layout="wide")
st.title("🌦️ Smart Weather Analysis Dashboard")


# Add chatbot functions 
def get_chat_response(question, context):
    patterns = {
        r"temperature|how hot|how cold": "temperature_2m_max",
        r"rain|precipitation": "precipitation_sum",
        r"wind": "windspeed_10m_max",
        r"forecast|prediction": "forecast"
    }
    
    for pattern, response_key in patterns.items():
        if re.search(pattern, question, re.IGNORECASE):
            return get_data_based_response(response_key)
    
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def get_data_based_response(response_key):
    """
    Generate response based on data type
    """
    # Check if historical data exists and is not empty
    if st.session_state.historical_data is None or st.session_state.historical_data.empty:
        return "Please fetch historical data first!"
    
    latest_data = st.session_state.historical_data.iloc[-1]
    
    responses = {
        "temperature_2m_max": f"Latest temperature: {latest_data['temperature_2m_max']}°C",
        "precipitation_sum": f"Recent precipitation: {latest_data['precipitation_sum']}mm",
        "windspeed_10m_max": f"Wind speed: {latest_data['windspeed_10m_max']}km/h",
        # Fixed forecast check
        "forecast": ("Here's the forecast:" 
                     if (st.session_state.forecast_temp is not None 
                         and not st.session_state.forecast_temp.empty) 
                     else "Generate forecast first!")
    }
    
    return responses.get(response_key, "I can help with temperature, precipitation, wind, and forecasts.")


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
            "temperature_2m_max": "Max Temp (°C)",
            "temperature_2m_min": "Min Temp (°C)",
            "precipitation_sum": "Precipitation (mm)",
            "rain_sum": "Rain (mm)",
            "windspeed_10m_max": "Max Wind (km/h)"
        })
        
        st.dataframe(historical_df, use_container_width=True)
        
        # Tabbed Visualizations
        tab1, tab2, tab3 = st.tabs(["Temperature", "Precipitation", "Wind"])
        
        with tab1:
            fig_temp = px.line(
                historical_df,
                x="Date",
                y=["Max Temp (°C)", "Min Temp (°C)"],
                title="Temperature Trends"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab2:
            fig_precip = px.bar(
                historical_df,
                x="Date",
                y="Precipitation (mm)",
                title="Daily Precipitation"
            )
            st.plotly_chart(fig_precip, use_container_width=True)
        
        with tab3:
            fig_wind = px.line(
                historical_df,
                x="Date",
                y="Max Wind (km/h)",
                title="Wind Speed"
            )
            st.plotly_chart(fig_wind, use_container_width=True)

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
    if st.session_state.forecast_temp is not None and st.session_state.forecast_precip is not None:
        st.subheader("🔮 Future Weather Forecast")
        
        fc_tab1, fc_tab2 = st.tabs(["Temperature Forecast", "Precipitation Forecast"])
        
        with fc_tab1:
            # Temperature forecast
            forecast_temp_df = st.session_state.forecast_temp.rename(columns={
                "ds": "Date",
                "yhat": "Predicted Temperature (°C)",
                "yhat_lower": "Min Temp",
                "yhat_upper": "Max Temp"
            })
            last_date = pd.to_datetime(st.session_state.historical_data["time"].max())
            future_temp = forecast_temp_df[forecast_temp_df["Date"] > last_date]
            
            st.dataframe(future_temp[["Date", "Predicted Temperature (°C)"]], 
                        use_container_width=True)
            
            fig_temp = px.line(
                future_temp,
                x="Date",
                y="Predicted Temperature (°C)",
                title="30-Day Temperature Forecast"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with fc_tab2:
            # Precipitation forecast
            forecast_precip_df = st.session_state.forecast_precip.rename(columns={
                "ds": "Date",
                "yhat": "Predicted Precipitation (mm)",
                "yhat_lower": "Min Precip",
                "yhat_upper": "Max Precip"
            })
            future_precip = forecast_precip_df[forecast_precip_df["Date"] > last_date]
            
            st.dataframe(future_precip[["Date", "Predicted Precipitation (mm)"]], 
                        use_container_width=True)
            
            fig_precip = px.line(
                future_precip,
                x="Date",
                y="Predicted Precipitation (mm)",
                title="30-Day Precipitation Forecast"
            )
            st.plotly_chart(fig_precip, use_container_width=True)


    # Add chatbot interface
    st.subheader("💬 Weather Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about weather (e.g., 'What's the temperature?'):"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate context for NLP
        context = ""
        if (st.session_state.historical_data is not None 
            and not st.session_state.historical_data.empty):
            context = f"""
            Weather Statistics:
            - Average Temperature: {st.session_state.historical_data['temperature_2m_max'].mean():.1f}°C
            - Total Precipitation: {st.session_state.historical_data['precipitation_sum'].sum()}mm
            - Max Wind Speed: {st.session_state.historical_data['windspeed_10m_max'].max()}km/h
            """
        
        # Get bot response
        try:
            response = get_chat_response(prompt, context)
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}"
        
        # Add bot response to chat history
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# ================= Common Functions =================
def fetch_weather_data(latitude, longitude):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,windspeed_10m_max",
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
                      "forecast_temp": None,
                      "forecast_precip": None
                  }))
    
    # Modified prediction button
    if right_col.button("🔮 Predict Next 30 Days"):
        if st.session_state.historical_data is not None:
            try:
                # Temperature model
                df_temp = (st.session_state.historical_data
                          .rename(columns={"time": "ds", "temperature_2m_max": "y"})
                          [["ds", "y"]])
                model_temp = Prophet()
                model_temp.fit(df_temp)
                
                # Precipitation model
                df_precip = (st.session_state.historical_data
                            .rename(columns={"time": "ds", "precipitation_sum": "y"})
                            [["ds", "y"]])
                model_precip = Prophet()
                model_precip.fit(df_precip)
                
                # Generate future dates
                future = model_temp.make_future_dataframe(periods=30)
                
                # Make predictions
                st.session_state.forecast_temp = model_temp.predict(future)
                st.session_state.forecast_precip = model_precip.predict(future)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            st.warning("Please fetch historical data first!")