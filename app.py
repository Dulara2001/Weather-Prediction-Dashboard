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
import google.generativeai as genai

# Initialize session state
if "historical_data" not in st.session_state:
    st.session_state.historical_data = None
if "forecast_temp" not in st.session_state:
    st.session_state.forecast_temp = None
if "forecast_precip" not in st.session_state:
    st.session_state.forecast_precip = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# ================= Chatbot Section =================
st.sidebar.title("💬 Weather Chatbot")

# Initialize Gemini Pro API
genai.configure(api_key="Your_Api_key")

def get_data_summary():
    """Generate a text summary of available data"""
    summary = []
    
    if st.session_state.historical_data is not None:
        hist_df = st.session_state.historical_data
        summary.append("Historical Weather Data:")
        summary.append(f"- Date Range: {hist_df['time'].min()} to {hist_df['time'].max()}")
        summary.append(f"- Avg Max Temp: {hist_df['temperature_2m_max'].mean():.1f}°C")
        summary.append(f"- Avg Min Temp: {hist_df['temperature_2m_min'].mean():.1f}°C")
        summary.append(f"- Total Precipitation: {hist_df['precipitation_sum'].sum()}mm")
    
    if st.session_state.forecast_temp is not None:
        forecast_df = st.session_state.forecast_temp
        summary.append("\nTemperature Forecast:")
        summary.append(f"- Predicted Avg Temp: {forecast_df['yhat'].mean():.1f}°C")
        summary.append(f"- Max Predicted Temp: {forecast_df['yhat'].max():.1f}°C")
    
    if st.session_state.forecast_precip is not None:
        precip_df = st.session_state.forecast_precip
        summary.append("\nPrecipitation Forecast:")
        summary.append(f"- Total Predicted Precipitation: {precip_df['yhat'].sum()}mm")
    
    return "\n".join(summary) if summary else "No data available"

def get_chatbot_response(user_input):
    data_summary = get_data_summary()
    
    prompt = f"""
    You are a weather data analyst. Use the following data to answer questions.
    Only use the provided data - don't make up answers.
    If asked about data that isn't available, say you don't have that information.
    
    Current Data Summary:
    {data_summary}
    
    User Question: {user_input}
    
    Answer in a clear, concise way using numbers from the data when possible.
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        print("API Response:", response.text)  # Debugging
        return response.text
    except Exception as e:
        print("API Error:", str(e))  # Debugging
        return f"Error: {str(e)}"

# Chatbot UI
user_input = st.sidebar.text_input("Ask about the weather data (e.g., 'What was the hottest day?')")

if user_input:
    print("User input received:", user_input)  # Debugging
    if st.session_state.historical_data is None and "forecast" not in user_input.lower():
        st.sidebar.warning("Please load historical data first!")
    else:
        with st.spinner("Analyzing data..."):
            try:
                print("Generating response...")  # Debugging
                chatbot_response = get_chatbot_response(user_input)
                print("Response generated:", chatbot_response)  # Debugging
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": chatbot_response})
                print("Chat history updated")  # Debugging
            except Exception as e:
                print("Error:", str(e))  # Debugging
                st.sidebar.error(f"Error generating response: {str(e)}")

# Display chat history
print("Current Chat History:", st.session_state.chat_history)  # Debugging
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.sidebar.text_area("You", value=message["content"], height=75, key=f"user_{message['content']}")
    else:
        st.sidebar.text_area("Chatbot", value=message["content"], height=100, key=f"assistant_{message['content']}")

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