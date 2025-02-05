# 🌦️ Smart Weather Analysis Dashboard

**Demo**: https://drive.google.com/file/d/1WSZyRoDpPAiFGICWjPh8ssmkM_bOyicr/view?usp=sharing

The **Smart Weather Analysis Dashboard** is a comprehensive web application designed to provide historical weather data analysis, future weather predictions, and an interactive chatbot for answering weather-related queries. Built with **Streamlit**, this tool leverages **weather APIs, machine learning models, and natural language processing** to deliver actionable insights and forecasts.

---

## 🌟 About
This project is a one-stop solution for weather data analysis and forecasting. It allows users to:

- **Visualize historical weather data** for any location.
- **Predict future weather conditions** using machine learning.
- **Interact with a chatbot** to get answers to weather-related questions based on the analyzed data.

The application is designed for **meteorologists, researchers, and anyone interested in understanding weather patterns and trends.**

---

## 🚀 Key Features

### 1. Interactive Map for Location Selection
- Users can **draw a rectangle on a world map** to select a specific area for weather analysis.
- Automatically calculates the **centroid** of the selected area for accurate data retrieval.

### 2. Historical Weather Data Analysis
- Fetches and displays **historical weather data** (temperature, precipitation, wind speed, etc.) for the selected location and date range.
- Visualizes data using **interactive charts** (line charts, bar charts, etc.).

### 3. Weather Forecasting
- Predicts **future weather conditions** (temperature and precipitation) for the next **30 days** using the **Prophet forecasting model**.
- Displays forecasted data in **tables and interactive charts**.

### 4. AI-Powered Chatbot
- Answers user questions about **historical and forecasted weather data**.
- Uses **Google's Gemini Pro API** for natural language understanding and response generation.
- Provides insights such as **average temperatures, highest precipitation days, and forecast comparisons**.

### 5. User-Friendly Interface
- Built with **Streamlit** for a **clean and intuitive** user experience.
- **Responsive design** for seamless use on both desktop and mobile devices.

---

## 🛠️ Technologies

### **Backend**
- **Python**: Core programming language.
- **Streamlit**: For building the web application interface.
- **Prophet**: For time-series forecasting of weather data.
- **Google Gemini Pro API**: For powering the chatbot's natural language capabilities.

### **Frontend**
- **Plotly**: For creating interactive visualizations (charts and graphs).
- **Folium**: For rendering interactive maps.
- **Streamlit Components**: For integrating custom UI elements.

### **APIs**
- **Open-Meteo API**: For fetching historical weather data.
- **Nominatim (Geopy)**: For reverse geocoding to get location details.

### **Libraries**
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Plotly Express**: For creating visualizations.
- **Geopy**: For geocoding and location-based services.

---

## 🏗️ Architecture

The application follows a **modular architecture**:

### **User Interface (UI):**
- Built with **Streamlit** for seamless interaction.
- Includes a **map for location selection, date range picker, and chatbot interface**.

### **Data Layer:**
- Fetches **historical weather data** from the **Open-Meteo API**.
- Stores and processes data using **Pandas**.

### **Machine Learning Layer:**
- Uses **Facebook's Prophet model** for **time-series forecasting**.
- Generates **predictions for temperature and precipitation**.

### **Chatbot Layer:**
- Integrates **Google's Gemini Pro API** for **natural language processing**.
- Answers **user queries** based on **historical and forecasted data**.

### **Visualization Layer:**
- Uses **Plotly and Folium** for **interactive charts and maps**.

---

## 🖼️ Screenshots

![image](https://github.com/user-attachments/assets/016915b1-7c35-4f0d-8fd0-670b8436cafe)

![image](https://github.com/user-attachments/assets/b03ae4ad-15fc-4242-aaec-4ffd2ff029a2)

![image](https://github.com/user-attachments/assets/d1bf592c-806f-4e02-ae7e-ad098d2084e2)

![image](https://github.com/user-attachments/assets/fc629507-db59-4db5-979e-1e538091e53a)



