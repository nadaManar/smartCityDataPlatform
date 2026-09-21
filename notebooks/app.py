import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


st.set_page_config(page_title="PM2.5 Dashboard", layout="wide")
df = pd.read_csv("data/openweather_weather.csv")

# Charger le modèle Random Forest
rf = joblib.load("models/rf_model.pkl")



st.markdown("""
    <div style="background-color:#4CAF50;padding:18px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;margin:0;">🌆 PM2.5 Smart City Dashboard</h1>
        <p style="color:white;text-align:center;margin:0;">
            Prédiction de la pollution fine PM2.5 à partir des données météo + pollution
        </p>
    </div>
""", unsafe_allow_html=True)
col_left, col_right = st.columns([1, 1])
with col_left:
# --- SECTION : Prédiction ---
 st.subheader("🔮 Prédiction PM2.5")

 pm10 = st.slider("PM10", float(df.pm10.min()), float(df.pm10.max()), float(df.pm10.mean()))
 co = st.slider("CO", float(df.co.min()), float(df.co.max()), float(df.co.mean()))
 no2 = st.slider("NO2", float(df.no2.min()), float(df.no2.max()), float(df.no2.mean()))
 so2 = st.slider("SO2", float(df.so2.min()), float(df.so2.max()), float(df.so2.mean()))
 humidity = st.slider("Humidité", float(df.humidity.min()), float(df.humidity.max()), float(df.humidity.mean()))
 pressure = st.slider("Pression", float(df.pressure.min()), float(df.pressure.max()), float(df.pressure.mean()))


 if st.button("Prédire PM2.5"):
    X_input = [[pm10, co, no2, so2, humidity, pressure]]
    prediction = rf.predict(X_input)[0]
    st.metric("PM2.5 prédit", f"{prediction:.2f} µg/m³")

# --- SECTION : Importance des variables ---
with col_right:
 st.subheader("📊 Importance des variables")

 importances = rf.feature_importances_
 features = ['pm10', 'co', 'no2', 'so2', 'humidity', 'pressure']

 imp_df = pd.DataFrame({"feature": features, "importance": importances})
 fig_imp = px.bar(imp_df, x="feature", y="importance", title="Importance des variables")
 st.plotly_chart(fig_imp, use_container_width=True)

