import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="PEMS-BAY Traffic Forecast",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Settings (Must match your training)
# -----------------------------
N_PAST = 24
N_FUTURE = 3
N_SENSORS = 325

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model(r"L:\Softwares\VS Code\Files\RNN\PEMS-BAY\pems_bay_gru_forecast.h5", compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    with open(r"L:\Softwares\VS Code\Files\RNN\PEMS-BAY\pems_bay_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

model, scaler = load_resources()

# -----------------------------
# Load full dataset (for selecting real windows)
# -----------------------------
@st.cache_data
def load_full_data():
    df = pd.read_csv(r"L:\4th sem\Mission Scopus\ML Datasets\PEMS-BAY.csv")
    
    # Drop timestamp column if exists
    if "2017" in str(df.iloc[0, 0]) or df.columns[0].lower() in ['timestamp', 'date', 'time']:
        df = df.drop(df.columns[0], axis=1)
    
    data = df.values.astype(np.float32)
    return data

try:
    full_data = load_full_data()
    has_real_data = True
except:
    full_data = None
    has_real_data = False

# -----------------------------
# Helper functions
# -----------------------------
def predict_future(past_data):
    scaled = scaler.transform(past_data)
    model_input = scaled.reshape(1, N_PAST, N_SENSORS)
    pred_scaled = model.predict(model_input, verbose=0)[0]
    pred_scaled = pred_scaled.reshape(N_FUTURE, N_SENSORS)
    return scaler.inverse_transform(pred_scaled)

def generate_random_data():
    np.random.seed(None)  # different every time
    base = np.random.uniform(40, 70, size=N_SENSORS)
    data = []
    for t in range(N_PAST):
        noise = np.random.normal(0, 3, size=N_SENSORS)
        speed = np.clip(base + noise - t*0.2, 8, 85)
        data.append(speed)
    return np.array(data)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🚦 Settings")

st.sidebar.markdown(f"**N_PAST:** {N_PAST} | **N_FUTURE:** {N_FUTURE}")
st.sidebar.markdown(f"**Sensors:** {N_SENSORS}")

selected_sensors = st.sidebar.multiselect(
    "Select sensors to display",
    options=list(range(N_SENSORS)),
    default=[0, 50, 100, 150, 200, 250],
    max_selections=8
)

st.sidebar.markdown("---")
input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Select from Real Data", "Generate Random Data", "Upload CSV"]
)

# -----------------------------
# Main Content
# -----------------------------
st.title("🚦 PEMS-BAY Traffic Speed Forecasting")

past_data = None

# ---------- Option 1: Select real window from dataset ----------
if input_method == "Select from Real Data":
    if not has_real_data:
        st.error("PEMS-BAY.csv not found. Please put it in the same folder.")
    else:
        max_start = len(full_data) - N_PAST - N_FUTURE
        
        st.subheader("Select a time window from the real dataset")
        
        start_idx = st.slider(
            "Starting row index",
            min_value=0,
            max_value=max_start,
            value=1000,
            step=10,
            help=f"Dataset has {len(full_data)} rows. Choose where the past window should start."
        )
        
        past_data = full_data[start_idx : start_idx + N_PAST]
        
        st.success(f"Selected rows **{start_idx}** to **{start_idx + N_PAST - 1}**")

# ---------- Option 2: Random Data ----------
elif input_method == "Generate Random Data":
    if "random_data" not in st.session_state:
        st.session_state.random_data = generate_random_data()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Generate New Random Data"):
            st.session_state.random_data = generate_random_data()
    
    past_data = st.session_state.random_data
    st.success("Random data generated successfully!")

# Upload CSV file
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        f"Upload CSV with at least {N_PAST} rows and {N_SENSORS} columns (sensors)",
        type=["csv"]
    )
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        if df_upload.shape[1] > N_SENSORS:
            df_upload = df_upload.iloc[:, :N_SENSORS]
        if len(df_upload) < N_PAST:
            st.error(f"Need at least {N_PAST} rows of past data")
        else:
            past_data = df_upload.values[-N_PAST:].astype(np.float32)
            st.success(f"✅ Loaded data with shape {past_data.shape}")   

# -----------------------------
# Show data + Predict
# -----------------------------
if past_data is not None and len(selected_sensors) > 0:
    st.markdown("---")
    st.subheader("Past Speeds (Input)")
    
    past_df = pd.DataFrame(
        past_data[:, selected_sensors],
        columns=[f"Sensor {s}" for s in selected_sensors]
    )
    past_df.index = [f"t-{N_PAST-i}" for i in range(N_PAST)]
    st.dataframe(past_df.style.format("{:.1f}"), use_container_width=True)
    
    if st.button("🔮 Predict Future Speeds", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            predictions = predict_future(past_data)
        
        st.markdown("---")
        st.subheader("Predicted Future Speeds (mph)")
        
        pred_df = pd.DataFrame(
            predictions[:, selected_sensors],
            columns=[f"Sensor {s}" for s in selected_sensors]
        )
        pred_df.index = [f"t+{i+1}" for i in range(N_FUTURE)]
        st.dataframe(pred_df.style.format("{:.1f}").background_gradient(cmap="YlOrRd"), use_container_width=True)
        
        # Chart
        fig = go.Figure()
        for s in selected_sensors:
            fig.add_trace(go.Scatter(
                x=list(range(-N_PAST, 0)),
                y=past_data[:, s],
                mode='lines+markers',
                name=f'Sensor {s} Past'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(0, N_FUTURE)),
                y=predictions[:, s],
                mode='lines+markers',
                name=f'Sensor {s} Predicted',
                line=dict(dash='dash')
            ))
        
        fig.update_layout(
            title="Past + Future Forecast",
            xaxis_title="Relative Timestep",
            yaxis_title="Speed (mph)",
            height=550,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.markdown("### 📋 Prediction Summary")
        summary = pd.DataFrame({
            "Sensor": selected_sensors,
            "Last Known Speed": past_data[-1, selected_sensors],
            "Predicted t+1": predictions[0, selected_sensors],
            "Predicted t+3": predictions[min(2, N_FUTURE-1), selected_sensors],
            "Change (t+1)": predictions[0, selected_sensors] - past_data[-1, selected_sensors]
            })
        st.dataframe(summary.style.format({
            "Last Known Speed": "{:.1f}",
            "Predicted t+1": "{:.1f}",
            "Predicted t+3": "{:.1f}",
            "Change (t+1)": "{:+.1f}"
            }), use_container_width=True)

else:
    st.warning("Please select at least one sensor.")

# Footer
st.markdown("---")
st.caption("PEMS-BAY Multivariate Traffic Forecasting | GRU Model")    