import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import numpy as np

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return csv_files

def extract_features(signal):
    if len(signal) == 0:
        return [0] * 10
    return [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        skew(signal), kurtosis(signal), np.sum(signal**2),
        len(find_peaks(signal)[0]),
        np.sqrt(np.mean(signal**2)),
        np.corrcoef(signal[:-1], signal[1:])[0, 1] if len(signal) > 1 else 0
    ]

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

# Upload training data (ZIP with normal data only)
with st.sidebar:
    st.header("Upload Training Data")
    zip_file = st.file_uploader("Upload ZIP (Normal Data)", type=["zip"])
    if zip_file:
        with open("temp.zip", "wb") as f:
            f.write(zip_file.getbuffer())
        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
    
    # Column selection and thresholding
    if zip_file:
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        signal_column = st.selectbox("Select signal column", columns)
        
        if st.button("Train Model"):
            feature_data = []
            for file in csv_files:
                df = pd.read_csv(file)
                signal = df[signal_column].values
                feature_data.append(extract_features(signal))
            
            X_train = np.array(feature_data)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = IsolationForest(contamination=0, random_state=42)
            model.fit(X_train_scaled)
            
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.success("Model trained successfully!")

# Upload new data for prediction
st.sidebar.header("Upload New Data")
prediction_file = st.sidebar.file_uploader("Upload CSV for Prediction", type=["csv"])
if prediction_file and "model" in st.session_state:
    df_pred = pd.read_csv(prediction_file)
    pred_signal = df_pred[signal_column].values
    pred_features = np.array([extract_features(pred_signal)])
    pred_scaled = st.session_state["scaler"].transform(pred_features)
    prediction = st.session_state["model"].predict(pred_scaled)
    
    anomaly_status = "Anomalous" if prediction[0] == -1 else "Normal"
    color = "red" if anomaly_status == "Anomalous" else "blue"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pred_signal, mode='lines', line=dict(color=color), name=f"New Data ({anomaly_status})"))
    fig.update_layout(title="Prediction Visualization", xaxis_title="Index", yaxis_title="Signal Value")
    st.plotly_chart(fig)
