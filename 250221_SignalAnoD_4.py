import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import numpy as np

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir


def segment_beads(df, column, threshold):
    start_indices = []
    end_indices = []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

def extract_time_freq_features(signal):
    n = len(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    energy = np.sum(np.square(signal)) / n
    skewness = skew(signal)
    kurt = kurtosis(signal)
    fft_values = fft(signal)
    fft_magnitude = np.abs(fft_values)[:n // 2]
    spectral_energy = np.sum(fft_magnitude ** 2) / len(fft_magnitude)
    dominant_freq = np.argmax(fft_magnitude)
    return [mean_val, std_val, min_val, max_val, energy, skewness, kurt, spectral_energy, dominant_freq]

st.set_page_config(layout="wide")

st.title("Laser Welding Anomaly Detection")

# Upload training (normal) data
with st.sidebar:
    uploaded_training_file = st.file_uploader("Upload a ZIP file containing CSV files (Training Data - Normal)", type=["zip"], key="train")
    if uploaded_training_file:
        with open("temp_train.zip", "wb") as f:
            f.write(uploaded_training_file.getbuffer())
        csv_train_files, extract_train_dir = extract_zip("temp_train.zip")
        st.success(f"Extracted {len(csv_train_files)} training CSV files")
        
        df_sample = pd.read_csv(csv_train_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns, key="column")
        threshold = st.number_input("Enter filtering threshold", value=0.0, key="threshold")
        
        if st.button("Train Model"):
            with st.spinner("Training on normal data..."):
                normal_bead_data = {}
                for file in csv_train_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        signal = df.iloc[start:end + 1][filter_column].values
                        if bead_num not in normal_bead_data:
                            normal_bead_data[bead_num] = []
                        normal_bead_data[bead_num].append(extract_time_freq_features(signal))
                
                # Train Isolation Forest on normal data
                trained_models = {}
                for bead_num, features in normal_bead_data.items():
                    feature_matrix = np.array(features)
                    iso_forest = IsolationForest(contamination=0.0, random_state=42)
                    iso_forest.fit(feature_matrix)
                    trained_models[bead_num] = iso_forest
                
                st.session_state["trained_models"] = trained_models
                st.success("Model trained on normal data!")

# Upload new data for anomaly detection
uploaded_new_file = st.file_uploader("Upload a ZIP file containing CSV files (New Data for Anomaly Detection)", type=["zip"], key="new")
if uploaded_new_file and "trained_models" in st.session_state:
    with open("temp_new.zip", "wb") as f:
        f.write(uploaded_new_file.getbuffer())
    csv_new_files, extract_new_dir = extract_zip("temp_new.zip")
    st.success(f"Extracted {len(csv_new_files)} new CSV files")
    
    if st.button("Detect Anomalies"):
        with st.spinner("Detecting anomalies..."):
            anomaly_results = {}
            for file in csv_new_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end + 1][filter_column].values
                    if bead_num in st.session_state["trained_models"]:
                        features = np.array([extract_time_freq_features(signal)])
                        model = st.session_state["trained_models"][bead_num]
                        prediction = model.predict(features)[0]
                        status = 'anomalous' if prediction == -1 else 'normal'
                        anomaly_results.setdefault(bead_num, []).append((file, status))
            
            st.session_state["anomaly_results"] = anomaly_results
            st.success("Anomaly detection complete!")

# Visualization
if "anomaly_results" in st.session_state:
    for bead_num, results in st.session_state["anomaly_results"].items():
        fig = go.Figure()
        for file, status in results:
            df = pd.read_csv(file)
            signal = df[filter_column].values
            color = 'red' if status == 'anomalous' else 'black'
            fig.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color=color, width=1), name=f"{file} ({status})"))
        fig.update_layout(title=f"Bead Number {bead_num}: Anomaly Detection Results", xaxis_title="Time Index", yaxis_title="Signal Value", showlegend=True)
        st.plotly_chart(fig)
    st.success("Visualization complete!")
