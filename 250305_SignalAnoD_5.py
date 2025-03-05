## ADVANCED FEATURE EXTRACTION

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

def extract_advanced_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 20

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    median_val = np.median(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val
    energy = np.sum(signal**2)
    cv = std_val / mean_val if mean_val != 0 else 0

    signal_fft = fft(signal)
    psd = np.abs(signal_fft)**2
    freqs = fftfreq(n, 1)
    positive_freqs = freqs[:n // 2]
    positive_psd = psd[:n // 2]
    dominant_freq = positive_freqs[np.argmax(positive_psd)] if len(positive_psd) > 0 else 0
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    peaks, _ = find_peaks(signal)
    peak_count = len(peaks)
    zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / n
    rms = np.sqrt(np.mean(signal**2))

    x = np.arange(n)
    slope, _ = np.polyfit(x, signal, 1)
    rolling_window = max(10, n // 10)
    rolling_mean = np.convolve(signal, np.ones(rolling_window) / rolling_window, mode='valid')
    moving_average = np.mean(rolling_mean)

    threshold = 3 * std_val
    outlier_count = np.sum(np.abs(signal - mean_val) > threshold)
    extreme_event_duration = 0
    current_duration = 0
    for value in signal:
        if np.abs(value - mean_val) > threshold:
            current_duration += 1
        else:
            extreme_event_duration = max(extreme_event_duration, current_duration)
            current_duration = 0

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv, 
            dominant_freq, spectral_entropy, autocorrelation, peak_count, zero_crossing_rate, rms, 
            slope, moving_average, outlier_count, extreme_event_duration]

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

with st.sidebar:
    # Step 1: Upload training data ZIP
    st.header("1. Upload Training Data")
    training_file = st.file_uploader("Upload a ZIP file containing CSV files (Normal Data)", type=["zip"], key="training")
    if training_file:
        with open("training.zip", "wb") as f:
            f.write(training_file.getbuffer())
        csv_files, extract_dir = extract_zip("training.zip")
        st.success(f"Extracted {len(csv_files)} training CSV files")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering (Training Data)", columns)
        threshold = st.number_input("Enter filtering threshold (Training Data)", value=0.0)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                bead_segments = {}
                metadata = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    if segments:
                        bead_segments[file] = segments
                        for bead_num, (start, end) in enumerate(segments, start=1):
                            metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
                
                # Extract features from training data
                training_features = []
                for entry in metadata:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    signal = bead_segment.iloc[:, 0].values
                    training_features.append(extract_advanced_features(signal))
                
                feature_matrix = np.array(training_features)
                scaler = RobustScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)

                # Train Isolation Forest with 0 contamination
                isolation_forest = IsolationForest(contamination=0, random_state=42)
                isolation_forest.fit(feature_matrix)
                
                # Store model and metadata in session state
                st.session_state["isolation_forest"] = isolation_forest
                st.session_state["scaler"] = scaler
                st.session_state["metadata"] = metadata
                st.session_state["filter_column"] = filter_column
                st.success("Model trained successfully!")

    # Step 2: Upload new data for prediction
    st.header("2. Upload New Data for Prediction")
    new_data_file = st.file_uploader("Upload a ZIP file containing new CSV files", type=["zip"], key="new_data")
    if new_data_file and "isolation_forest" in st.session_state:
        with open("new_data.zip", "wb") as f:
            f.write(new_data_file.getbuffer())
        csv_files, extract_dir = extract_zip("new_data.zip")
        st.success(f"Extracted {len(csv_files)} new data CSV files")

        bead_numbers = st.text_input("Enter bead numbers (comma-separated) for prediction")
        if st.button("Predict Anomalies"):
            with st.spinner("Predicting anomalies..."):
                selected_beads = [int(b.strip()) for b in bead_numbers.split(",") if b.strip().isdigit()]
                anomaly_results = {}
                anomaly_scores = {}
                
                for file in csv_files:
                    df = pd.read_csv(file)
                    for bead_number in selected_beads:
                        bead_data = [
                            entry for entry in st.session_state["metadata"]
                            if entry["file"] == file and entry["bead_number"] == bead_number
                        ]
                        if bead_data:
                            bead_data = bead_data[0]
                            bead_segment = df.iloc[bead_data["start_index"]:bead_data["end_index"] + 1]
                            signal = bead_segment.iloc[:, 0].values
                            features = extract_advanced_features(signal)
                            scaled_features = st.session_state["scaler"].transform([features])
                            
                            prediction = st.session_state["isolation_forest"].predict(scaled_features)
                            score = -st.session_state["isolation_forest"].decision_function(scaled_features)
                            status = "anomalous" if prediction[0] == -1 else "normal"
                            anomaly_results[file] = status
                            anomaly_scores[file] = score[0]
                
                # Plot results
                st.write("## Visualization")
                for file, status in anomaly_results.items():
                    df = pd.read_csv(file)
                    signal = df[st.session_state["filter_column"]].values
                    color = "red" if status == "anomalous" else "black"
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=signal,
                        mode="lines",
                        line=dict(color=color, width=1),
                        name=f"{file}",
                        hoverinfo="text",
                        text=f"File: {file}<br>Status: {status}<br>Anomaly Score: {anomaly_scores[file]:.4f}"
                    ))
                    fig.update_layout(
                        title=f"File: {file} - {status.capitalize()}",
                        xaxis_title="Time Index",
                        yaxis_title="Signal Value",
                        showlegend=True
                    )
                    st.plotly_chart(fig)
