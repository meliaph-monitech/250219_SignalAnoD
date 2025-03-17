import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import numpy as np
from collections import defaultdict


def extract_zip(zip_path, extract_dir="extracted_csvs"):
    """Extracts a ZIP file containing CSV files."""
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        st.stop()

    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    if not csv_files:
        st.error("No CSV files found in the ZIP file.")
        st.stop()

    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir


def segment_beads(df, column, threshold):
    """Segments data into beads based on a threshold."""
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
    """Extracts advanced statistical and signal processing features from a signal."""
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
    positive_psd = psd[:n // 2]
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    rms = np.sqrt(np.mean(signal**2))

    x = np.arange(n)
    slope, _ = np.polyfit(x, signal, 1) if len(signal) > 1 else (0, 0)

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv,
            spectral_entropy, rms, slope]


def normalize_signal_per_bead(bead_data):
    """Normalize the signal for each bead using MinMaxScaler."""
    normalized_data = []
    for bead in bead_data:
        raw_signal = bead["data"].iloc[:, 0].values
        scaler = MinMaxScaler()
        if len(raw_signal) > 1:
            normalized_signal = scaler.fit_transform(raw_signal.reshape(-1, 1)).flatten()
        else:
            normalized_signal = raw_signal
        bead["data"]["normalized_signal"] = normalized_signal
        bead["raw_signal"] = raw_signal
        normalized_data.append(bead)
    return normalized_data


st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection V9 - Visualization for Raw and Normalized Data")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files, extract_dir = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        
        if st.button("Segment Beads"):
            with st.spinner("Segmenting beads..."):
                bead_segments = {}
                metadata = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    if segments:
                        bead_segments[file] = segments
                        for bead_num, (start, end) in enumerate(segments, start=1):
                            metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
                st.success("Bead segmentation complete")
                st.session_state["metadata"] = metadata
        
        if st.button("Normalize Beads") and "metadata" in st.session_state:
            chosen_bead_data = []
            for entry in st.session_state["metadata"]:
                df = pd.read_csv(entry["file"])
                bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                chosen_bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"]})
            chosen_bead_data = normalize_signal_per_bead(chosen_bead_data)
            st.session_state["chosen_bead_data"] = chosen_bead_data
            st.success("Beads normalized successfully!")
        
        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1)

        if st.button("Run Isolation Forest") and "chosen_bead_data" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                bead_data = st.session_state["chosen_bead_data"]
                normalized_signals = [bead["data"]["normalized_signal"].values for bead in bead_data]
                features = np.array([extract_advanced_features(signal) for signal in normalized_signals])
                
                scaler = RobustScaler()
                scaled_features = scaler.fit_transform(features)
                
                iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                predictions = iso_forest.fit_predict(scaled_features)
                anomaly_scores = -iso_forest.decision_function(scaled_features)

                st.session_state["anomaly_results"] = {
                    bead["file"]: ("anomalous" if pred == -1 else "normal") for bead, pred in zip(bead_data, predictions)
                }
                st.session_state["anomaly_scores"] = {
                    bead["file"]: score for bead, score in zip(bead_data, anomaly_scores)
                }
                st.success("Isolation Forest analysis complete!")

if "anomaly_results" in st.session_state:
    st.write("## Visualization")
    visualization_type = st.sidebar.radio(
        "Select Visualization Type",
        ("Raw Signal", "Normalized Signal")
    )
    
    bead_data = st.session_state["chosen_bead_data"]
    for bead in bead_data:
        file_name = bead["file"]
        status = st.session_state["anomaly_results"][file_name]
        score = st.session_state["anomaly_scores"][file_name]
        signal = (
            bead["raw_signal"]
            if visualization_type == "Raw Signal"
            else bead["data"]["normalized_signal"]
        )
        color = 'red' if status == "anomalous" else 'black'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=signal,
            mode='lines',
            line=dict(color=color, width=1),
            name=f"{file_name} ({status})",
            hoverinfo='text',
            text=f"File: {file_name}<br>Status: {status}<br>Anomaly Score: {score:.4f}"
        ))
        fig.update_layout(
            title=f"File: {file_name} - Anomaly Detection ({visualization_type} Data)",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
