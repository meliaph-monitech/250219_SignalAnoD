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
        
        bead_numbers = st.text_input("Enter bead numbers (comma-separated)")
        if st.button("Select Beads") and "metadata" in st.session_state:
            selected_beads = [int(b.strip()) for b in bead_numbers.split(",") if b.strip().isdigit()]
            chosen_bead_data = []
            raw_signals_by_bead = {}
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] in selected_beads:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    chosen_bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"], "start_index": entry["start_index"], "end_index": entry["end_index"]})
                    raw_signals_by_bead.setdefault(entry["bead_number"], []).append(bead_segment.iloc[:, 0].values)
            
            # Scale raw signals by bead
            scaled_signals_by_bead = {}
            scaler = RobustScaler()
            for bead_number, signals in raw_signals_by_bead.items():
                flattened_signals = np.concatenate(signals).reshape(-1, 1)
                scaled_flattened = scaler.fit_transform(flattened_signals).flatten()
                scaled_signals_by_bead[bead_number] = np.split(scaled_flattened, np.cumsum([len(sig) for sig in signals[:-1]]))
            
            # Replace raw signals with scaled signals in chosen_bead_data
            for entry in chosen_bead_data:
                bead_number = entry["bead_number"]
                entry["data"].iloc[:, 0] = scaled_signals_by_bead[bead_number].pop(0)
            
            st.session_state["chosen_bead_data"] = chosen_bead_data
            st.success("Beads selected and scaled successfully!")
        
        # Contamination rate logic in the sidebar section
        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        use_contamination_rate = st.checkbox("Use Contamination Rate", value=True)
        
        if st.button("Run Isolation Forest") and "chosen_bead_data" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                anomaly_results_isoforest = {}
                anomaly_scores_isoforest = {}
                for bead_number in sorted(set(seg["bead_number"] for seg in st.session_state["chosen_bead_data"])): 
                    bead_data = [seg for seg in st.session_state["chosen_bead_data"] if seg["bead_number"] == bead_number]
                    signals = [seg["data"].iloc[:, 0].values for seg in bead_data]
                    file_names = [seg["file"] for seg in bead_data]
                    feature_matrix = np.array([extract_advanced_features(signal) for signal in signals])
        
                    # Conditional logic for IsolationForest
                    if use_contamination_rate:
                        iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                    else:
                        iso_forest = IsolationForest(random_state=42)
        
                    predictions = iso_forest.fit_predict(feature_matrix)
                    anomaly_scores = -iso_forest.decision_function(feature_matrix)
                    bead_results = {}
                    bead_scores = {}
                    for idx, prediction in enumerate(predictions):
                        status = 'anomalous' if prediction == -1 else 'normal'
                        bead_results[file_names[idx]] = status
                        bead_scores[file_names[idx]] = anomaly_scores[idx]
                    anomaly_results_isoforest[bead_number] = bead_results
                    anomaly_scores_isoforest[bead_number] = bead_scores

# Visualization remains the same
