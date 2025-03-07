## ADVANCED FEATURE EXTRACTION
import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
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
        return [0] * 20  # Default feature values

    # Handle NaN or Inf values
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return [0] * 20  # Return default values if data is bad

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

    # FFT calculations
    signal_fft = fft(signal)
    psd = np.abs(signal_fft)**2
    freqs = fftfreq(n, 1)
    positive_freqs = freqs[:n // 2]
    positive_psd = psd[:n // 2]
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    rms = np.sqrt(np.mean(signal**2))

    # Handle edge cases for np.polyfit
    x = np.arange(n)
    if len(set(signal)) == 1 or len(signal) < 2:  # Constant or too short signal
        slope = 0
    else:
        try:
            slope, _ = np.polyfit(x, signal, 1)
        except np.linalg.LinAlgError:
            slope = 0

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
            spectral_entropy, autocorrelation, rms, 
            slope, moving_average, outlier_count, extreme_event_duration]

def normalize_signal_per_bead(bead_data):
    """
    Normalize the raw signal data for each bead segment.
    This ensures the scale for each bead is consistent while preserving the pattern.
    """
    normalized_data = []
    for bead in bead_data:
        raw_signal = bead["data"].iloc[:, 0].values  # Keep a copy of the raw signal
        scaler = MinMaxScaler()  # You can change to StandardScaler() for z-score normalization
        if len(raw_signal) > 1:  # Only normalize if there is more than one data point
            normalized_signal = scaler.fit_transform(raw_signal.reshape(-1, 1)).flatten()
        else:
            normalized_signal = raw_signal  # If only one data point, no normalization

        # Add normalized signal as a new column in the DataFrame (preserve the raw signal)
        bead["data"]["normalized_signal"] = normalized_signal
        bead["raw_signal"] = raw_signal  # Store the raw signal separately
        normalized_data.append(bead)
    return normalized_data

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(feature_matrix, epochs=50, batch_size=32):
    autoencoder = build_autoencoder(feature_matrix.shape[1])
    autoencoder.fit(feature_matrix, feature_matrix, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    return autoencoder

# Streamlit App Setup
st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection with Autoencoder")

with st.sidebar:
    # Assign a unique key to the file_uploader element
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"], key="zip_uploader")
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
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] in selected_beads:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    chosen_bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"], "start_index": entry["start_index"], "end_index": entry["end_index"]})
            chosen_bead_data = normalize_signal_per_bead(chosen_bead_data)
            st.session_state["chosen_bead_data"] = chosen_bead_data
            st.success("Beads selected and normalized successfully!")

if st.button("Run Autoencoder") and "chosen_bead_data" in st.session_state:
    with st.spinner("Running Autoencoder Anomaly Detection..."):
        bead_data = [seg["data"]["normalized_signal"].values for seg in st.session_state["chosen_bead_data"]]
        
        # Extract features for each bead
        feature_matrix = np.array([extract_advanced_features(signal) for signal in bead_data])
        scaler = MinMaxScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

        # Train Autoencoder and calculate reconstruction error
        autoencoder = train_autoencoder(feature_matrix, epochs=50, batch_size=32)
        reconstruction_error = np.mean(np.square(feature_matrix - autoencoder.predict(feature_matrix)), axis=1)
        
        # Set anomaly threshold (95th percentile of reconstruction error)
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold

        # Save anomaly results
        st.session_state["anomaly_results"] = [{"Bead Number": seg["bead_number"], 
                                                "File": seg["file"], 
                                                "Status": "Anomalous" if anomaly else "Normal", 
                                                "Reconstruction Error": error} 
                                               for seg, anomaly, error in zip(st.session_state["chosen_bead_data"], anomalies, reconstruction_error)]
        st.success("Anomaly detection complete!")

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
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] in selected_beads:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    chosen_bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"], "start_index": entry["start_index"], "end_index": entry["end_index"]})
            chosen_bead_data = normalize_signal_per_bead(chosen_bead_data)
            st.session_state["chosen_bead_data"] = chosen_bead_data
            st.success("Beads selected and normalized successfully!")

if st.button("Run Autoencoder") and "chosen_bead_data" in st.session_state:
    with st.spinner("Running Autoencoder Anomaly Detection..."):
        bead_data = [seg["data"]["normalized_signal"].values for seg in st.session_state["chosen_bead_data"]]
        
        # Extract features for each bead
        feature_matrix = np.array([extract_advanced_features(signal) for signal in bead_data])
        scaler = MinMaxScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

        # Train Autoencoder and calculate reconstruction error
        autoencoder = train_autoencoder(feature_matrix, epochs=50, batch_size=32)
        reconstruction_error = np.mean(np.square(feature_matrix - autoencoder.predict(feature_matrix)), axis=1)
        
        # Set anomaly threshold (95th percentile of reconstruction error)
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold

        # Save anomaly results
        st.session_state["anomaly_results"] = [{"Bead Number": seg["bead_number"], 
                                                "File": seg["file"], 
                                                "Status": "Anomalous" if anomaly else "Normal", 
                                                "Reconstruction Error": error} 
                                               for seg, anomaly, error in zip(st.session_state["chosen_bead_data"], anomalies, reconstruction_error)]
        st.success("Anomaly detection complete!")
