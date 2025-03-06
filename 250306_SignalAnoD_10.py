import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
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


def normalize_signals_per_bead(bead_segments, csv_files, column):
    """Normalizes raw signal values per bead number across all CSV files."""
    bead_signals = defaultdict(list)
    for file in csv_files:
        df = pd.read_csv(file)
        for bead_num, (start, end) in enumerate(bead_segments.get(file, []), start=1):
            signal = df.iloc[start:end + 1][column].values
            bead_signals[bead_num].append(signal)
    
    # Normalize per bead number across all files
    normalized_signals = defaultdict(dict)
    for bead_num, signals in bead_signals.items():
        scaler = RobustScaler()
        stacked_signals = np.concatenate(signals).reshape(-1, 1)
        scaler.fit(stacked_signals)
        
        for file, (start, end) in bead_segments.get(file, []):
            df = pd.read_csv(file)
            df.loc[start:end, column] = scaler.transform(df.loc[start:end, column].values.reshape(-1, 1)).flatten()
            normalized_signals[file][bead_num] = df
    
    return normalized_signals


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
    positive_freqs = freqs[:n // 2]
    positive_psd = psd[:n // 2]
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    rms = np.sqrt(np.mean(signal**2))

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv, spectral_entropy, autocorrelation, rms]


st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection V9 - Feature Selection and Normalization")

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

        # Feature selection
        feature_names = ["Mean Value", "STD Value", "Min Value", "Max Value", "Median Value", "Skewness", "Kurtosis", "Peak-to-Peak", "Energy", "Coefficient of Variation (CV)",
                         "Spectral Entropy", "Autocorrelation", "Root Mean Square (RMS)"]
        # Add "All" as the first option
        options = ["All"] + feature_names

        # Feature selection in the sidebar
        selected_features = st.multiselect(
            "Select features to use for Isolation Forest",
            options=options,  # Includes "All" and individual features
            default="All"  # Default selection is "All"
        )

        # Enforce logical behavior: If "All" is selected, deselect all other features
        if "All" in selected_features and len(selected_features) > 1:
            selected_features = ["All"]

        # If individual features are selected, deselect "All"
        elif "All" not in selected_features and len(selected_features) == 0:
            st.error("You must select at least one feature.")
            st.stop()

        # If "All" is selected, use all features
        if "All" in selected_features:
            selected_features = feature_names  # Automatically use all features
        
        # Map selected feature names to their corresponding indices
        selected_indices = [feature_names.index(f) for f in selected_features]

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
                st.session_state["normalized_data"] = normalize_signals_per_bead(bead_segments, csv_files, filter_column)
        
        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

        if st.button("Run Isolation Forest") and "metadata" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                features = []
                filenames = []
                for entry in st.session_state["metadata"]:
                    file, bead_number = entry["file"], entry["bead_number"]
                    df = st.session_state["normalized_data"][file][bead_number]
                    signal = df.iloc[entry["start_index"]:entry["end_index"] + 1, 0].values
                    all_features = extract_advanced_features(signal)
                    features.append([all_features[i] for i in selected_indices])
                    filenames.append((file, bead_number))
                
                features = np.array(features)
                iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                predictions = iso_forest.fit_predict(features)
                anomaly_scores = -iso_forest.decision_function(features)

                # Save results
                st.session_state["anomaly_results"] = {fn: ('anomalous' if p == -1 else 'normal') for fn, p in zip(filenames, predictions)}
                st.session_state["anomaly_scores"] = {fn: s for fn, s in zip(filenames, anomaly_scores)}

st.write("## Visualization")
if "anomaly_results" in st.session_state:
    bead_numbers = sorted(set(num for _, num in st.session_state["anomaly_results"].keys()))
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)

    if selected_bead:
        fig = go.Figure()

        # Filter data for the selected bead number
        selected_bead_data = [entry for entry in st.session_state["metadata"] if entry["bead_number"] == selected_bead]

        for bead_info in selected_bead_data:
            file_name = bead_info["file"]
            start_idx = bead_info["start_index"]
            end_idx = bead_info["end_index"]

            # Load data and extract the signal for the specific bead
            df = pd.read_csv(file_name)
            signal = df.iloc[start_idx:end_idx + 1, 0].values  # Extract only the bead's signal

            # Get anomaly status and score
            status = st.session_state["anomaly_results"].get((file_name, selected_bead), "normal")
            anomaly_score = st.session_state["anomaly_scores"].get((file_name, selected_bead), 0)

            # Set color based on anomaly status
            color = 'red' if status == 'anomalous' else 'black'

            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                line=dict(color=color, width=1),
                name=f"{file_name}",
                hoverinfo='text',
                text=f"File: {file_name}<br>Status: {status}<br>Anomaly Score: {anomaly_score:.4f}"
            ))

        fig.update_layout(
            title=f"Bead Number {selected_bead}: Anomaly Detection Results",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )

        st.plotly_chart(fig)
