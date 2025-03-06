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
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
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
            spectral_entropy, autocorrelation, rms, 
            slope, moving_average, outlier_count, extreme_event_duration]

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection V8 - V4 with Feature Selection")

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
            st.session_state["chosen_bead_data"] = chosen_bead_data
            st.success("Beads selected successfully!")
        
        # Feature selection
        feature_names = ["Mean Value", "STD Value", "Min Value", "Max Value", "Median Value", "Skewness", "Kurtosis", "Peak-to-Peak", "Energy", "Coefficient of Variation (CV)",
                         "Spectral Entropy", "Autocorrelation", "Root Mean Square (RMS)", "Slope", "Moving Average",
                         "Outlier Count", "Extreme Event Duration"]
        options = ["All"] + feature_names
        
        selected_features = st.multiselect(
            "Select features to use for Isolation Forest",
            options=options,
            default="All"
        )
        
        if "All" in selected_features and len(selected_features) > 1:
            selected_features = ["All"]
        elif "All" not in selected_features and len(selected_features) == 0:
            st.error("You must select at least one feature.")
            st.stop()
        if "All" in selected_features:
            selected_features = feature_names
        selected_indices = [feature_names.index(f) for f in selected_features]
        
        # Contamination rate logic in the sidebar section
        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        use_contamination_rate = st.checkbox("Use Contamination Rate", value=True)
        
        # Modify the IsolationForest call to either include or omit contamination based on user choice
        if st.button("Run Isolation Forest") and "chosen_bead_data" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                anomaly_results_isoforest = {}
                anomaly_scores_isoforest = {}
                for bead_number in sorted(set(seg["bead_number"] for seg in st.session_state["chosen_bead_data"])): 
                    bead_data = [seg for seg in st.session_state["chosen_bead_data"] if seg["bead_number"] == bead_number]
                    signals = [seg["data"].iloc[:, 0].values for seg in bead_data]
                    file_names = [seg["file"] for seg in bead_data]
                    feature_matrix = np.array([extract_advanced_features(signal) for signal in signals])
                    feature_matrix = feature_matrix[:, selected_indices]  # Select only the chosen features
                    scaler = RobustScaler()
                    feature_matrix = scaler.fit_transform(feature_matrix)
        
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


st.write("## Visualization")
if "chosen_bead_data" in st.session_state and "anomaly_results_isoforest" in locals():
    for bead_number, results in anomaly_results_isoforest.items():
        bead_data = [seg for seg in st.session_state["chosen_bead_data"] if seg["bead_number"] == bead_number]
        file_names = [seg["file"] for seg in bead_data]
        signals = [seg["data"].iloc[:, 0].values for seg in bead_data]
        fig = go.Figure()
        for idx, signal in enumerate(signals):
            file_name = file_names[idx]
            status = results[file_name]
            anomaly_score = anomaly_scores_isoforest[bead_number][file_name]
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
            title=f"Bead Number {bead_number}: Anomaly Detection Results",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
    st.success("Anomaly detection complete!")
