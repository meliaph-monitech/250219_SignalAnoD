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
st.title("Laser Welding Anomaly Detection V15 - Global Analysis with Feature Selection")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files, extract_dir = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        # Load the first CSV file as a sample to get column names
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        if len(columns) < 2:
            st.error("The CSV files must have at least two columns for this application.")
            st.stop()
        
        # Add an option to choose between the first or second column
        column_to_analyze_option = st.radio(
            "Select which column to analyze:",
            options=["First Column", "Second Column"]
        )
        if column_to_analyze_option == "First Column":
            column_to_analyze = columns[0]
        else:
            column_to_analyze = columns[1]
        
        threshold = st.number_input("Enter filtering threshold", value=0.0)

        feature_names = ["Mean Value", "STD Value", "Min Value", "Max Value", "Median Value", "Skewness", "Kurtosis", "Peak-to-Peak", "Energy", "Coefficient of Variation (CV)",
                         "Spectral Entropy", "Autocorrelation", "Root Mean Square (RMS)", "Slope", "Moving Average",
                         "Outlier Count", "Extreme Event Duration"]
        options = ["All"] + feature_names
        selected_features = st.multiselect("Select features to use for Isolation Forest", options=options, default="All")

        if "All" in selected_features and len(selected_features) > 1:
            selected_features = ["All"]
        elif "All" not in selected_features and len(selected_features) == 0:
            st.error("You must select at least one feature.")
            st.stop()
        
        if "All" in selected_features:
            selected_features = feature_names
        
        selected_indices = [feature_names.index(f) for f in selected_features]

        if st.button("Segment Beads"):
            with st.spinner("Segmenting beads..."):
                bead_segments = {}
                metadata = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, column_to_analyze, threshold)
                    if segments:
                        bead_segments[file] = segments
                        for bead_num, (start, end) in enumerate(segments, start=1):
                            metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
                st.success("Bead segmentation complete")
                st.session_state["metadata"] = metadata
        
        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        use_contamination_rate = st.checkbox("Use Contamination Rate", value=True)

        if st.button("Run Isolation Forest") and "metadata" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                features_by_bead = defaultdict(list)
                files_by_bead = defaultdict(list)

                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    features = extract_advanced_features(bead_segment[column_to_analyze].values)
                    bead_number = entry["bead_number"]
                    features_by_bead[bead_number].append([features[i] for i in selected_indices])
                    files_by_bead[bead_number].append((entry["file"], bead_number))

                scaled_features_by_bead = {}
                for bead_number, feature_matrix in features_by_bead.items():
                    scaler = RobustScaler()
                    scaled_features_by_bead[bead_number] = scaler.fit_transform(feature_matrix)

                all_scaled_features = []
                all_file_names = []
                for bead_number, scaled_features in scaled_features_by_bead.items():
                    all_scaled_features.extend(scaled_features)
                    all_file_names.extend(files_by_bead[bead_number])

                all_scaled_features = np.array(all_scaled_features)

                iso_forest = IsolationForest(contamination=contamination_rate if use_contamination_rate else 'auto', random_state=42)
                predictions = iso_forest.fit_predict(all_scaled_features)
                anomaly_scores = -iso_forest.decision_function(all_scaled_features)

                st.session_state["anomaly_results_isoforest"] = {fn: ('anomalous' if p == -1 else 'normal') for fn, p in zip(all_file_names, predictions)}
                st.session_state["anomaly_scores_isoforest"] = {fn: s for fn, s in zip(all_file_names, anomaly_scores)}

if "anomaly_results_isoforest" in st.session_state:
    st.write("## Anomaly Detection Results")
    results_df = pd.DataFrame([
        {
            "File Name": file_name, 
            "Bead Number": bead_num, 
            "Anomaly Score": round(st.session_state["anomaly_scores_isoforest"][(file_name, bead_num)], 4), 
            "Notation": f":red[Anomalous]" if st.session_state["anomaly_results_isoforest"][(file_name, bead_num)] == 'anomalous' else "Normal"
        }
        for (file_name, bead_num) in st.session_state["anomaly_results_isoforest"]
    ])
    st.dataframe(results_df, height=400)

    st.write("## Visualization")
    bead_numbers = sorted(set(num for _, num in st.session_state["anomaly_results_isoforest"].keys()))
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)

    if selected_bead:
        fig = go.Figure()
        selected_bead_data = [entry for entry in st.session_state["metadata"] if entry["bead_number"] == selected_bead]
        traces = []

        for bead_info in selected_bead_data:
            file_name = bead_info["file"]
            start_idx = bead_info["start_index"]
            end_idx = bead_info["end_index"]
            df = pd.read_csv(file_name)
            signal = df.iloc[start_idx:end_idx + 1, 0].values  # Use the first column for plotting, as originally requested.
            status = st.session_state["anomaly_results_isoforest"].get((file_name, selected_bead), "normal")
            anomaly_score = st.session_state["anomaly_scores_isoforest"].get((file_name, selected_bead), 0)
            color = 'red' if status == 'anomalous' else 'black'

            trace = go.Scatter(
                y=signal,
                mode='lines',
                line=dict(color=color, width=1),
                name=f"{file_name} ({status})",
                hoverinfo='text',
                text=f"File: {file_name}<br>Status: {status}<br>Anomaly Score: {anomaly_score:.4f}"
            )
            traces.append((status, trace))

        sorted_traces = sorted(traces, key=lambda x: 0 if x[0] == 'anomalous' else 1)
        for _, trace in sorted_traces:
            fig.add_trace(trace)

        fig.update_layout(
            title=f"Bead Number {selected_bead}: Anomaly Detection Results",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
