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

    # FFT calculations
    signal_fft = fft(signal)
    psd = np.abs(signal_fft)**2
    freqs = fftfreq(n, 1)
    psd_normalized = psd[:n // 2] / np.sum(psd[:n // 2]) if np.sum(psd[:n // 2]) > 0 else np.zeros_like(psd[:n // 2])
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    rms = np.sqrt(np.mean(signal**2))

    # Handle slope
    x = np.arange(n)
    slope = np.polyfit(x, signal, 1)[0] if len(set(signal)) > 1 else 0

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak,
            energy, cv, spectral_entropy, rms, slope]

def normalize_signal_per_bead(bead_data):
    """Normalizes signal data per bead."""
    for bead in bead_data:
        raw_signal = bead["data"].iloc[:, 0].values
        scaler = MinMaxScaler()
        if len(raw_signal) > 1:
            normalized_signal = scaler.fit_transform(raw_signal.reshape(-1, 1)).flatten()
        else:
            normalized_signal = raw_signal
        bead["data"]["normalized_signal"] = normalized_signal
        bead["raw_signal"] = raw_signal
    return bead_data

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection V13")

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

        if "metadata" in st.session_state:
            bead_data = []
            for entry in st.session_state["metadata"]:
                df = pd.read_csv(entry["file"])
                bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"]})

            bead_data = normalize_signal_per_bead(bead_data)
            st.session_state["chosen_bead_data"] = bead_data

        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        use_contamination_rate = st.checkbox("Use Contamination Rate", value=True)

        if st.button("Run Isolation Forest") and "chosen_bead_data" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                features_by_bead = defaultdict(list)
                files_by_bead = defaultdict(list)

                for entry in st.session_state["chosen_bead_data"]:
                    raw_signal = entry["raw_signal"]
                    features = extract_advanced_features(raw_signal)
                    bead_number = entry["bead_number"]
                    features_by_bead[bead_number].append(features)
                    files_by_bead[bead_number].append(entry["file"])

                # Normalize features per bead number
                scaled_features_by_bead = {}
                for bead_number, feature_matrix in features_by_bead.items():
                    scaler = RobustScaler()
                    scaled_features_by_bead[bead_number] = scaler.fit_transform(feature_matrix)

                # Combine all scaled features into a single matrix
                all_scaled_features = []
                all_file_names = []
                for bead_number, scaled_features in scaled_features_by_bead.items():
                    all_scaled_features.extend(scaled_features)
                    all_file_names.extend(files_by_bead[bead_number])

                # Convert the list to a NumPy array for Isolation Forest
                all_scaled_features = np.array(all_scaled_features)

                # Train Isolation Forest
                iso_forest = IsolationForest(contamination=contamination_rate if use_contamination_rate else 'auto', random_state=42)
                predictions = iso_forest.fit_predict(all_scaled_features)
                anomaly_scores = -iso_forest.decision_function(all_scaled_features)

                # Save results
                st.session_state["anomaly_results_isoforest"] = {fn: ('anomalous' if p == -1 else 'normal') for fn, p in zip(all_file_names, predictions)}
                st.session_state["anomaly_scores_isoforest"] = {fn: s for fn, s in zip(all_file_names, anomaly_scores)}

if "anomaly_results_isoforest" in st.session_state:
    if st.button("Generate Result"):
        results_df = pd.DataFrame([
            {"File Name": file_name, "Bead Number": bead_num, "Status": status}
            for (file_name, bead_num), status in st.session_state["anomaly_results_isoforest"].items()
        ])

        # Convert DataFrame to CSV
        csv_data = results_df.to_csv(index=False).encode('utf-8')

        # Create a download button
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )

st.write("## Visualization")
if "anomaly_results_isoforest" in st.session_state:
    visualization_type = st.sidebar.radio(
        "Select Visualization Type",
        ("Raw Signal", "Normalized Signal")
    )
    
    bead_numbers = sorted(set(num for _, num in st.session_state["anomaly_results_isoforest"].keys()))
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)

    if selected_bead:
        fig = go.Figure()

        # Filter data for the selected bead number
        selected_bead_data = [entry for entry in st.session_state["chosen_bead_data"] if entry["bead_number"] == selected_bead]

        for bead_info in selected_bead_data:
            file_name = bead_info["file"]
            raw_signal = bead_info["raw_signal"]
            normalized_signal = bead_info["data"]["normalized_signal"].values  # Extract normalized signal

            # Get anomaly status and score
            status = st.session_state["anomaly_results_isoforest"].get((file_name, selected_bead), "normal")
            anomaly_score = st.session_state["anomaly_scores_isoforest"].get((file_name, selected_bead), 0)

            # Choose signal to visualize based on user selection
            signal_to_plot = normalized_signal if visualization_type == "Normalized Signal" else raw_signal
            color = 'red' if status == 'anomalous' else 'black'

            fig.add_trace(go.Scatter(
                y=signal_to_plot,
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
