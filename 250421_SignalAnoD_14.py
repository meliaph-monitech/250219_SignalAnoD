import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.signal import spectrogram

# === Helper Functions ===

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


def extract_frequency_features(signal, fs, fmin, fmax, nperseg, noverlap, nfft):
    """
    Extracts frequency-domain features (spectrogram intensities) within a selected frequency range.
    """
    # Compute the spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)  # Convert to decibels

    # Select frequencies within the range [fmin, fmax]
    band_indices = np.where((f >= fmin) & (f <= fmax))[0]
    if len(band_indices) == 0:
        return np.zeros(len(t))  # Return zeros if no frequencies are in the range

    # Average the intensities over the selected frequency range
    frequency_features = np.mean(Sxx_dB[band_indices, :], axis=0)

    return frequency_features

# === Streamlit App ===

st.set_page_config(layout="wide")
st.title("Anomaly Detection with Frequency Domain Features")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files, extract_dir = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        # Load sample file to get column names
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)

        # Frequency domain parameters
        fs = st.number_input("Sampling frequency (Hz)", min_value=1000, value=10000)
        fmin = st.number_input("Min frequency (Hz)", min_value=1, value=100)
        fmax = st.number_input("Max frequency (Hz)", min_value=1, value=500)
        nperseg = st.number_input("nperseg", min_value=128, max_value=8192, value=1024)
        noverlap_ratio = st.slider("Overlap ratio", 0.0, 0.99, value=0.5)
        noverlap = int(nperseg * noverlap_ratio)
        nfft = st.number_input("nfft", min_value=256, value=2048)
        
        # Contamination rate for Isolation Forest
        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

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

        if st.button("Run Isolation Forest") and "metadata" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                features_by_bead = defaultdict(list)
                files_by_bead = defaultdict(list)

                # Group features by bead number
                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    signal = bead_segment.iloc[:, 0].values

                    # Extract frequency features
                    frequency_features = extract_frequency_features(signal, fs, fmin, fmax, nperseg, noverlap, nfft)
                    bead_number = entry["bead_number"]

                    features_by_bead[bead_number].append(frequency_features)
                    files_by_bead[bead_number].append((entry["file"], bead_number))

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
                iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                predictions = iso_forest.fit_predict(all_scaled_features)
                anomaly_scores = -iso_forest.decision_function(all_scaled_features)

                # Save results
                st.session_state["anomaly_results_isoforest"] = {fn: ('anomalous' if p == -1 else 'normal') for fn, p in zip(all_file_names, predictions)}
                st.session_state["anomaly_scores_isoforest"] = {fn: s for fn, s in zip(all_file_names, anomaly_scores)}

if "anomaly_results_isoforest" in st.session_state:
    st.write("## Results")
    if st.button("Generate Result"):
        # Prepare data for export
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
