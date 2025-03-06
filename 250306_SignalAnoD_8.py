import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.signal import spectrogram
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


def calculate_bead_zscores(signal, start_points, end_points, Fs, window_size, noverlap, nfft, target_frequency, spectrogram_dB_scale):
    """Calculates a representative Z-score for each bead."""
    bead_zscores = []
    all_intensities = []

    # Step 1: Calculate spectrogram for each bead and extract relevant intensities
    for start, end in zip(start_points, end_points):
        bead_data = signal[start:end + 1]
        fsg, tsg, sg = spectrogram(bead_data, fs=Fs, nperseg=window_size, noverlap=noverlap, nfft=nfft, mode='magnitude')
        
        sg_dBpeak = 20 * np.log10(np.abs(sg)) + 20 * np.log10(2 / len(fsg))
        min_disp_dB = np.round(np.max(sg_dBpeak)) - spectrogram_dB_scale
        sg_dBpeak[sg_dBpeak < min_disp_dB] = min_disp_dB

        # Extract target frequency range
        target_freq_indices = np.where((fsg >= target_frequency - 5) & (fsg <= target_frequency + 5))[0]
        target_sg_avg_intensity_over_time = np.mean(sg_dBpeak[target_freq_indices, :] - min_disp_dB, axis=0)
        all_intensities.extend(target_sg_avg_intensity_over_time)

    # Step 2: Calculate overall mean and standard deviation
    overall_mean = np.mean(all_intensities)
    overall_std = np.std(all_intensities)

    # Step 3: Calculate Z-scores for each bead
    for start, end in zip(start_points, end_points):
        bead_data = signal[start:end + 1]
        fsg, tsg, sg = spectrogram(bead_data, fs=Fs, nperseg=window_size, noverlap=noverlap, nfft=nfft, mode='magnitude')
        
        sg_dBpeak = 20 * np.log10(np.abs(sg)) + 20 * np.log10(2 / len(fsg))
        min_disp_dB = np.round(np.max(sg_dBpeak)) - spectrogram_dB_scale
        sg_dBpeak[sg_dBpeak < min_disp_dB] = min_disp_dB

        target_freq_indices = np.where((fsg >= target_frequency - 5) & (fsg <= target_frequency + 5))[0]
        target_sg_avg_intensity_over_time = np.mean(sg_dBpeak[target_freq_indices, :] - min_disp_dB, axis=0)

        z_scores = (target_sg_avg_intensity_over_time - overall_mean) / overall_std
        bead_zscores.append(np.mean(z_scores))

    return bead_zscores


st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection with Z-scores")

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
        
        Fs = st.number_input("Sampling Frequency", value=1000)  # Example value, user can set it
        window_size = st.number_input("Window Size", value=256)
        noverlap = st.number_input("Number of Overlaps", value=128)
        nfft = st.number_input("NFFT", value=512)
        target_frequency = st.number_input("Target Frequency", value=50.0)
        spectrogram_dB_scale = st.number_input("Spectrogram dB Scale", value=70.0)

        if st.button("Calculate Z-scores") and "metadata" in st.session_state:
            with st.spinner("Calculating Z-scores..."):
                zscores_by_bead = defaultdict(list)

                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    signal = df[filter_column].to_numpy()
                    segments = segment_beads(df, filter_column, threshold)
                    start_points, end_points = zip(*segments)
                    bead_zscores = calculate_bead_zscores(
                        signal, start_points, end_points, Fs, window_size, noverlap, nfft, target_frequency, spectrogram_dB_scale
                    )
                    for bead_num, zscore in enumerate(bead_zscores, start=1):
                        zscores_by_bead[bead_num].append(zscore)

                st.success("Z-score calculation complete")
                st.session_state["zscores_by_bead"] = zscores_by_bead

        if st.button("Run Isolation Forest") and "zscores_by_bead" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                all_zscores = []
                bead_labels = []

                for bead_num, zscores in st.session_state["zscores_by_bead"].items():
                    all_zscores.extend(zscores)
                    bead_labels.extend([bead_num] * len(zscores))

                all_zscores = np.array(all_zscores).reshape(-1, 1)
                scaler = RobustScaler()
                scaled_zscores = scaler.fit_transform(all_zscores)

                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(scaled_zscores)
                anomaly_scores = -iso_forest.decision_function(scaled_zscores)

                st.session_state["anomaly_results"] = {
                    bead: ("anomalous" if pred == -1 else "normal") for bead, pred in zip(bead_labels, predictions)
                }
                st.session_state["anomaly_scores"] = {
                    bead: score for bead, score in zip(bead_labels, anomaly_scores)
                }

st.write("## Visualization")
if "anomaly_results" in st.session_state:
    bead_numbers = sorted(st.session_state["anomaly_results"].keys())
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)

    if selected_bead:
        st.write(f"Bead {selected_bead} is {st.session_state['anomaly_results'][selected_bead]}")
