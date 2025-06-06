## ADVANCED FEATURE EXTRACTION
import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
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
    positive_psd = psd[:n // 2]
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    rms = np.sqrt(np.mean(signal**2))

    # Handle slope
    x = np.arange(n)
    if len(set(signal)) == 1 or len(signal) < 2:
        slope = 0
    else:
        try:
            slope, _ = np.polyfit(x, signal, 1)
        except np.linalg.LinAlgError:
            slope = 0

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak,
            energy, cv, spectral_entropy, rms, slope]

def normalize_signal_per_bead(bead_data):
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
st.title("Laser Welding Clustering Visualization")

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
        
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        
        if st.button("Run K-Means Clustering") and "chosen_bead_data" in st.session_state:
            with st.spinner("Running K-Means Clustering..."):
                bead_data = st.session_state["chosen_bead_data"]
                normalized_signals = [seg["data"]["normalized_signal"].values for seg in bead_data]
                file_names = [seg["file"] for seg in bead_data]
                feature_matrix = np.array([extract_advanced_features(signal) for signal in normalized_signals])
                scaler = RobustScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)
                
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(feature_matrix)
                
                cluster_results = {}
                for idx, cluster in enumerate(clusters):
                    cluster_results[file_names[idx]] = cluster
                
                st.session_state["cluster_results"] = cluster_results
                st.success("K-Means clustering complete!")

if "cluster_results" in st.session_state:
    st.write("## Visualization")
    bead_data = st.session_state["chosen_bead_data"]
    cluster_results = st.session_state["cluster_results"]
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
    
    for bead_number in set(seg["bead_number"] for seg in bead_data):
        bead_segments = [seg for seg in bead_data if seg["bead_number"] == bead_number]
        file_names = [seg["file"] for seg in bead_segments]
        fig = go.Figure()
        for idx, segment in enumerate(bead_segments):
            file_name = segment["file"]
            cluster = cluster_results[file_name]
            raw_signal = segment["raw_signal"]
            color = colors[cluster % len(colors)]
            fig.add_trace(go.Scatter(
                y=raw_signal,
                mode='lines',
                line=dict(color=color, width=1),
                name=f"{file_name} (Cluster {cluster})",
                hoverinfo='text',
                text=f"File: {file_name}<br>Cluster: {cluster}"
            ))
        fig.update_layout(
            title=f"Bead Number {bead_number}: Clustering Results (Raw Data)",
            xaxis_title="Time Index",
            yaxis_title="Raw Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
