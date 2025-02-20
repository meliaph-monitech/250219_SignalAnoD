import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import numpy as np

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    os.makedirs(extract_dir, exist_ok=True)
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

def extract_time_freq_features(signal):
    n = len(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    energy = np.sum(np.square(signal)) / n
    skewness = skew(signal)
    kurt = kurtosis(signal)
    fft_values = fft(signal)
    fft_magnitude = np.abs(fft_values)[:n // 2]
    spectral_energy = np.sum(fft_magnitude ** 2) / len(fft_magnitude)
    dominant_freq = np.argmax(fft_magnitude)
    return [mean_val, std_val, min_val, max_val, energy, skewness, kurt, spectral_energy, dominant_freq]

st.title("Laser Welding Anomaly Detection")
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
    
    if st.button("Run Isolation Forest") and "chosen_bead_data" in st.session_state:
        anomaly_results_isoforest = {}
        anomaly_scores_isoforest = {}
        feature_data = []
        feature_matrix_list = []
        file_names_list = []
        
        for bead_number in sorted(set(seg["bead_number"] for seg in st.session_state["chosen_bead_data"])):
            bead_data = [seg for seg in st.session_state["chosen_bead_data"] if seg["bead_number"] == bead_number]
            signals = [seg["data"].iloc[:, 0].values for seg in bead_data]
            file_names = [seg["file"] for seg in bead_data]
            feature_matrix = np.array([extract_time_freq_features(signal) for signal in signals])
            iso_forest = IsolationForest(contamination=0.2, random_state=42)
            predictions = iso_forest.fit_predict(feature_matrix)
            anomaly_scores = -iso_forest.decision_function(feature_matrix)
            
            bead_results = {}
            bead_scores = {}
            for idx, prediction in enumerate(predictions):
                status = 'anomalous' if prediction == -1 else 'normal'
                bead_results[file_names[idx]] = status
                bead_scores[file_names[idx]] = anomaly_scores[idx]
                feature_matrix_list.append(feature_matrix[idx])
                file_names_list.append(file_names[idx])
            
            anomaly_results_isoforest[bead_number] = bead_results
            anomaly_scores_isoforest[bead_number] = bead_scores
        
        pca = PCA(n_components=3)
        reduced_features = pca.fit_transform(feature_matrix_list)
        
        scatter_fig_3d = go.Figure()
        for i, (x, y, z) in enumerate(reduced_features):
            status = anomaly_results_isoforest[bead_number][file_names_list[i]]
            color = 'red' if status == 'anomalous' else 'black'
            scatter_fig_3d.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(color=color, size=5),
                text=f"File: {file_names_list[i]}<br>Status: {status}<br>Features: {feature_matrix_list[i]}",
                hoverinfo='text'
            ))
        scatter_fig_3d.update_layout(title="3D PCA Feature Space", scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
        st.plotly_chart(scatter_fig_3d)
