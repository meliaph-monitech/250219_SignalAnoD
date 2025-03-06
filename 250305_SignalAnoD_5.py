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

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv, 
            dominant_freq, spectral_entropy, autocorrelation, peak_count, zero_crossing_rate, rms]

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection Across All Beads")

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
                bead_data = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        bead_segment = df.iloc[start:end + 1]
                        bead_data.append({
                            "data": bead_segment, "file": file, "bead_number": bead_num,
                            "start_index": start, "end_index": end
                        })
                st.session_state["bead_data"] = bead_data
                st.success("Bead segmentation complete")
        
        contamination_rate = st.slider("Set Contamination Rate", 0.01, 0.5, 0.1, 0.01)
        use_contamination_rate = st.checkbox("Use Contamination Rate", value=True)
        
        if st.button("Run Isolation Forest") and "bead_data" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                feature_matrix = np.array([extract_advanced_features(seg["data"].iloc[:, 0].values) for seg in st.session_state["bead_data"]])
                scaler = RobustScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)
                
                if use_contamination_rate:
                    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                else:
                    iso_forest = IsolationForest(random_state=42)
                
                predictions = iso_forest.fit_predict(feature_matrix)
                anomaly_scores = -iso_forest.decision_function(feature_matrix)
                
                for idx, seg in enumerate(st.session_state["bead_data"]):
                    seg["status"] = 'anomalous' if predictions[idx] == -1 else 'normal'
                    seg["anomaly_score"] = anomaly_scores[idx]
                
                st.session_state["anomaly_results"] = st.session_state["bead_data"]
                st.success("Anomaly detection complete!")

st.write("## Visualization")
if "anomaly_results" in st.session_state:
    fig = go.Figure()
    for seg in st.session_state["anomaly_results"]:
        signal = seg["data"].iloc[:, 0].values
        color = 'red' if seg["status"] == 'anomalous' else 'black'
        fig.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color=color), 
                                 name=f"{seg['file']} - Bead {seg['bead_number']}",
                                 text=f"File: {seg['file']}<br>Status: {seg['status']}<br>Anomaly Score: {seg['anomaly_score']:.4f}",
                                 hoverinfo='text'))
    fig.update_layout(title="Anomaly Detection Across All Beads", xaxis_title="Time Index", yaxis_title="Signal Value", showlegend=True)
    st.plotly_chart(fig)
