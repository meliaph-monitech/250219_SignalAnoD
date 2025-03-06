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
    dominant_freq = positive_freqs[np.argmax(positive_psd)] if len(positive_psd) > 0 else 0
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    peaks, _ = find_peaks(signal)
    peak_count = len(peaks)
    zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / n
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
            dominant_freq, spectral_entropy, autocorrelation, peak_count, zero_crossing_rate, rms, 
            slope, moving_average, outlier_count, extreme_event_duration]

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

# Visualization
st.write("## Visualization")
if "anomaly_results_isoforest" in st.session_state:
    bead_numbers = sorted(set(num for _, num in st.session_state["anomaly_results_isoforest"].keys()))
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)
    
    if selected_bead:
        fig = go.Figure()
        for (file_name, bead_num), status in st.session_state["anomaly_results_isoforest"].items():
            if bead_num == selected_bead:
                df = pd.read_csv(file_name)
                signal = df.iloc[:, 0].values
                anomaly_score = st.session_state["anomaly_scores_isoforest"].get((file_name, bead_num), 0)
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
        st.success("Anomaly detection complete!")
