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
    return [os.path.join(extract_dir, f) for f in csv_files]

def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
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

def extract_features(signal):
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
    dominant_freq = freqs[np.argmax(psd[:n // 2])]
    
    peaks, _ = find_peaks(signal)
    peak_count = len(peaks)
    zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / n
    rms = np.sqrt(np.mean(signal**2))
    slope, _ = np.polyfit(np.arange(n), signal, 1)
    
    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv,
            dominant_freq, peak_count, zero_crossing_rate, rms, slope]

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

with st.sidebar:
    st.subheader("Upload Training Data (Normal Only)")
    train_zip = st.file_uploader("Upload a ZIP file containing CSVs", type=["zip"], key="train")
    
    if train_zip:
        with open("train.zip", "wb") as f:
            f.write(train_zip.getbuffer())
        train_csvs = extract_zip("train.zip")
        st.success(f"Extracted {len(train_csvs)} CSV files for training")
        
        df_sample = pd.read_csv(train_csvs[0])
        filter_column = st.selectbox("Select column for filtering", df_sample.columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        
        bead_segments = {}
        for file in train_csvs:
            df = pd.read_csv(file)
            segments = segment_beads(df, filter_column, threshold)
            bead_segments[file] = segments
        
        bead_numbers = st.text_input("Enter bead numbers (comma-separated)")
        
        if st.button("Train Model"):
            selected_beads = [int(b.strip()) for b in bead_numbers.split(",") if b.strip().isdigit()]
            training_data = []
            for file, segments in bead_segments.items():
                df = pd.read_csv(file)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    if bead_num in selected_beads:
                        training_data.append(extract_features(df.iloc[start:end + 1][filter_column].values))
            
            if training_data:
                X_train = np.array(training_data)
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                model = IsolationForest(contamination=0, random_state=42)
                model.fit(X_train)
                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["selected_beads"] = selected_beads
                st.success("Model trained successfully!")

st.subheader("Upload New Data for Prediction")
new_file = st.file_uploader("Upload a CSV file", type=["csv"], key="predict")

if new_file and "model" in st.session_state:
    df_new = pd.read_csv(new_file)
    segments = segment_beads(df_new, filter_column, threshold)
    predictions = {}
    
    for bead_num, (start, end) in enumerate(segments, start=1):
        if bead_num in st.session_state["selected_beads"]:
            features = extract_features(df_new.iloc[start:end + 1][filter_column].values)
            X_test = st.session_state["scaler"].transform([features])
            pred = st.session_state["model"].predict(X_test)[0]
            predictions[bead_num] = "anomalous" if pred == -1 else "normal"
    
    st.write("### Predictions")
    for bead, status in predictions.items():
        st.write(f"Bead {bead}: {status}")
    
    # Visualization
    st.write("### Visualization")
    fig = go.Figure()
    for bead_num, (start, end) in enumerate(segments, start=1):
        color = "red" if predictions.get(bead_num, "normal") == "anomalous" else "blue"
        fig.add_trace(go.Scatter(y=df_new.iloc[start:end + 1][filter_column].values, mode='lines', line=dict(color=color), name=f"Bead {bead_num}"))
    
    fig.update_layout(title="Signal Visualization", xaxis_title="Time Index", yaxis_title="Signal Value")
    st.plotly_chart(fig)
    st.success("Anomaly detection complete!")
