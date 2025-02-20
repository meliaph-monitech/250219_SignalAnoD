import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
from scipy.fft import fft

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

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
    return [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        np.sum(np.square(signal)) / n, skew(signal), kurtosis(signal),
        np.sum(np.abs(fft(signal)[:n // 2]) ** 2) / (n // 2), np.argmax(np.abs(fft(signal)[:n // 2]))
    ]

st.title("Laser Welding Signal Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload ZIP file containing CSVs", type=["zip"])
if uploaded_file:
    zip_path = "uploaded.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_files = extract_zip(zip_path)
    st.success(f"Extracted {len(csv_files)} CSV files.")
    
    # Select column and threshold
    df_sample = pd.read_csv(csv_files[0])
    column = st.selectbox("Select filter column", df_sample.columns)
    threshold = st.number_input("Enter filtering threshold", value=0.0)
    
    if st.button("Segment Beads"):
        metadata = []
        bead_segments = {}
        for file in csv_files:
            df = pd.read_csv(file)
            segments = segment_beads(df, column, threshold)
            if segments:
                bead_segments[file] = segments
                for bead_num, (start, end) in enumerate(segments, start=1):
                    metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
        st.session_state.metadata = metadata
        st.success("Bead segmentation complete!")

    if "metadata" in st.session_state:
        bead_numbers = sorted(set(m["bead_number"] for m in st.session_state.metadata))
        selected_beads = st.multiselect("Select Bead Numbers", bead_numbers, default=bead_numbers)
        
        if st.button("Analyze with Isolation Forest"):
            chosen_bead_data = []
            for entry in st.session_state.metadata:
                if entry["bead_number"] in selected_beads:
                    df = pd.read_csv(entry["file"])
                    segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    chosen_bead_data.append({"data": segment, "file": entry["file"], "bead_number": entry["bead_number"]})
            
            # Feature Extraction & Anomaly Detection
            anomaly_results = {}
            anomaly_scores = {}
            for bead_number in selected_beads:
                bead_data = [b for b in chosen_bead_data if b["bead_number"] == bead_number]
                signals = [b["data"].iloc[:, 0].values for b in bead_data]
                file_names = [b["file"] for b in bead_data]
                
                feature_matrix = np.array([extract_features(signal) for signal in signals])
                iso_forest = IsolationForest(contamination=0.2, random_state=42)
                predictions = iso_forest.fit_predict(feature_matrix)
                scores = -iso_forest.decision_function(feature_matrix)
                
                results = {file_names[i]: 'anomalous' if predictions[i] == -1 else 'normal' for i in range(len(file_names))}
                anomaly_results[bead_number] = results
                anomaly_scores[bead_number] = {file_names[i]: scores[i] for i in range(len(file_names))}
            
            # Visualization
            for bead_number, results in anomaly_results.items():
                fig = go.Figure()
                bead_data = [b for b in chosen_bead_data if b["bead_number"] == bead_number]
                file_names = [b["file"] for b in bead_data]
                signals = [b["data"].iloc[:, 0].values for b in bead_data]
                
                for i, signal in enumerate(signals):
                    color = 'red' if results[file_names[i]] == 'anomalous' else 'black'
                    fig.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color=color, width=1), name=file_names[i]))
                
                fig.update_layout(title=f"Bead Number {bead_number}", xaxis_title="Time Index", yaxis_title="Signal Value", showlegend=False)
                st.plotly_chart(fig)
            st.success("Analysis Complete!")
