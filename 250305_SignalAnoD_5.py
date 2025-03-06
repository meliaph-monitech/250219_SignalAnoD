import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import numpy as np

# Function to extract ZIP file
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

# Function to segment beads
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

# Feature extraction function
def extract_advanced_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 10  # Reduced feature count for simplicity
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    median_val = np.median(signal)
    peak_to_peak = max_val - min_val
    energy = np.sum(signal**2)
    zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / n
    rms = np.sqrt(np.mean(signal**2))
    
    return [mean_val, std_val, min_val, max_val, median_val, peak_to_peak, energy, zero_crossing_rate, rms]

st.set_page_config(layout="wide")
st.title("Global Anomaly Detection for Laser Welding")

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
                bead_segments = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        bead_segments.append({
                            "file": file,
                            "bead_number": bead_num,
                            "start_index": start,
                            "end_index": end,
                            "data": df.iloc[start:end + 1]
                        })
                st.session_state["bead_segments"] = bead_segments
                st.success("Bead segmentation complete")
        
        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        if st.button("Run Isolation Forest") and "bead_segments" in st.session_state:
            with st.spinner("Running Isolation Forest on all beads..."):
                bead_segments = st.session_state["bead_segments"]
                feature_matrix = np.array([extract_advanced_features(seg["data"].iloc[:, 0].values) for seg in bead_segments])
                scaler = RobustScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)
                
                iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                predictions = iso_forest.fit_predict(feature_matrix)
                anomaly_scores = -iso_forest.decision_function(feature_matrix)
                
                for idx, seg in enumerate(bead_segments):
                    seg["status"] = "anomalous" if predictions[idx] == -1 else "normal"
                    seg["anomaly_score"] = anomaly_scores[idx]
                
                st.session_state["anomaly_results"] = bead_segments
                st.success("Anomaly detection complete!")

st.write("## Visualization")
if "anomaly_results" in st.session_state:
    for seg in st.session_state["anomaly_results"]:
        fig = go.Figure()
        signal = seg["data"].iloc[:, 0].values
        color = 'red' if seg["status"] == 'anomalous' else 'black'
        
        fig.add_trace(go.Scatter(
            y=signal,
            mode='lines',
            line=dict(color=color, width=1),
            name=f"File: {seg['file']} | Bead: {seg['bead_number']}",
            hoverinfo='text',
            text=f"File: {seg['file']}<br>Bead: {seg['bead_number']}<br>Status: {seg['status']}<br>Anomaly Score: {seg['anomaly_score']:.4f}"
        ))
        
        fig.update_layout(
            title=f"Bead {seg['bead_number']} from {seg['file']}: Anomaly Detection",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
