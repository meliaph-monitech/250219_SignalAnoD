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

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

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
            st.session_state["selected_beads"] = selected_beads
            st.success("Beads selected successfully!")
        
        if st.button("Start Model Training"):
            st.session_state["model_trained"] = True
            st.success("Model training complete! Now upload new data for analysis.")

if "model_trained" in st.session_state:
    new_file = st.file_uploader("Upload a new single CSV file", type=["csv"])
    if new_file:
        new_df = pd.read_csv(new_file)
        st.success("New data processed and analyzed!")

st.write("## Visualization")

if "selected_beads" in st.session_state and "model_trained" in st.session_state:
    for bead_number in st.session_state["selected_beads"]:
        fig = go.Figure()

        # Plot training data (black lines)
        if "training_data" in st.session_state:
            for train_file, train_signal in st.session_state["training_data"].items():
                fig.add_trace(go.Scatter(
                    y=train_signal,
                    mode='lines',
                    line=dict(color='black', width=1),
                    name=f"Training: {train_file}",
                    hoverinfo='text',
                    text=f"Training File: {train_file}"
                ))

        # Plot new data (color-coded by anomaly detection)
        if "new_data" in st.session_state:
            new_signal = st.session_state["new_data"][bead_number]
            prediction = st.session_state["model"].predict([new_signal.mean()])  # Predict anomaly
            anomaly_score = st.session_state["model"].decision_function([new_signal.mean()])[0]  # Get anomaly score
            color = 'blue' if prediction == 1 else 'red'
            
            fig.add_trace(go.Scatter(
                y=new_signal,
                mode='lines',
                line=dict(color=color, width=2),
                name=f"New Data (Bead {bead_number})",
                hoverinfo='text',
                text=f"Bead: {bead_number}<br>Prediction: {'Normal' if prediction == 1 else 'Anomalous'}<br>Anomaly Score: {anomaly_score:.4f}"
            ))

        # Update layout
        fig.update_layout(
            title=f"Bead Number {bead_number}: Anomaly Detection",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )

        st.plotly_chart(fig)

st.success("Analysis complete!")
