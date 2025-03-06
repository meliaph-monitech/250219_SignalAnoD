import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import skew, kurtosis
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


def extract_features(signal):
    """
    Extracts statistical and shape-based features from the signal.
    """
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "min": np.min(signal),
        "max": np.max(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "energy": np.sum(np.square(signal)),
        "peak_to_peak": np.ptp(signal)
    }
    return list(features.values())


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

        contamination_rate = st.slider("Set Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

        if st.button("Run Isolation Forest") and "metadata" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                features = []
                bead_identifiers = []

                # Extract features bead by bead
                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1, 0].values
                    
                    # Extract features
                    bead_features = extract_features(bead_segment)
                    features.append(bead_features)
                    bead_identifiers.append((entry["file"], entry["bead_number"]))

                # Scale features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # Train Isolation Forest
                iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
                predictions = iso_forest.fit_predict(features_scaled)
                anomaly_scores = -iso_forest.decision_function(features_scaled)

                # Save results
                st.session_state["anomaly_results"] = {
                    bead: ("anomalous" if pred == -1 else "normal") for bead, pred in zip(bead_identifiers, predictions)
                }
                st.session_state["anomaly_scores"] = {
                    bead: score for bead, score in zip(bead_identifiers, anomaly_scores)
                }

st.write("## Visualization")
if "anomaly_results" in st.session_state:
    # Safely extract bead numbers from the anomaly results
    bead_numbers = sorted(
        set(key[1] for key in st.session_state["anomaly_results"].keys() if isinstance(key, tuple) and len(key) == 2)
    )

    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)

    if selected_bead:
        fig = go.Figure()

        # Filter data for the selected bead number
        selected_bead_data = [entry for entry in st.session_state["metadata"] if entry["bead_number"] == selected_bead]

        for bead_info in selected_bead_data:
            file_name = bead_info["file"]
            start_idx = bead_info["start_index"]
            end_idx = bead_info["end_index"]

            # Load data and extract the raw signal for the specific bead
            df = pd.read_csv(file_name)
            raw_signal = df.iloc[start_idx:end_idx + 1, 0].values

            # Get anomaly status and score
            status = st.session_state["anomaly_results"].get((file_name, selected_bead), "normal")
            anomaly_score = st.session_state["anomaly_scores"].get((file_name, selected_bead), 0)

            # Set color based on anomaly status
            color = "red" if status == "anomalous" else "black"

            fig.add_trace(go.Scatter(
                y=raw_signal,
                mode="lines",
                line=dict(color=color, width=1),
                name=f"{file_name}",
                hoverinfo="text",
                text=f"File: {file_name}<br>Status: {status}<br>Anomaly Score: {anomaly_score:.4f}"
            ))

        fig.update_layout(
            title=f"Bead Number {selected_bead}: Anomaly Detection Results",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )

        st.plotly_chart(fig)
