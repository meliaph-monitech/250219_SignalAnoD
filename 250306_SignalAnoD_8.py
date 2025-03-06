import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
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


def compute_z_scores(bead_data, global_mean, global_std):
    """Computes Z-scores for a bead's signal relative to global statistics."""
    if global_std == 0:  # Avoid division by zero
        return np.zeros_like(bead_data)
    return (bead_data - global_mean) / global_std


st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection with Z-Score Analysis")

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

        if st.button("Run Z-Score Analysis") and "metadata" in st.session_state:
            with st.spinner("Running Z-Score Analysis..."):
                # Step 1: Organize segmented data by bead number
                bead_data_by_number = defaultdict(list)
                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1, 0].values  # Extract signal
                    bead_data_by_number[entry["bead_number"]].append(bead_segment)

                # Step 2: Compute global statistics (mean, std) for each bead number
                global_stats = {}
                for bead_number, signals in bead_data_by_number.items():
                    combined_signal = np.concatenate(signals)
                    global_mean = np.mean(combined_signal)
                    global_std = np.std(combined_signal)
                    global_stats[bead_number] = (global_mean, global_std)

                # Step 3: Compute Z-scores and aggregate results
                anomalies = defaultdict(dict)
                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1, 0].values  # Extract signal

                    bead_number = entry["bead_number"]
                    global_mean, global_std = global_stats[bead_number]
                    z_scores = compute_z_scores(bead_segment, global_mean, global_std)

                    # Aggregate Z-scores for the bead (mean absolute Z-score)
                    aggregated_z_score = np.mean(np.abs(z_scores))
                    anomalies[entry["file"]][bead_number] = {
                        "aggregated_z_score": aggregated_z_score,
                        "is_anomalous": aggregated_z_score > 3  # Flag as anomalous if Z > 3
                    }

                # Save results to session state
                st.session_state["anomaly_results_zscore"] = anomalies

st.write("## Visualization")
if "anomaly_results_zscore" in st.session_state:
    bead_numbers = sorted(set(num for file_results in st.session_state["anomaly_results_zscore"].values() for num in file_results.keys()))
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)

    if selected_bead:
        fig = go.Figure()

        # Filter data for the selected bead number
        selected_bead_data = [entry for entry in st.session_state["metadata"] if entry["bead_number"] == selected_bead]

        for bead_info in selected_bead_data:
            file_name = bead_info["file"]
            start_idx = bead_info["start_index"]
            end_idx = bead_info["end_index"]

            # Load data and extract the signal for the specific bead
            df = pd.read_csv(file_name)
            signal = df.iloc[start_idx:end_idx + 1, 0].values  # Extract only the bead's signal

            # Get anomaly status and score
            result = st.session_state["anomaly_results_zscore"][file_name].get(selected_bead, {})
            aggregated_z_score = result.get("aggregated_z_score", 0)
            is_anomalous = result.get("is_anomalous", False)

            # Set color based on anomaly status
            color = 'red' if is_anomalous else 'black'

            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                line=dict(color=color, width=1),
                name=f"{file_name}",
                hoverinfo='text',
                text=f"File: {file_name}<br>Aggregated Z-Score: {aggregated_z_score:.4f}<br>Anomalous: {is_anomalous}"
            ))

        fig.update_layout(
            title=f"Bead Number {selected_bead}: Z-Score Anomaly Detection Results",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )

        st.plotly_chart(fig)
