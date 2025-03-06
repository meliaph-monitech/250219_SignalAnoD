import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
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


def compute_z_score_features(signal):
    """
    Z-score feature extraction.
    Computes the mean absolute Z-score for a given signal.
    """
    if len(signal) == 0:
        return 0  # Return 0 if the signal is empty

    mean = np.mean(signal)
    std = np.std(signal)

    # Avoid division by zero
    if std == 0:
        return 0

    z_scores = np.abs((signal - mean) / std)
    aggregated_z_score = np.mean(z_scores)  # Aggregate Z-scores using mean
    return aggregated_z_score


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
        use_contamination_rate = st.checkbox("Use Contamination Rate", value=True)

        if st.button("Run Isolation Forest") and "metadata" in st.session_state:
            with st.spinner("Running Isolation Forest..."):
                features_by_bead = defaultdict(list)
                files_by_bead = defaultdict(list)

                # Group features by bead number
                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1, 0].values
                    z_score_feature = compute_z_score_features(bead_segment)  # Replace advanced features with Z-score
                    bead_number = entry["bead_number"]
                    features_by_bead[bead_number].append([z_score_feature])  # Wrap in a list for compatibility
                    files_by_bead[bead_number].append((entry["file"], bead_number))

                # Combine all features into a single matrix
                all_features = []
                all_file_names = []
                for bead_number, feature_matrix in features_by_bead.items():
                    all_features.extend(feature_matrix)
                    all_file_names.extend(files_by_bead[bead_number])

                # Normalize all features
                scaler = MinMaxScaler()
                all_scaled_features = scaler.fit_transform(all_features)

                # Train Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination_rate if use_contamination_rate else "auto",
                    random_state=42
                )
                predictions = iso_forest.fit_predict(all_scaled_features)
                anomaly_scores = -iso_forest.decision_function(all_scaled_features)

                # Save results
                st.session_state["anomaly_results_isoforest"] = {
                    fn: ("anomalous" if p == -1 else "normal") for fn, p in zip(all_file_names, predictions)
                }
                st.session_state["anomaly_scores_isoforest"] = {
                    fn: s for fn, s in zip(all_file_names, anomaly_scores)
                }

st.write("## Visualization")
if "anomaly_results_isoforest" in st.session_state:
    bead_numbers = sorted(set(num for _, num in st.session_state["anomaly_results_isoforest"].keys()))
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
            status = st.session_state["anomaly_results_isoforest"].get((file_name, selected_bead), "normal")
            anomaly_score = st.session_state["anomaly_scores_isoforest"].get((file_name, selected_bead), 0)

            # Set color based on anomaly status
            color = "red" if status == "anomalous" else "black"

            fig.add_trace(go.Scatter(
                y=signal,
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
