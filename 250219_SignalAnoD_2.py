import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from scipy.fft import fft
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

def extract_time_freq_features(signal):
    n = len(signal)
    if n == 0 or np.all(np.isnan(signal)):
        return [0] * 9  
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    energy = np.sum(np.square(signal)) / n
    skewness = skew(signal, nan_policy='omit')
    kurt = kurtosis(signal, nan_policy='omit')
    fft_values = fft(signal)
    fft_magnitude = np.abs(fft_values)[:n // 2]
    spectral_energy = np.sum(fft_magnitude ** 2) / len(fft_magnitude) if len(fft_magnitude) > 0 else 0
    dominant_freq = np.argmax(fft_magnitude) if np.any(fft_magnitude > 0) else 0
    return [mean_val, std_val, min_val, max_val, energy, skewness, kurt, spectral_energy, dominant_freq]

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
            chosen_bead_data = []
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] in selected_beads:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    chosen_bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"], "start_index": entry["start_index"], "end_index": entry["end_index"]})
            st.session_state["chosen_bead_data"] = chosen_bead_data
            st.success("Beads selected successfully!")

if "chosen_bead_data" in st.session_state:
    if st.button("Run Isolation Forest"):
        with st.spinner("Running Isolation Forest..."):
            st.session_state["show_scatter"] = False
            feature_matrix = []
            bead_labels = []
            anomaly_labels = []
            
            for bead_data in st.session_state["chosen_bead_data"]:
                signal = bead_data["data"].iloc[:, 0].values
                features = extract_time_freq_features(signal)
                feature_matrix.append(features)
                bead_labels.append(bead_data["bead_number"])
            
            feature_matrix = np.array(feature_matrix)
            iso_forest = IsolationForest(contamination=0.2, random_state=42)
            predictions = iso_forest.fit_predict(feature_matrix)
            anomaly_labels = predictions
            st.session_state["anomaly_labels"] = anomaly_labels
            st.session_state["feature_matrix"] = feature_matrix
            st.session_state["bead_labels"] = bead_labels
            st.success("Anomaly detection complete!")
            
            fig = go.Figure()
            for bead_data, anomaly in zip(st.session_state["chosen_bead_data"], anomaly_labels):
                signal = bead_data["data"].iloc[:, 0].values
                file_name = bead_data["file"]
                anomaly_score = iso_forest.decision_function([extract_time_freq_features(signal)])[0]
                status = "Anomalous" if anomaly == -1 else "Normal"
                fig.add_trace(go.Scatter(y=signal, mode='lines', name=f'Bead {bead_data["bead_number"]}', line=dict(color='red' if anomaly == -1 else 'black'), text=f"File: {file_name}<br>Status: {status}<br>Anomaly Score: {anomaly_score:.4f}"))
            st.plotly_chart(fig)

if "feature_matrix" in st.session_state and st.button("Show 3D Scatter Plot"):
    st.session_state["show_scatter"] = True
    pca = PCA(n_components=3)
    transformed_features = pca.fit_transform(st.session_state["feature_matrix"])
    df_pca = pd.DataFrame(transformed_features, columns=["PC1", "PC2", "PC3"])
    df_pca["File"] = [bead_data["file"] for bead_data in st.session_state["chosen_bead_data"]]
    df_pca["Bead"] = st.session_state["bead_labels"]
    df_pca["Anomaly"] = st.session_state["anomaly_labels"]
    df_pca["Color"] = df_pca["Anomaly"].apply(lambda x: "red" if x == -1 else "black")
    fig = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", color=df_pca["Color"], hover_data=df_pca.columns)
    st.plotly_chart(fig)
