import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense
import plotly.graph_objects as go

# **Autoencoder Model**
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# **Train Autoencoder**
def train_autoencoder(feature_matrix, epochs=50, batch_size=32):
    autoencoder = build_autoencoder(feature_matrix.shape[1])
    autoencoder.fit(feature_matrix, feature_matrix, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    return autoencoder

# **Feature Extraction**
def extract_advanced_features(signal):
    if len(signal) == 0:
        return [0] * 10  # Default features if signal is empty
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.ptp(signal),  # Peak-to-peak
        np.sum(signal**2),  # Energy
        np.sqrt(np.mean(signal**2)),  # RMS
        np.var(signal),  # Variance
        np.mean(np.abs(signal - np.mean(signal)))  # Mean absolute deviation
    ]

# **File Extraction**
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

# **Signal Segmentation**
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

# **Streamlit App**
st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection with Autoencoder")

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

if st.button("Run Autoencoder") and "chosen_bead_data" in st.session_state:
    with st.spinner("Running Autoencoder Anomaly Detection..."):
        bead_data = [seg["data"].iloc[:, 0].values for seg in st.session_state["chosen_bead_data"]]
        feature_matrix = np.array([extract_advanced_features(signal) for signal in bead_data])
        scaler = MinMaxScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

        autoencoder = train_autoencoder(feature_matrix)
        reconstruction_error = np.mean(np.square(feature_matrix - autoencoder.predict(feature_matrix)), axis=1)
        threshold = np.percentile(reconstruction_error, 95)  # 95th percentile as threshold
        anomalies = reconstruction_error > threshold

        st.session_state["anomaly_results"] = [{"Bead Number": seg["bead_number"], 
                                                "File": seg["file"], 
                                                "Status": "Anomalous" if anomaly else "Normal", 
                                                "Reconstruction Error": error} 
                                               for seg, anomaly, error in zip(st.session_state["chosen_bead_data"], anomalies, reconstruction_error)]
        st.success("Anomaly detection complete!")

if "anomaly_results" in st.session_state:
    st.write("### Anomaly Detection Results")
    result_df = pd.DataFrame(st.session_state["anomaly_results"])
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Anomaly Results",
        data=csv,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )

    # Visualization
    st.write("## Visualization")
    for bead_number in set(result_df["Bead Number"]):
        bead_results = result_df[result_df["Bead Number"] == bead_number]
        fig = go.Figure()
        for _, row in bead_results.iterrows():
            df = pd.read_csv(row["File"])
            signal = df.iloc[row["start_index"]:row["end_index"] + 1, 0].values
            color = 'red' if row["Status"] == "Anomalous" else 'black'
            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                line=dict(color=color, width=2),
                name=row["File"]
            ))
        fig.update_layout(
            title=f"Bead Number {bead_number}: Anomaly Detection Results",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
