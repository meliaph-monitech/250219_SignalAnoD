import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import Model
from keras.layers import Input, Dense
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

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

# **Feature Extraction (Unchanged)**
def extract_advanced_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 20  # Default feature values

    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return [0] * 20  # Return default values if data is bad

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
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    rms = np.sqrt(np.mean(signal**2))

    x = np.arange(n)
    if len(set(signal)) == 1 or len(signal) < 2:  # Constant or too short signal
        slope = 0
    else:
        try:
            slope, _ = np.polyfit(x, signal, 1)
        except np.linalg.LinAlgError:
            slope = 0

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
            spectral_entropy, autocorrelation, rms, slope, moving_average, outlier_count, extreme_event_duration]

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
st.title("Laser Welding Anomaly Detection V11 with Autoencoder")

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

# Feature Names
feature_names = ["Mean Value", "STD Value", "Min Value", "Max Value", "Median Value", "Skewness", "Kurtosis", "Peak-to-Peak", "Energy", "Coefficient of Variation (CV)",
                 "Spectral Entropy", "Autocorrelation", "Root Mean Square (RMS)", "Slope", "Moving Average",
                 "Outlier Count", "Extreme Event Duration"]
options = ["All"] + feature_names
selected_features = st.multiselect(
    "Select features to use for Autoencoder",
    options=options,
    default="All"
)

if "All" in selected_features and len(selected_features) > 1:
    selected_features = ["All"]
elif "All" not in selected_features and len(selected_features) == 0:
    st.error("You must select at least one feature.")
    st.stop()
if "All" in selected_features:
    selected_features = feature_names
selected_indices = [feature_names.index(f) for f in selected_features]

if st.button("Run Autoencoder") and "chosen_bead_data" in st.session_state:
    with st.spinner("Running Autoencoder Anomaly Detection..."):
        bead_data = [seg["data"].iloc[:, 0].values for seg in st.session_state["chosen_bead_data"]]
        feature_matrix = np.array([extract_advanced_features(signal) for signal in bead_data])
        feature_matrix = feature_matrix[:, selected_indices]  # Select only the chosen features
        scaler = MinMaxScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

        autoencoder = train_autoencoder(feature_matrix)
        reconstruction_error = np.mean(np.square(feature_matrix - autoencoder.predict(feature_matrix)), axis=1)
        threshold = np.percentile(reconstruction_error, 95)  # 95th percentile as threshold
        anomalies = reconstruction_error > threshold

        st.session_state["anomaly_results"] = [{"Bead Number": seg["bead_number"], 
                                                "File": seg["file"], 
                                                "start_index": seg["start_index"],
                                                "end_index": seg["end_index"],
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
