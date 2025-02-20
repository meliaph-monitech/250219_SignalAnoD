import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sklearn.ensemble import IsolationForest

# Function to extract ZIP structure
def extract_zip_structure(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_structure = {"root": []}
        for file in zip_ref.namelist():
            parts = file.split('/')
            if len(parts) > 1:
                folder, filename = parts[0], parts[-1]
                if folder not in file_structure:
                    file_structure[folder] = []
                if filename.endswith('.csv'):
                    file_structure[folder].append(filename)
            else:
                if file.endswith('.csv'):
                    file_structure["root"].append(file)
    return file_structure

# Feature extraction function
def extract_advanced_features(signal):
    features = {}
    n = len(signal)

    # Time domain features
    features["mean"] = np.mean(signal)
    features["std_dev"] = np.std(signal)
    features["min"] = np.min(signal)
    features["max"] = np.max(signal)
    features["energy"] = np.sum(np.square(signal)) / n
    features["skewness"] = skew(signal)
    features["kurt"] = kurtosis(signal)

    # Frequency domain features using FFT
    fft_values = fft(signal)
    fft_magnitude = np.abs(fft_values)[:n // 2]  # Take half spectrum (positive frequencies)
    features["spectral_energy"] = np.sum(fft_magnitude ** 2) / len(fft_magnitude)
    features["dominant_freq"] = np.argmax(fft_magnitude)  # Index of the dominant frequency

    return features

# Streamlit UI Setup
st.title("Laser Welding Data Visualization & Anomaly Detection")

# Upload ZIP File
uploaded_file = st.file_uploader("Upload ZIP file", type=["zip"])

if uploaded_file:
    zip_path = "temp.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract ZIP Structure
    file_structure = extract_zip_structure(zip_path)
    folders = list(file_structure.keys())
    
    # Folder Selection
    selected_folder = st.selectbox("Select Folder", folders) if len(folders) > 1 else folders[0]
    
    # CSV File Selection
    csv_files = file_structure[selected_folder]
    selected_csv = st.selectbox("Select CSV File", csv_files)
    
    # Load Data Immediately
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(f"{selected_folder}/{selected_csv}" if selected_folder != "root" else selected_csv) as f:
            df = pd.read_csv(f)
            st.session_state.df = df
    
    # See Raw Data Button
    if st.button("See Raw Data"):
        # Plot Raw Data
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        for i, col in enumerate(df.columns[:3]):
            axes[i].plot(df[col])
            axes[i].set_title(col)
        st.pyplot(fig)
    
    # Bead Segmentation Inputs
    if "df" in st.session_state:
        df = st.session_state.df
        filter_column = st.selectbox("Select Filter Column", df.columns[:3])
        filter_threshold = st.number_input("Set Filter Threshold", value=0.0)
    
        if st.button("Start Bead Segmentation"):
            filter_values = df[filter_column].astype(float)
            start_points, end_points = [], []
            i = 0
            while i < len(filter_values):
                if filter_values[i] > filter_threshold:
                    start = i
                    while i < len(filter_values) and filter_values[i] > filter_threshold:
                        i += 1
                    end = i - 1
                    start_points.append(start)
                    end_points.append(end)
                else:
                    i += 1
            bead_counts = [end - start + 1 for start, end in zip(start_points, end_points) if (end - start + 1) >= 10]
            st.session_state.bead_segments = {selected_csv: list(zip(start_points, end_points))}
            
            # Heatmap Visualization
            heatmap_data = pd.DataFrame(bead_counts, columns=["Bead Count"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap_data.T, cmap="jet", cbar=True, xticklabels=False)
            st.pyplot(fig)
    
    # Anomaly Detection
    if "df" in st.session_state and "bead_segments" in st.session_state:
        st.subheader("Anomaly Detection")
        selected_column = st.selectbox("Select Column for Anomaly Detection", df.columns)
        selected_bead_number = st.selectbox("Select Bead Number", list(range(len(st.session_state.bead_segments[selected_csv]))))
    
        if st.button("Start Detection"):
            bead_features = []
            bead_names = []
            raw_signals = {}
            
            for csv_file in csv_files:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    with zip_ref.open(f"{selected_folder}/{csv_file}" if selected_folder != "root" else csv_file) as f:
                        df = pd.read_csv(f)
                        if csv_file in st.session_state.bead_segments:
                            start, end = st.session_state.bead_segments[csv_file][selected_bead_number]
                            bead_signal = df[selected_column][start:end].values
                            raw_signals[csv_file] = bead_signal
                            if len(bead_signal) > 0:
                                features = extract_advanced_features(bead_signal)
                                bead_features.append(list(features.values()))
                                bead_names.append(csv_file)
            
            # Convert to DataFrame
            if bead_features:
                feature_df = pd.DataFrame(bead_features, columns=features.keys(), index=bead_names)
                
                # Apply Isolation Forest
                model = IsolationForest()
                predictions = model.fit_predict(feature_df)
                feature_df['Anomaly'] = predictions
                
                # Visualize Raw Signals
                fig, ax = plt.subplots(figsize=(10, 6))
                for name, signal in raw_signals.items():
                    color = 'red' if feature_df.loc[name, 'Anomaly'] == -1 else 'blue'
                    ax.plot(signal, color=color, alpha=0.7)
                ax.set_title("Raw Signals with Anomaly Detection")
                st.pyplot(fig)
            else:
                st.warning("No valid bead data available for anomaly detection.")
