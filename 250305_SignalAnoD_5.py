import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import numpy as np

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return csv_files

def extract_features(df, column):
    signal = df[column].values
    return [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal), np.median(signal),
        np.max(signal) - np.min(signal), np.sum(signal**2), np.sqrt(np.mean(signal**2))
    ]

def train_model(training_data):
    models = {}
    for bead_number, features_list in training_data.items():
        X_train = np.array(features_list)
        scaler = RobustScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_train_scaled)
        models[bead_number] = (model, scaler)
    return models

st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        if st.button("Train Model"):
            training_data = {}
            for file in csv_files:
                df = pd.read_csv(file)
                for bead_number in df['bead_number'].unique():
                    bead_df = df[df['bead_number'] == bead_number]
                    features = extract_features(bead_df, filter_column)
                    training_data.setdefault(bead_number, []).append(features)
            st.session_state["models"] = train_model(training_data)
            st.success("Model training complete!")

if "models" in st.session_state:
    new_file = st.file_uploader("Upload a new CSV file for analysis", type=["csv"])
    if new_file:
        new_df = pd.read_csv(new_file)
        predictions = {}
        
        for bead_number in new_df['bead_number'].unique():
            if bead_number in st.session_state["models"]:
                model, scaler = st.session_state["models"][bead_number]
                bead_df = new_df[new_df['bead_number'] == bead_number]
                features = extract_features(bead_df, filter_column)
                features_scaled = scaler.transform([features])
                pred = model.predict(features_scaled)[0]
                predictions[bead_number] = "Normal" if pred == 1 else "Anomalous"
        
        st.session_state["predictions"] = predictions
        st.success("Analysis complete!")

st.write("## Visualization")
if "models" in st.session_state and "predictions" in st.session_state:
    for bead_number, status in st.session_state["predictions"].items():
        fig = go.Figure()
        
        # Plot training data (black lines)
        for file in csv_files:
            df = pd.read_csv(file)
            bead_df = df[df['bead_number'] == bead_number]
            if not bead_df.empty:
                fig.add_trace(go.Scatter(
                    y=bead_df[filter_column], mode='lines', line=dict(color='black', width=1),
                    name=f"Training: {file}"
                ))
        
        # Plot new data (colored by anomaly result)
        new_bead_df = new_df[new_df['bead_number'] == bead_number]
        color = 'blue' if status == "Normal" else 'red'
        fig.add_trace(go.Scatter(
            y=new_bead_df[filter_column], mode='lines', line=dict(color=color, width=2),
            name=f"New Data (Bead {bead_number})"
        ))
        
        fig.update_layout(
            title=f"Bead Number {bead_number}: Anomaly Detection",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
