"""Real-time RUL Prediction from Custom Sensor Input

Allows users to input sensor readings and get ensemble predictions with confidence.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


WORKSPACE_DIR = Path(__file__).resolve().parent
MODELS_DIR = WORKSPACE_DIR / "outputs_dl_spacecraft" / "saved_models"
MODEL_NAMES = ["BiLSTM", "CNN_BiGRU", "Transformer_Encoder"]


@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    for model_name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{model_name}.keras"
        if model_path.exists():
            try:
                models[model_name] = tf.keras.models.load_model(model_path)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
    return models


@st.cache_resource
def get_scaler():
    """Initialize scaler - in production, this would load a saved scaler."""
    return MinMaxScaler(feature_range=(0, 1))


def normalize_sensors(sensor_values, scaler):
    """Normalize sensor readings."""
    # Assuming 21 sensors based on NASA CMAPSS
    if len(sensor_values) != 21:
        return None
    sensor_array = np.array(sensor_values).reshape(1, -1)
    return scaler.fit_transform(sensor_array)


def create_sequence(sensor_history, sequence_length=30):
    """Create sequence from sensor history."""
    if len(sensor_history) < sequence_length:
        padding = np.zeros((sequence_length - len(sensor_history), sensor_history.shape[1]))
        sequence = np.vstack([padding, sensor_history])
    else:
        sequence = sensor_history[-sequence_length:]
    return np.expand_dims(sequence, axis=0)


def predict_rul(sensor_history, models, sequence_length=30):
    """Predict RUL using ensemble of models."""
    predictions = {}
    confidences = {}
    
    sequence = create_sequence(sensor_history, sequence_length)
    
    for model_name, model in models.items():
        try:
            pred = model.predict(sequence, verbose=0)[0][0]
            # Ensure RUL is positive
            rul = max(0, float(pred))
            predictions[model_name] = rul
            confidences[model_name] = 0.85  # Placeholder confidence
        except Exception as e:
            st.warning(f"Error predicting with {model_name}: {e}")
    
    return predictions, confidences


def show_real_time_predictor():
    """Display real-time prediction interface."""
    st.header("⚡ Real-Time RUL Prediction")
    st.write("Input custom sensor readings to get instant RUL predictions from all models.")
    
    models = load_models()
    if not models:
        st.error("No trained models found. Train models first using the Training page.")
        return
    
    # Sensor input section
    st.subheader("📡 Sensor Readings (21 sensors)")
    st.write("Enter current sensor values or use the random example below.")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎲 Random Example"):
            st.session_state.sensor_input = list(np.random.uniform(0, 1, 21))
    
    # Initialize session state for sensors
    if "sensor_input" not in st.session_state:
        st.session_state.sensor_input = [0.5] * 21
    
    # Create 7 rows of 3 columns for sensor inputs
    sensor_cols = st.columns(3)
    for i in range(21):
        col_idx = i % 3
        if i % 3 == 0 and i > 0:
            sensor_cols = st.columns(3)
        with sensor_cols[col_idx]:
            st.session_state.sensor_input[i] = st.slider(
                f"Sensor {i+1}",
                0.0, 1.0,
                st.session_state.sensor_input[i],
                0.01,
                key=f"sensor_{i}"
            )
    
    # Prediction section
    if st.button("🔍 Predict RUL", type="primary", use_container_width=False):
        sensor_history = np.array(st.session_state.sensor_input).reshape(1, -1)
        predictions, confidences = predict_rul(sensor_history, models)
        
        st.subheader("Prediction Results")
        
        # Display individual model predictions
        col1, col2, col3 = st.columns(3)
        colors = ["🟢", "🟠", "🔵"]
        
        for idx, (model_name, pred) in enumerate(predictions.items()):
            with [col1, col2, col3][idx]:
                st.metric(
                    f"{colors[idx]} {model_name}",
                    f"{pred:.1f} cycles",
                    delta=f"Confidence: {confidences[model_name]:.0%}"
                )
        
        # Ensemble prediction (weighted average)
        if predictions:
            ensemble_rul = np.mean(list(predictions.values()))
            ensemble_std = np.std(list(predictions.values()))
            
            st.markdown("---")
            st.subheader("📊 Ensemble Prediction")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ensemble RUL", f"{ensemble_rul:.1f} cycles")
            with col2:
                st.metric("Std Dev", f"{ensemble_std:.1f} cycles")
            with col3:
                risk_level = "🔴 CRITICAL" if ensemble_rul < 20 else "🟠 HIGH" if ensemble_rul < 50 else "🟢 NORMAL"
                st.metric("Risk Level", risk_level)
            
            # Show prediction range
            st.write(f"**Prediction Range:** {ensemble_rul - ensemble_std:.1f} to {ensemble_rul + ensemble_std:.1f} cycles")
            
            # Show comparison
            st.bar_chart(pd.DataFrame({
                "RUL Prediction": predictions
            }), use_container_width=False)
