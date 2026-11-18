"""
AI Spacecraft Anomaly Detection System (Deep Learning + Advanced Models)
Dataset: NASA CMAPSS / Turbofan Engine Degradation Dataset

What this script does:
1. Loads train/test/RUL files
2. Cleans and preprocesses the data
3. Builds sequence windows for time-series learning
4. Trains three advanced deep learning models:
   - BiLSTM
   - CNN-BiGRU
   - Transformer Encoder
5. Evaluates them on the test set
6. Predicts Remaining Useful Life (RUL)
7. Creates anomaly/risk labels
8. Saves trained models and plots

Files required in the same folder:
- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

Run:
python deep_learning_spacecraft_project.py
"""

import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    GRU,
    Bidirectional,
    Conv1D,
    BatchNormalization,
    LayerNormalization,
    GlobalAveragePooling1D,
    Add,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# =========================
# 1. SETTINGS
# =========================
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"

OUTPUT_DIR = "outputs_dl_spacecraft"
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
RISK_THRESHOLD = 20   # predicted RUL < 20 => high-risk / anomaly

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =========================
# 2. REPRODUCIBILITY
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# =========================
# 3. LOAD DATA
# =========================
def load_cmapss(train_path: str, test_path: str, rul_path: str):
    columns = [
        "unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)

    train_df = train_df.dropna(axis=1, how="all")
    test_df = test_df.dropna(axis=1, how="all")

    train_df.columns = columns
    test_df.columns = columns
    rul_df.columns = ["RUL"]

    return train_df, test_df, rul_df


# =========================
# 4. RUL CREATION
# =========================
def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycle.columns = ["unit_number", "max_cycle"]
    train_df = train_df.merge(max_cycle, on="unit_number")
    train_df["RUL"] = train_df["max_cycle"] - train_df["time_in_cycles"]
    train_df.drop(columns=["max_cycle"], inplace=True)
    return train_df


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    test_max = test_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    test_max.columns = ["unit_number", "max_cycle"]

    rul_df = rul_df.copy()
    rul_df["unit_number"] = rul_df.index + 1

    test_max = test_max.merge(rul_df, on="unit_number")
    test_max["final_cycle"] = test_max["max_cycle"] + test_max["RUL"]

    test_df = test_df.merge(test_max[["unit_number", "final_cycle"]], on="unit_number")
    test_df["RUL"] = test_df["final_cycle"] - test_df["time_in_cycles"]
    test_df.drop(columns=["final_cycle"], inplace=True)
    return test_df


# =========================
# 5. PREPROCESSING
# =========================
def get_feature_columns(df: pd.DataFrame):
    # Remove identifier/target columns
    excluded = {"unit_number", "time_in_cycles", "RUL"}
    all_features = [c for c in df.columns if c not in excluded]

    # Remove constant columns automatically
    useful_features = []
    for col in all_features:
        if df[col].nunique() > 1:
            useful_features.append(col)

    return useful_features


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


# =========================
# 6. WINDOW CREATION
# =========================
def create_train_sequences(df: pd.DataFrame, feature_cols, seq_len: int):
    X, y = [], []
    for unit in df["unit_number"].unique():
        unit_df = df[df["unit_number"] == unit].reset_index(drop=True)
        feature_values = unit_df[feature_cols].values
        rul_values = unit_df["RUL"].values

        for i in range(len(unit_df) - seq_len + 1):
            X.append(feature_values[i:i + seq_len])
            y.append(rul_values[i + seq_len - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_test_sequences_last_window(df: pd.DataFrame, feature_cols, seq_len: int):
    X, y, engine_ids = [], [], []
    for unit in df["unit_number"].unique():
        unit_df = df[df["unit_number"] == unit].reset_index(drop=True)
        feature_values = unit_df[feature_cols].values
        rul_values = unit_df["RUL"].values

        if len(unit_df) >= seq_len:
            X.append(feature_values[-seq_len:])
            y.append(rul_values[-1])
            engine_ids.append(unit)
        else:
            # Pad short sequences at the front with zeros
            pad_len = seq_len - len(unit_df)
            padded = np.vstack([np.zeros((pad_len, len(feature_cols))), feature_values])
            X.append(padded)
            y.append(rul_values[-1])
            engine_ids.append(unit)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(engine_ids)


# =========================
# 7. MODEL BUILDERS
# =========================
def build_bilstm(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, name="rul")(x)
    model = Model(inputs, outputs, name="BiLSTM")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def build_cnn_bigru(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(32))(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, name="rul")(x)
    model = Model(inputs, outputs, name="CNN_BiGRU")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def transformer_encoder_block(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(x.shape[-1])(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dense(64)(x)
    x = transformer_encoder_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, name="rul")(x)
    model = Model(inputs, outputs, name="Transformer_Encoder")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


# =========================
# 8. TRAINING FUNCTION
# =========================
def train_model(model, X_train, y_train, X_val, y_val, model_name):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history


# =========================
# 9. EVALUATION
# =========================
def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def classify_risk(rul_pred, threshold=20):
    return np.where(rul_pred < threshold, 1, 0)


def save_training_plot(history, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Training Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_training_curve.png"), dpi=200)
    plt.close()


def save_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual RUL", marker="o")
    plt.plot(y_pred, label="Predicted RUL", marker="o")
    plt.title(f"Actual vs Predicted RUL - {model_name}")
    plt.xlabel("Test Engine")
    plt.ylabel("RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_actual_vs_predicted.png"), dpi=200)
    plt.close()


# =========================
# 10. MAIN PIPELINE
# =========================
def main():
    print("Loading dataset...")
    train_df, test_df, rul_df = load_cmapss(TRAIN_FILE, TEST_FILE, RUL_FILE)

    print("Creating RUL columns...")
    train_df = add_train_rul(train_df)
    test_df = add_test_rul(test_df, rul_df)

    feature_cols = get_feature_columns(train_df)
    print(f"Using {len(feature_cols)} features")

    print("Scaling features...")
    train_df, test_df, scaler = scale_features(train_df, test_df, feature_cols)

    print("Building sequence windows...")
    X_all, y_all = create_train_sequences(train_df, feature_cols, SEQUENCE_LENGTH)
    X_test, y_test, test_engine_ids = create_test_sequences_last_window(test_df, feature_cols, SEQUENCE_LENGTH)

    print(f"Train sequences: {X_all.shape}")
    print(f"Test sequences : {X_test.shape}")

    # Time-aware split by index ratio
    split_idx = int(len(X_all) * 0.8)
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    models = {
        "BiLSTM": build_bilstm(input_shape),
        "CNN_BiGRU": build_cnn_bigru(input_shape),
        "Transformer_Encoder": build_transformer(input_shape),
    }

    all_results = []
    engine_predictions = pd.DataFrame({"unit_number": test_engine_ids, "actual_RUL": y_test})

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        history = train_model(model, X_train, y_train, X_val, y_val, model_name)
        save_training_plot(history, model_name)

        print(f"Evaluating {model_name}...")
        pred = model.predict(X_test, verbose=0).reshape(-1)

        rmse, mae, r2 = regression_metrics(y_test, pred)
        risk_label = classify_risk(pred, threshold=RISK_THRESHOLD)

        all_results.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        engine_predictions[f"{model_name}_predicted_RUL"] = pred
        engine_predictions[f"{model_name}_anomaly"] = risk_label

        save_actual_vs_predicted(y_test, pred, model_name)
        print(f"{model_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    results_df = pd.DataFrame(all_results).sort_values(by="RMSE")
    print("\nFinal Model Comparison:")
    print(results_df)

    best_model = results_df.iloc[0]["Model"]
    print(f"\nBest Model: {best_model}")

    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    engine_predictions.to_csv(os.path.join(REPORTS_DIR, "engine_level_predictions.csv"), index=False)

    summary = {
        "sequence_length": SEQUENCE_LENGTH,
        "features_used": feature_cols,
        "best_model": best_model,
        "risk_threshold_rul": RISK_THRESHOLD,
        "results": all_results,
    }
    with open(os.path.join(REPORTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot model comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.title("Model Comparison by RMSE")
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_rmse.png"), dpi=200)
    plt.close()

    print("\nSaved outputs:")
    print(f"- Models:  {MODELS_DIR}")
    print(f"- Plots:   {PLOTS_DIR}")
    print(f"- Reports: {REPORTS_DIR}")
    print("\nProject completed successfully.")


# =========================
# 11. SAFE ENTRY POINT
# =========================
if __name__ == "__main__":
    # Quick file check
    required_files = [TRAIN_FILE, TEST_FILE, RUL_FILE]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("Missing required files:")
        for f in missing:
            print(" -", f)
        print("\nPlace the NASA CMAPSS files in the same folder and run again.")
    else:
        main()
