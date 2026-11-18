"""Spacecraft RUL Prediction Dashboard

A Streamlit-based web dashboard for the NASA CMAPSS RUL prediction project.

Features:
- View dataset summaries and model comparison results
- Visualize saved training curves and predictions
- Run the full training pipeline from the UI
- Inspect engine-level predictions and risk labels
- Real-time RUL predictions
- Advanced model interpretability
- Predictive analytics dashboard
- Ensemble analysis and optimization
- Data augmentation strategies
- Export and reporting
- Custom UI components

Run:
    streamlit run streamlit_app.py

"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# Import new advanced modules
try:
    from real_time_predictor import show_real_time_predictor
except ImportError:
    show_real_time_predictor = None

try:
    from model_interpreter import show_model_interpreter
except ImportError:
    show_model_interpreter = None

try:
    from predictive_dashboard import show_predictive_dashboard
except ImportError:
    show_predictive_dashboard = None

try:
    from ensemble_analyzer import show_ensemble_analyzer
except ImportError:
    show_ensemble_analyzer = None

try:
    from data_augmentation import show_data_augmentation
except ImportError:
    show_data_augmentation = None

try:
    from report_generator import show_report_generator
except ImportError:
    show_report_generator = None

try:
    from ui_components import apply_custom_styling, create_spacecraft_header
except ImportError:
    apply_custom_styling = None
    create_spacecraft_header = None

# Paths and globals
WORKSPACE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE_DIR / "outputs_dl_spacecraft"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "saved_models"

MODEL_NAMES = ["BiLSTM", "CNN_BiGRU", "Transformer_Encoder"]


@st.cache_data(show_spinner=False)
def load_summary():
    summary_path = REPORTS_DIR / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_comparison():
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_engine_predictions():
    path = REPORTS_DIR / "engine_level_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_plot(path: Path):
    if not path.exists():
        return None
    return path


def list_available_models():
    return [p.stem for p in MODELS_DIR.glob("*.keras")]


def run_training_command():
    """Run the existing training script in a subprocess.

    This avoids importing TensorFlow inside the Streamlit app, which may not be
    available in the current Python environment.
    """

    script_path = WORKSPACE_DIR / "deep_learning_spacecraft_project.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Use the same Python executable that runs Streamlit.
    cmd = [os.sys.executable, str(script_path)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    stdout_lines = []
    stderr_lines = []

    for line in proc.stdout:
        stdout_lines.append(line)
        st.text(line.strip())

    proc.wait()
    stderr_lines.extend(proc.stderr.readlines())

    return proc.returncode, "".join(stdout_lines), "".join(stderr_lines)


def show_dashboard():
    st.title("🚀 Spacecraft RUL Prediction Dashboard")
    st.write(
        "This dashboard provides insights and controls for the NASA CMAPSS turbofan Remaining Useful Life (RUL) prediction project."
    )

    summary = load_summary()
    if summary is None:
        st.warning(
            "No output reports found yet. Run the training pipeline using the sidebar button below to generate models and reports."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Project Summary")
        st.write(
            {
                "Sequence length": summary.get("sequence_length"),
                "Features used": len(summary.get("features_used", [])),
                "Risk threshold (RUL)": summary.get("risk_threshold_rul"),
            }
        )

    with col2:
        st.subheader("Best Model")
        st.metric("Best model", summary.get("best_model", "-"))

    st.markdown("---")

    st.subheader("Model Comparison")
    comp = load_model_comparison()
    if comp is not None:
        st.dataframe(comp.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R2": "{:.3f}"}))
        st.bar_chart(comp.set_index("Model")["RMSE"])

    st.subheader("Available Outputs")
    st.write("**Saved plots:**")
    plot_files = sorted(PLOTS_DIR.glob("*.png"))
    for plot in plot_files:
        st.markdown(f"- `{plot.name}`")

    st.write("**Saved models:**")
    model_files = sorted(MODELS_DIR.glob("*.keras"))
    for model in model_files:
        st.markdown(f"- `{model.name}`")


def show_plots():
    st.header("📈 Plots & Visualizations")
    plot_files = sorted(PLOTS_DIR.glob("*.png"))

    if not plot_files:
        st.warning("No plots found. Run training to generate plots.")
        return

    selected = st.selectbox(
        "Select a plot", [p.name for p in plot_files], index=0
    )
    st.image(str(PLOTS_DIR / selected), use_column_width=True)


def show_predictions():
    st.header("🧠 Engine-Level Predictions")

    eng_df = load_engine_predictions()
    if eng_df is None:
        st.warning("No engine-level predictions found. Run training to generate predictions.")
        return

    st.markdown(
        "Use this table to explore predicted Remaining Useful Life (RUL) and the associated risk label for each test engine."
    )

    model_choice = st.selectbox("Which model predictions to view?", MODEL_NAMES)
    pred_col = f"{model_choice}_predicted_RUL"
    anomaly_col = f"{model_choice}_anomaly"

    if pred_col not in eng_df.columns:
        st.warning(f"Predictions for '{model_choice}' not found in the results.")
        return

    show_cols = ["unit_number", "actual_RUL", pred_col, anomaly_col]
    df_show = eng_df[show_cols].rename(
        columns={pred_col: "predicted_RUL", anomaly_col: "risk_label"}
    )

    st.dataframe(df_show)

    st.markdown("---")
    st.subheader("Risk Summary")
    risk_counts = df_show["risk_label"].value_counts().rename(
        index={0: "Normal (RUL >= threshold)", 1: "High-risk (RUL < threshold)"}
    )
    st.bar_chart(risk_counts)


def show_data_inspector():
    st.header("📊 Data Explorer")

    st.write("This section shows the raw dataset tables used during training.")

    dataset = st.selectbox("Dataset version", ["FD001", "FD002", "FD003", "FD004"], index=0)
    st.write("Upload `train`, `test`, `RUL` files to override local files; otherwise local files are used.")

    train_upload = st.file_uploader(f"Upload train_{dataset}.txt", type=["txt"], key=f"train_{dataset}")
    test_upload = st.file_uploader(f"Upload test_{dataset}.txt", type=["txt"], key=f"test_{dataset}")
    rul_upload = st.file_uploader(f"Upload RUL_{dataset}.txt", type=["txt"], key=f"rul_{dataset}")

    def read_df(source):
        df = pd.read_csv(source, sep=r"\s+", header=None)
        return df.dropna(axis=1, how="all")

    if st.button("Load data"):
        try:
            if train_upload is not None and test_upload is not None and rul_upload is not None:
                train_df = read_df(train_upload)
                test_df = read_df(test_upload)
                rul_df = pd.read_csv(rul_upload, sep=r"\s+", header=None)
            else:
                data_prefix = f"{dataset}"
                train_path = WORKSPACE_DIR / f"train_{data_prefix}.txt"
                test_path = WORKSPACE_DIR / f"test_{data_prefix}.txt"
                rul_path = WORKSPACE_DIR / f"RUL_{data_prefix}.txt"

                if not train_path.exists() or not test_path.exists() or not rul_path.exists():
                    st.warning("Please upload all three files or ensure local files are present in the workspace.")
                    return

                train_df = read_df(train_path)
                test_df = read_df(test_path)
                rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)

            cols = [
                "unit_number",
                "time_in_cycles",
                "op_setting_1",
                "op_setting_2",
                "op_setting_3",
            ] + [f"sensor_{i}" for i in range(1, 22)]

            train_df.columns = cols[: train_df.shape[1]]
            test_df.columns = cols[: test_df.shape[1]]
            rul_df.columns = ["RUL"]

            st.subheader("Train dataset (first 5 rows)")
            st.dataframe(train_df.head())
            st.subheader("Test dataset (first 5 rows)")
            st.dataframe(test_df.head())
            st.subheader("RUL labels (first 5 rows)")
            st.dataframe(rul_df.head())

            st.success(f"{dataset} data loaded successfully.")
        except Exception as e:
            st.error(f"Error loading data: {e}")


def show_training():
    st.header("⚙️ Train / Retrain Models")
    st.write(
        "Start the full training pipeline (this may take several minutes depending on your hardware). "
        "Training will save models, plots, and reports under the outputs folder."
    )

    st.write(
        "This app does not require TensorFlow (which may not be available for your Python version). "
        "If you want to train models, run `python deep_learning_spacecraft_project.py` in a terminal."
    )

    if st.button("Run training script now"):
        with st.spinner("Running training script..."):
            return_code, stdout, stderr = run_training_command()
            if return_code == 0:
                st.success("Training script finished successfully.")
            else:
                st.error(f"Training script exited with code {return_code}.")

            with st.expander("Stdout"):
                st.code(stdout)
            with st.expander("Stderr"):
                st.code(stderr)


def main():
    st.set_page_config(
        page_title="Spacecraft RUL Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom styling
    if apply_custom_styling:
        apply_custom_styling()

    st.sidebar.title("Navigation")
    
    # Core pages
    core_pages = [
        "Dashboard",
        "Plots",
        "Predictions",
        "Data Explorer",
        "Train / Retrain",
    ]
    
    # Advanced features pages
    advanced_pages = []
    if show_real_time_predictor:
        advanced_pages.append("⚡ Real-Time Prediction")
    if show_model_interpreter:
        advanced_pages.append("🧠 Model Interpreter")
    if show_predictive_dashboard:
        advanced_pages.append("📊 Analytics Dashboard")
    if show_ensemble_analyzer:
        advanced_pages.append("🤝 Ensemble Analysis")
    if show_data_augmentation:
        advanced_pages.append("📈 Data Augmentation")
    if show_report_generator:
        advanced_pages.append("📄 Export & Reports")
    
    all_pages = core_pages + advanced_pages
    
    page = st.sidebar.radio(
        "Select view",
        all_pages,
    )

    # Core functionality
    if page == "Dashboard":
        show_dashboard()
    elif page == "Plots":
        show_plots()
    elif page == "Predictions":
        show_predictions()
    elif page == "Data Explorer":
        show_data_inspector()
    elif page == "Train / Retrain":
        show_training()
    
    # Advanced features
    elif page == "⚡ Real-Time Prediction" and show_real_time_predictor:
        show_real_time_predictor()
    elif page == "🧠 Model Interpreter" and show_model_interpreter:
        show_model_interpreter()
    elif page == "📊 Analytics Dashboard" and show_predictive_dashboard:
        show_predictive_dashboard()
    elif page == "🤝 Ensemble Analysis" and show_ensemble_analyzer:
        show_ensemble_analyzer()
    elif page == "📈 Data Augmentation" and show_data_augmentation:
        show_data_augmentation()
    elif page == "📄 Export & Reports" and show_report_generator:
        show_report_generator()


if __name__ == "__main__":
    main()
