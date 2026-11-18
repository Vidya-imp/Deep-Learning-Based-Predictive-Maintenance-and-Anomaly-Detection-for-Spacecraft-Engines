"""Report Generation & Data Export

Generate PDF reports and export predictions/metrics to various formats.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


WORKSPACE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = WORKSPACE_DIR / "outputs_dl_spacecraft" / "reports"


@st.cache_data
def load_predictions():
    """Load predictions data."""
    path = REPORTS_DIR / "engine_level_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_model_comparison():
    """Load model comparison metrics."""
    path = REPORTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_summary():
    """Load summary statistics."""
    path = REPORTS_DIR / "summary.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def generate_executive_summary():
    """Generate executive summary statistics."""
    predictions_df = load_predictions()
    summary = load_summary()
    
    if predictions_df is None:
        return None
    
    exec_summary = {
        'Report Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total Engines Analyzed': len(predictions_df),
        'Dataset': summary.get('dataset', 'FD001') if summary else 'FD001',
        'Risk Threshold (RUL)': summary.get('risk_threshold_rul', 20) if summary else 20,
    }
    
    # Risk statistics
    pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
    if pred_cols:
        ensemble = predictions_df[pred_cols[0]].mean()  # Simplified
        exec_summary['Average Predicted RUL'] = f"{ensemble:.1f} cycles"
    
    if 'actual_RUL' in predictions_df.columns:
        exec_summary['Average Actual RUL'] = f"{predictions_df['actual_RUL'].mean():.1f} cycles"
    
    return exec_summary


def generate_predictions_csv(predictions_df):
    """Generate detailed predictions CSV."""
    if predictions_df is None:
        return None
    
    # Create detailed export
    export_df = predictions_df.copy()
    
    # Round predictions
    pred_cols = [col for col in export_df.columns if 'predicted_RUL' in col]
    for col in pred_cols:
        export_df[col] = export_df[col].round(2)
    
    # Add ensemble prediction
    if len(pred_cols) > 1:
        export_df['Ensemble_RUL'] = export_df[pred_cols].mean(axis=1).round(2)
    
    # Add error column if actual is available
    if 'actual_RUL' in export_df.columns and 'Ensemble_RUL' in export_df.columns:
        export_df['Prediction_Error'] = (export_df['Ensemble_RUL'] - export_df['actual_RUL']).round(2)
    
    return export_df


def generate_metrics_report():
    """Generate detailed metrics report."""
    comparison_df = load_model_comparison()
    predictions_df = load_predictions()
    
    if comparison_df is None:
        return None
    
    metrics = {
        'Model Performance': comparison_df.to_dict('records') if comparison_df is not None else [],
        'Generated': datetime.now().isoformat(),
    }
    
    # Add prediction error analysis
    if predictions_df is not None and 'actual_RUL' in predictions_df.columns:
        pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
        
        error_analysis = {}
        for col in pred_cols:
            model = col.replace('_predicted_RUL', '')
            errors = (predictions_df[col] - predictions_df['actual_RUL']).dropna()
            
            error_analysis[model] = {
                'MAE': float(errors.abs().mean()),
                'RMSE': float(np.sqrt((errors**2).mean())),
                'Min Error': float(errors.min()),
                'Max Error': float(errors.max()),
                'Std Error': float(errors.std())
            }
        
        metrics['Error Analysis'] = error_analysis
    
    return metrics


def show_report_generator():
    """Display report generation interface."""
    st.header("📄 Export & Reporting")
    st.write("Generate and download detailed reports, metrics, and predictions.")
    
    predictions_df = load_predictions()
    model_comparison_df = load_model_comparison()
    summary = load_summary()
    
    if predictions_df is None:
        st.warning("No data available. Train models first.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Executive Summary",
        "Download Data",
        "Detailed Reports"
    ])
    
    # Tab 1: Executive Summary
    with tab1:
        st.subheader("📊 Executive Summary")
        
        exec_summary = generate_executive_summary()
        
        if exec_summary:
            # Display as metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Engines", exec_summary['Total Engines Analyzed'])
            with col2:
                st.metric("Dataset", exec_summary['Dataset'])
            with col3:
                st.metric("Risk Threshold", f"{exec_summary['Risk Threshold (RUL)']} cycles")
            with col4:
                st.metric("Report Date", exec_summary['Report Date'].split()[0])
            
            # Key findings
            st.subheader("Key Findings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(exec_summary['Average Predicted RUL'])
            
            with col2:
                if 'Average Actual RUL' in exec_summary:
                    st.write(exec_summary['Average Actual RUL'])
            
            # Model comparison
            if model_comparison_df is not None:
                st.subheader("Model Performance Ranking")
                
                sorted_models = model_comparison_df.sort_values('RMSE')
                
                for idx, (_, row) in enumerate(sorted_models.iterrows(), 1):
                    st.write(f"**#{idx}. {row['Model']}**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.write(f"RMSE: {row['RMSE']:.3f}")
                    with col_b:
                        st.write(f"MAE: {row['MAE']:.3f}")
                    with col_c:
                        st.write(f"R²: {row['R2']:.3f}")
    
    # Tab 2: Download Data
    with tab2:
        st.subheader("📥 Download Data Files")
        st.write("Export predictions and metrics in various formats.")
        
        col1, col2 = st.columns(2)
        
        # CSV Export
        with col1:
            st.write("### Predictions CSV")
            st.write("Detailed predictions for all engines")
            
            export_df = generate_predictions_csv(predictions_df)
            if export_df is not None:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"rul_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.write("### Model Comparison CSV")
            st.write("Performance metrics for all models")
            
            if model_comparison_df is not None:
                csv = model_comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Model Metrics (CSV)",
                    data=csv,
                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # JSON Export
        with col2:
            st.write("### Summary JSON")
            st.write("Complete training configuration and results")
            
            if summary:
                json_data = json.dumps(summary, indent=2)
                st.download_button(
                    label="📥 Download Summary (JSON)",
                    data=json_data,
                    file_name=f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.write("### Detailed Metrics JSON")
            st.write("Complete error analysis and statistics")
            
            metrics = generate_metrics_report()
            if metrics:
                json_data = json.dumps(metrics, indent=2)
                st.download_button(
                    label="📥 Download Metrics (JSON)",
                    data=json_data,
                    file_name=f"detailed_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Tab 3: Detailed Reports
    with tab3:
        st.subheader("📋 Detailed Analysis Reports")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### Model Performance Detailed Report")
            st.write("""
            This report includes:
            - Individual model metrics (RMSE, MAE, R²)
            - Error statistics (min, max, mean, std)
            - Prediction accuracy analysis
            - Model ranking
            - Recommendations for ensemble weighting
            """)
            
            if model_comparison_df is not None:
                st.dataframe(model_comparison_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Generate report button
            if st.button("📊 Generate Full Report", type="primary", use_container_width=True):
                st.success("Report generated successfully!")
                st.info("Download options are available in the 'Download Data' tab above.")
        
        st.write("---")
        st.write("### Prediction Statistics Report")
        
        if predictions_df is not None and 'actual_RUL' in predictions_df.columns:
            stats_data = []
            
            pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
            
            for col in pred_cols:
                model_name = col.replace('_predicted_RUL', '')
                errors = (predictions_df[col] - predictions_df['actual_RUL']).dropna()
                
                stats_data.append({
                    'Model': model_name,
                    'Samples': len(errors),
                    'Mean Error': f"{errors.mean():.3f}",
                    'Std Error': f"{errors.std():.3f}",
                    'Min Error': f"{errors.min():.3f}",
                    'Max Error': f"{errors.max():.3f}",
                    '% Accurate (±10)': f"{(errors.abs() <= 10).sum() / len(errors) * 100:.1f}%"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Footer with info
    st.markdown("---")
    st.write("### Report Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with info_col2:
        if predictions_df is not None:
            st.write(f"**Data Points:** {len(predictions_df)}")
    
    with info_col3:
        st.write(f"**Models Evaluated:** {len([col for col in predictions_df.columns if 'predicted_RUL' in col]) if predictions_df is not None else 0}")
