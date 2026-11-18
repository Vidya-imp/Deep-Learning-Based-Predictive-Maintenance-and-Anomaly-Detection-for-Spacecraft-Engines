"""Ensemble Model Analysis & Improvements

Compare and analyze how different models contribute to ensemble predictions.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


WORKSPACE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = WORKSPACE_DIR / "outputs_dl_spacecraft" / "reports"
MODEL_NAMES = ["BiLSTM", "CNN_BiGRU", "Transformer_Encoder"]


@st.cache_data
def load_predictions():
    """Load engine-level predictions."""
    path = REPORTS_DIR / "engine_level_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def calculate_model_weights(predictions_df):
    """Calculate optimal weights for ensemble based on accuracy."""
    if predictions_df is None or 'actual_RUL' not in predictions_df.columns:
        return {model: 1/3 for model in MODEL_NAMES}
    
    weights = {}
    total_error = 0
    
    for model in MODEL_NAMES:
        col = f"{model}_predicted_RUL"
        if col in predictions_df.columns:
            errors = np.abs(predictions_df[col] - predictions_df['actual_RUL']).dropna()
            mae = errors.mean()
            # Inverse of error gives higher weight to better models
            weight = 1.0 / (mae + 0.1) if mae > 0 else 0.33
            weights[model] = weight
            total_error += weight
    
    # Normalize weights
    if total_error > 0:
        weights = {k: v/total_error for k, v in weights.items()}
    else:
        weights = {model: 1/3 for model in MODEL_NAMES}
    
    return weights


def calculate_prediction_agreement(predictions_df):
    """Measure how much models agree on predictions."""
    if predictions_df is None:
        return None
    
    pred_cols = [f"{model}_predicted_RUL" for model in MODEL_NAMES if f"{model}_predicted_RUL" in predictions_df.columns]
    
    if len(pred_cols) < 2:
        return None
    
    agreements = []
    
    for idx, row in predictions_df.iterrows():
        preds = [row[col] for col in pred_cols]
        std = np.std(preds)
        mean = np.mean(preds)
        # Coefficient of variation
        cv = (std / mean) * 100 if mean > 0 else 0
        
        # Agreement score (lower CV = higher agreement)
        agreement = 100 - min(cv, 100)
        agreements.append({
            'unit': row.get('unit_number', idx),
            'std': std,
            'mean': mean,
            'agreement': agreement
        })
    
    return pd.DataFrame(agreements)


def create_ensemble_comparison_chart(predictions_df):
    """Create visualization comparing individual models vs ensemble."""
    if predictions_df is None:
        return None
    
    pred_cols = [f"{model}_predicted_RUL" for model in MODEL_NAMES if f"{model}_predicted_RUL" in predictions_df.columns]
    
    if len(pred_cols) < 2:
        return None
    
    # Sample subset for clarity
    sample_df = predictions_df.head(20).copy()
    
    fig = go.Figure()
    
    # Add each model's predictions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, (model, col) in enumerate(zip(MODEL_NAMES, pred_cols)):
        if col in sample_df.columns:
            fig.add_trace(go.Bar(
                x=sample_df.index,
                y=sample_df[col],
                name=model,
                marker_color=colors[idx],
                opacity=0.8
            ))
    
    # Add ensemble prediction
    ensemble = sample_df[pred_cols].mean(axis=1) if pred_cols else None
    if ensemble is not None:
        fig.add_trace(go.Scatter(
            x=sample_df.index,
            y=ensemble,
            name='Ensemble (Avg)',
            mode='lines+markers',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(size=10)
        ))
    
    # Add actual if available
    if 'actual_RUL' in sample_df.columns:
        fig.add_trace(go.Scatter(
            x=sample_df.index,
            y=sample_df['actual_RUL'],
            name='Actual RUL',
            mode='lines',
            line=dict(color='black', width=2, dash='dot'),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Model Predictions vs Ensemble vs Actual',
        xaxis_title='Engine Index',
        yaxis_title='Remaining Useful Life (cycles)',
        barmode='group',
        height=500,
        template='plotly_white',
        hovermode='x'
    )
    
    return fig


def create_weight_visualization(weights):
    """Visualize model ensemble weights."""
    fig = go.Figure(data=[
        go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c']),
            textposition='inside',
            textinfo='label+percent'
        )
    ])
    
    fig.update_layout(
        title='Ensemble Model Weights (Based on Accuracy)',
        height=400
    )
    
    return fig


def create_agreement_visualization(agreement_df):
    """Visualize model agreement levels."""
    if agreement_df is None or len(agreement_df) == 0:
        return None
    
    # Create histogram of agreement
    fig = go.Figure(data=[
        go.Histogram(
            x=agreement_df['agreement'],
            nbinsx=20,
            marker_color='rgba(0, 100, 200, 0.7)',
            name='Agreement Score'
        )
    ])
    
    fig.update_layout(
        title='Distribution of Model Agreement',
        xaxis_title='Agreement Score (%)',
        yaxis_title='Number of Engines',
        height=400,
        template='plotly_white'
    )
    
    return fig


def show_ensemble_analyzer():
    """Display ensemble analysis interface."""
    st.header("🤝 Model Ensemble Analysis & Optimization")
    st.write("Analyze how different models contribute to predictions and optimize ensemble weights.")
    
    predictions_df = load_predictions()
    
    if predictions_df is None:
        st.warning("No prediction data found. Train models first.")
        return
    
    # Check if we have predictions from multiple models
    pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
    if len(pred_cols) < 2:
        st.warning("Need predictions from at least 2 models for ensemble analysis.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Weights",
        "Prediction Comparison",
        "Agreement Analysis",
        "Uncertainty Quantification"
    ])
    
    # Tab 1: Model Weights
    with tab1:
        st.subheader("📊 Ensemble Model Weights")
        st.write("How much should each model contribute to the final prediction?")
        
        weights = calculate_model_weights(predictions_df)
        
        fig = create_weight_visualization(weights)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Weight Details")
        weight_data = pd.DataFrame({
            'Model': list(weights.keys()),
            'Weight': [f"{v:.3f}" for v in weights.values()],
            'Percentage': [f"{v*100:.1f}%" for v in weights.values()]
        })
        st.dataframe(weight_data, use_container_width=True, hide_index=True)
        
        st.info("""
        **Interpretation:**
        - Weights are calculated based on Mean Absolute Error (MAE)
        - Models with lower errors get higher weights
        - Weights are normalized to sum to 1.0
        - In production, use weighted averaging: `ensemble = Σ(weight_i × prediction_i)`
        """)
    
    # Tab 2: Prediction Comparison
    with tab2:
        st.subheader("🎯 Individual vs Ensemble Predictions")
        st.write("How do individual model predictions compare?")
        
        fig = create_ensemble_comparison_chart(predictions_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Prediction Comparison (First 10 Engines)")
        comparison_data = []
        
        for idx, row in predictions_df.head(10).iterrows():
            actual = row.get('actual_RUL', None)
            comparison_row = {'Engine': row.get('unit_number', idx)}
            
            for model in MODEL_NAMES:
                col = f"{model}_predicted_RUL"
                if col in predictions_df.columns:
                    pred = row[col]
                    comparison_row[model] = f"{pred:.1f}"
            
            # Ensemble prediction
            ensemble_preds = [row[f"{m}_predicted_RUL"] for m in MODEL_NAMES 
                             if f"{m}_predicted_RUL" in predictions_df.columns]
            if ensemble_preds:
                comparison_row['Ensemble'] = f"{np.mean(ensemble_preds):.1f}"
            
            if actual:
                comparison_row['Actual'] = f"{actual:.1f}"
            
            comparison_data.append(comparison_row)
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # Tab 3: Agreement Analysis
    with tab3:
        st.subheader("🤝 Model Agreement & Confidence")
        st.write("How much do models agree on their predictions?")
        
        agreement_df = calculate_prediction_agreement(predictions_df)
        
        if agreement_df is not None:
            fig = create_agreement_visualization(agreement_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Agreement", f"{agreement_df['agreement'].mean():.1f}%")
            with col2:
                st.metric("Min Agreement", f"{agreement_df['agreement'].min():.1f}%")
            with col3:
                st.metric("Max Agreement", f"{agreement_df['agreement'].max():.1f}%")
            with col4:
                high_agreement = len(agreement_df[agreement_df['agreement'] > 80])
                st.metric("High Confidence Cases", high_agreement)
            
            # Show low-agreement cases (potential uncertainty)
            st.subheader("⚠️ Low Confidence Predictions (Agreement < 70%)")
            low_conf = agreement_df[agreement_df['agreement'] < 70].sort_values('agreement')
            
            if len(low_conf) > 0:
                st.dataframe(low_conf[['unit', 'agreement', 'std']].head(10), 
                           use_container_width=True, hide_index=True)
                st.info(f"Found {len(low_conf)} engines where models disagree significantly. These predictions have high uncertainty.")
            else:
                st.success("✅ All predictions have high agreement (>70%). Models are confident.")
    
    # Tab 4: Uncertainty Quantification
    with tab4:
        st.subheader("📈 Prediction Uncertainty Analysis")
        st.write("Quantify uncertainty in ensemble predictions using prediction variance.")
        
        # Calculate uncertainty for each engine
        uncertainty_data = []
        
        for idx, row in predictions_df.iterrows():
            preds = [row[f"{m}_predicted_RUL"] for m in MODEL_NAMES 
                    if f"{m}_predicted_RUL" in predictions_df.columns]
            
            if preds:
                ensemble = np.mean(preds)
                std = np.std(preds)
                conf_interval_low = ensemble - 1.96 * std
                conf_interval_high = ensemble + 1.96 * std
                
                uncertainty_data.append({
                    'unit': row.get('unit_number', idx),
                    'ensemble': ensemble,
                    'std': std,
                    'ci_low': max(0, conf_interval_low),
                    'ci_high': conf_interval_high,
                    'actual': row.get('actual_RUL', None)
                })
        
        if uncertainty_data:
            unc_df = pd.DataFrame(uncertainty_data)
            
            # Create error bars chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=unc_df['unit'],
                y=unc_df['ensemble'],
                error_y=dict(
                    type='data',
                    array=unc_df['std'],
                    visible=True
                ),
                mode='markers',
                name='Ensemble ± 1σ',
                marker=dict(size=8, color='blue')
            ))
            
            if 'actual' in unc_df.columns and unc_df['actual'].notna().sum() > 0:
                fig.add_trace(go.Scatter(
                    x=unc_df['unit'],
                    y=unc_df['actual'],
                    mode='markers',
                    name='Actual RUL',
                    marker=dict(size=6, color='red', symbol='x')
                ))
            
            fig.update_layout(
                title='Ensemble Predictions with Uncertainty Bounds',
                xaxis_title='Engine',
                yaxis_title='RUL (cycles)',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence interval table
            st.subheader("95% Confidence Intervals")
            st.dataframe(
                unc_df[['unit', 'ensemble', 'std', 'ci_low', 'ci_high']].head(15),
                use_container_width=True,
                hide_index=True
            )
            
            st.info("""
            **Interpretation:**
            - **Ensemble**: Mean prediction across all 3 models
            - **Std**: Standard deviation of predictions
            - **CI [Low, High]**: 95% Confidence interval
            - Wider intervals = more uncertain predictions
            - Use confidence intervals for decision-making thresholds
            """)
