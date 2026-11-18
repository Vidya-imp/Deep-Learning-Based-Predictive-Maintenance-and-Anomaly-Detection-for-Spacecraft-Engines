"""Predictive Analytics Dashboard

Advanced analytics including trend analysis, forecasting, and anomaly detection.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")


WORKSPACE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = WORKSPACE_DIR / "outputs_dl_spacecraft" / "reports"


@st.cache_data
def load_predictions():
    """Load engine-level predictions."""
    path = REPORTS_DIR / "engine_level_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_summary():
    """Load training summary."""
    path = REPORTS_DIR / "summary.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def create_trend_analysis(predictions_df):
    """Analyze degradation trends across engines."""
    if predictions_df is None:
        return None
    
    # Group by unit and calculate trend metrics
    trend_data = []
    if 'unit_number' in predictions_df.columns:
        for unit in predictions_df['unit_number'].unique():
            unit_data = predictions_df[predictions_df['unit_number'] == unit]
            if 'actual_RUL' in unit_data.columns:
                actual = unit_data['actual_RUL'].values[0]
            else:
                actual = None
            
            # Get ensemble prediction (average of all models)
            pred_cols = [col for col in unit_data.columns if 'predicted_RUL' in col]
            if pred_cols:
                ensemble = np.mean([unit_data[col].values[0] for col in pred_cols])
            else:
                ensemble = None
            
            trend_data.append({
                'unit': unit,
                'actual': actual,
                'predicted': ensemble
            })
    
    return pd.DataFrame(trend_data) if trend_data else None


def create_degradation_curve(predictions_df):
    """Create interactive degradation curves."""
    trend_df = create_trend_analysis(predictions_df)
    if trend_df is None or len(trend_df) == 0:
        return None
    
    # Sort by actual RUL for visualization
    if 'actual' in trend_df.columns:
        trend_df = trend_df.dropna(subset=['actual']).sort_values('actual')
    
    if len(trend_df) == 0:
        return None
    
    fig = go.Figure()
    
    # Create x-axis as list of indices
    x_axis = list(range(len(trend_df)))
    
    # Add actual RUL
    if 'actual' in trend_df.columns and trend_df['actual'].notna().sum() > 0:
        try:
            actual_values = trend_df['actual'].fillna(0).values.tolist()
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=actual_values,
                mode='lines+markers',
                name='Actual RUL',
                line=dict(color='rgba(0, 255, 0, 0.8)', width=3),
                marker=dict(size=8, symbol='circle')
            ))
        except Exception as e:
            pass
    
    # Add predicted RUL
    if 'predicted' in trend_df.columns and trend_df['predicted'].notna().sum() > 0:
        try:
            predicted_values = trend_df['predicted'].fillna(0).values.tolist()
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=predicted_values,
                mode='lines+markers',
                name='Predicted RUL',
                line=dict(color='rgba(255, 165, 0, 0.8)', width=3, dash='dash'),
                marker=dict(size=8, symbol='square')
            ))
        except Exception as e:
            pass
    
    # Add risk zones - only if we have data
    if len(trend_df) > 0:
        max_rul = max(
            trend_df['actual'].max() if 'actual' in trend_df.columns and trend_df['actual'].notna().sum() > 0 else 200,
            trend_df['predicted'].max() if 'predicted' in trend_df.columns and trend_df['predicted'].notna().sum() > 0 else 100
        )
        
        try:
            fig.add_hrect(y0=0, y1=20, fillcolor='red', opacity=0.1, layer='below', annotation_text='Critical', annotation_position='right')
            fig.add_hrect(y0=20, y1=50, fillcolor='orange', opacity=0.1, layer='below', annotation_text='High Risk', annotation_position='right')
            fig.add_hrect(y0=50, y1=100, fillcolor='yellow', opacity=0.1, layer='below', annotation_text='Moderate', annotation_position='right')
            fig.add_hrect(y0=100, y1=max_rul, fillcolor='green', opacity=0.1, layer='below', annotation_text='Normal', annotation_position='right')
        except Exception as e:
            pass
    
    fig.update_layout(
        title='Engine Degradation Trends Across Test Set',
        xaxis_title='Engine Index',
        yaxis_title='Remaining Useful Life (cycles)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_error_distribution(predictions_df):
    """Analyze prediction errors."""
    if predictions_df is None or 'actual_RUL' not in predictions_df.columns:
        return None
    
    pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
    if not pred_cols:
        return None
    
    errors = []
    for col in pred_cols:
        model_name = col.replace('_predicted_RUL', '')
        error = (predictions_df[col] - predictions_df['actual_RUL']).dropna()
        errors.append({'Model': model_name, 'Error': error})
    
    fig = go.Figure()
    
    for error_data in errors:
        fig.add_trace(go.Box(
            y=error_data['Error'],
            name=error_data['Model'],
            boxmean='sd'
        ))
    
    fig.update_layout(
        title='Prediction Error Distribution by Model',
        yaxis_title='Prediction Error (cycles)',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_risk_matrix(predictions_df):
    """Create risk classification matrix."""
    if predictions_df is None:
        return None
    
    # Risk classification thresholds
    THRESHOLD = 20
    
    risk_data = []
    if 'unit_number' in predictions_df.columns:
        for unit in predictions_df['unit_number'].unique():
            unit_data = predictions_df[predictions_df['unit_number'] == unit].iloc[0]
            
            # Get predictions
            pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
            if pred_cols:
                ensemble_pred = np.mean([unit_data[col] for col in pred_cols])
                
                # Determine risk level
                if ensemble_pred < THRESHOLD:
                    risk = 'Critical'
                    color = 'red'
                elif ensemble_pred < 50:
                    risk = 'High'
                    color = 'orange'
                elif ensemble_pred < 100:
                    risk = 'Moderate'
                    color = 'yellow'
                else:
                    risk = 'Normal'
                    color = 'green'
                
                actual = unit_data.get('actual_RUL', None)
                risk_data.append({
                    'unit': unit,
                    'predicted': ensemble_pred,
                    'actual': actual,
                    'risk': risk,
                    'color': color
                })
    
    return pd.DataFrame(risk_data) if risk_data else None


def create_alert_system(predictions_df, threshold=20):
    """Generate alerts for critical engines."""
    if predictions_df is None:
        return []
    
    alerts = []
    
    if 'unit_number' in predictions_df.columns:
        for unit in predictions_df['unit_number'].unique():
            unit_data = predictions_df[predictions_df['unit_number'] == unit].iloc[0]
            
            # Get ensemble prediction
            pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
            if pred_cols:
                ensemble = np.mean([unit_data[col] for col in pred_cols])
                
                if ensemble < threshold:
                    alerts.append({
                        'severity': '🔴 CRITICAL',
                        'engine': f"Unit {unit}",
                        'predicted_rul': f"{ensemble:.1f}",
                        'action': 'IMMEDIATE MAINTENANCE REQUIRED'
                    })
                elif ensemble < 50:
                    alerts.append({
                        'severity': '🟠 HIGH',
                        'engine': f"Unit {unit}",
                        'predicted_rul': f"{ensemble:.1f}",
                        'action': 'Schedule maintenance within 2-3 cycles'
                    })
    
    return alerts


def show_predictive_dashboard():
    """Display predictive analytics dashboard."""
    st.header("📊 Predictive Analytics Dashboard")
    st.write("Advanced analytics for degradation trends, risk assessment, and anomaly detection.")
    
    predictions_df = load_predictions()
    summary = load_summary()
    
    if predictions_df is None:
        st.warning("No prediction data found. Train models first.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Degradation Trends",
        "Error Analysis",
        "Risk Matrix",
        "Alerts & Anomalies",
        "Statistics"
    ])
    
    # Tab 1: Degradation Trends
    with tab1:
        st.subheader("Engine Degradation Patterns")
        st.write("How well do predictions align with actual RUL?")
        
        fig = create_degradation_curve(predictions_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        trend_df = create_trend_analysis(predictions_df)
        if trend_df is not None:
            st.subheader("Trend Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'actual' in trend_df.columns and trend_df['actual'].notna().sum() > 0:
                    st.metric("Avg Actual RUL", f"{trend_df['actual'].mean():.1f} cycles")
            with col2:
                if 'predicted' in trend_df.columns and trend_df['predicted'].notna().sum() > 0:
                    st.metric("Avg Predicted RUL", f"{trend_df['predicted'].mean():.1f} cycles")
            with col3:
                if 'actual' in trend_df.columns and 'predicted' in trend_df.columns:
                    diff = (trend_df['actual'] - trend_df['predicted']).abs().mean()
                    st.metric("Mean Absolute Error", f"{diff:.1f} cycles")
    
    # Tab 2: Error Analysis
    with tab2:
        st.subheader("Prediction Error Analysis")
        st.write("How accurate is each model?")
        
        fig = create_error_distribution(predictions_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison metrics
        st.subheader("Model Performance Metrics")
        if 'actual_RUL' in predictions_df.columns:
            metrics_data = []
            for col in predictions_df.columns:
                if 'predicted_RUL' in col:
                    model_name = col.replace('_predicted_RUL', '')
                    errors = (predictions_df[col] - predictions_df['actual_RUL']).dropna()
                    
                    mae = errors.abs().mean()
                    rmse = np.sqrt((errors ** 2).mean())
                    r2 = 1 - (errors.var() / predictions_df['actual_RUL'].var()) if predictions_df['actual_RUL'].var() > 0 else 0
                    
                    metrics_data.append({
                        'Model': model_name,
                        'MAE': f"{mae:.2f}",
                        'RMSE': f"{rmse:.2f}",
                        'R²': f"{r2:.3f}"
                    })
            
            if metrics_data:
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    # Tab 3: Risk Matrix
    with tab3:
        st.subheader("Risk Classification Matrix")
        st.write("How many engines are in each risk category?")
        
        risk_df = create_risk_matrix(predictions_df)
        if risk_df is not None:
            # Risk distribution
            risk_counts = risk_df['risk'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            risk_colors = {'Critical': '🔴', 'High': '🟠', 'Moderate': '🟡', 'Normal': '🟢'}
            for risk_level in ['Critical', 'High', 'Moderate', 'Normal']:
                count = risk_counts.get(risk_level, 0)
                col = [col1, col2, col3, col4][list(risk_colors.keys()).index(risk_level)]
                with col:
                    st.metric(f"{risk_colors.get(risk_level, '')} {risk_level}", count)
            
            # Risk chart
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Risk Distribution Across Fleet',
                color_discrete_map={'Critical': '#FF4444', 'High': '#FF8844', 
                                   'Moderate': '#FFBB44', 'Normal': '#44FF44'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed risk table
            st.subheader("Risk Details by Engine")
            display_cols = ['unit', 'predicted', 'risk']
            if 'actual' in risk_df.columns:
                display_cols.append('actual')
            st.dataframe(risk_df[display_cols].rename(columns={
                'unit': 'Engine ID',
                'predicted': 'Predicted RUL',
                'risk': 'Risk Level',
                'actual': 'Actual RUL'
            }), use_container_width=True, hide_index=True)
    
    # Tab 4: Alerts & Anomalies
    with tab4:
        st.subheader("⚠️ System Alerts & Anomalies")
        
        alerts = create_alert_system(predictions_df)
        
        if alerts:
            for alert in alerts:
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 2, 1.5, 2])
                    with col1:
                        st.write(alert['severity'])
                    with col2:
                        st.write(f"**{alert['engine']}**")
                    with col3:
                        st.write(f"RUL: {alert['predicted_rul']}")
                    with col4:
                        st.write(f"*{alert['action']}*")
                    st.divider()
        else:
            st.info("✅ No critical alerts. Fleet is operating normally.")
    
    # Tab 5: Statistics Summary
    with tab5:
        st.subheader("Fleet Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Engines", len(predictions_df) if predictions_df is not None else 0)
        
        with col2:
            if 'actual_RUL' in predictions_df.columns:
                st.metric("Avg Actual RUL", f"{predictions_df['actual_RUL'].mean():.1f} cycles")
        
        with col3:
            pred_cols = [col for col in predictions_df.columns if 'predicted_RUL' in col]
            if pred_cols:
                avg_pred = predictions_df[pred_cols[0]].mean()
                st.metric("Avg Predicted RUL", f"{avg_pred:.1f} cycles")
        
        with col4:
            critical_count = create_alert_system(predictions_df, threshold=20)
            st.metric("Engines at Risk", len(critical_count))
        
        # Additional statistics
        st.write("**Additional Metrics:**")
        if predictions_df is not None:
            stats_dict = {
                "Total Test Samples": len(predictions_df),
                "Features Per Engine": 21 if summary else 'N/A',
                "Sequence Length": summary.get('sequence_length', 'N/A') if summary else 'N/A',
            }
            
            for key, value in stats_dict.items():
                st.write(f"• {key}: {value}")
