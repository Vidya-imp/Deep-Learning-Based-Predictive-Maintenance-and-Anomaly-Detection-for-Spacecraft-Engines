"""Custom UI Components & Styling

Enhanced visual components and custom CSS for better UX.
"""

import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app."""
    custom_css = """
    <style>
    
    /* Dark sidebar and main page background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 50%, #141820 100%);
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.8);
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background-color: transparent;
    }
    
    /* Sidebar text styling */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #64b5f6 !important;
        text-shadow: 0 0 10px rgba(100, 181, 246, 0.5);
    }
    
    /* Main content area with space theme background */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        background-attachment: fixed;
        padding: 2rem;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Animated starfield background */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 60px 70px, #fff, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 50px 50px, #fff, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 130px 80px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 90px 10px, #fff, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 200px 200px;
        animation: twinkle 5s infinite;
        opacity: 0.5;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 0.8; }
    }
    
    /* Main content wrapper to sit above starfield */
    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }
    
    /* Spacecraft floating animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px) translateX(0px); }
        25% { transform: translateY(-20px) translateX(10px); }
        50% { transform: translateY(-40px) translateX(-10px); }
        75% { transform: translateY(-20px) translateX(10px); }
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 10px rgba(100, 181, 246, 0.5); }
        50% { text-shadow: 0 0 20px rgba(100, 181, 246, 0.8), 0 0 30px rgba(66, 165, 245, 0.6); }
    }
    
    /* Page title with glow effect */
    h1 {
        color: #64b5f6 !important;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1976d2;
        padding-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite;
        text-shadow: 0 0 20px rgba(25, 118, 210, 0.8);
    }
    
    h2 {
        color: #81c784 !important;
        font-weight: 600;
        margin-top: 1.5rem;
        text-shadow: 0 0 10px rgba(129, 199, 132, 0.5);
    }
    
    h3 {
        color: #ffb74d !important;
        font-weight: 500;
        text-shadow: 0 0 8px rgba(255, 183, 77, 0.4);
    }
    
    /* Metric cards with deep space theme */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(25, 118, 210, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
        border: 1px solid rgba(100, 181, 246, 0.5);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.4) 0%, rgba(142, 68, 173, 0.4) 100%) !important;
        border-color: rgba(100, 181, 246, 0.8);
        box-shadow: 0 12px 24px rgba(25, 118, 210, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transform: translateY(-4px);
    }
    
    /* Button styling with neon glow */
    .stButton > button {
        background: linear-gradient(90deg, #1976d2 0%, #7b1fa2 100%);
        color: white;
        border: 1px solid rgba(100, 181, 246, 0.5);
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.4);
        text-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2196f3 0%, #9c27b0 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(25, 118, 210, 0.6), 0 0 20px rgba(100, 181, 246, 0.4);
        border-color: rgba(100, 181, 246, 0.8);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.4);
    }
    
    /* Expander with dark theme */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(15, 20, 25, 0.8), rgba(25, 118, 210, 0.2));
        border: 1px solid rgba(100, 181, 246, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #64b5f6 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(25, 118, 210, 0.2), rgba(25, 118, 210, 0.3));
        border-color: rgba(100, 181, 246, 0.6);
    }
    
    /* Alert boxes with space theme */
    .stAlert {
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid;
        backdrop-filter: blur(5px);
        background-color: rgba(0, 0, 0, 0.3);
    }
    
    /* Info box */
    [data-testid="stInfo"] {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.2), rgba(13, 110, 253, 0.1)) !important;
        border: 1px solid rgba(13, 110, 253, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Success box */
    [data-testid="stSuccess"] {
        background: linear-gradient(135deg, rgba(25, 135, 84, 0.2), rgba(25, 135, 84, 0.1)) !important;
        border: 1px solid rgba(25, 135, 84, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Warning box */
    [data-testid="stWarning"] {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2), rgba(255, 193, 7, 0.1)) !important;
        border: 1px solid rgba(255, 193, 7, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Error box */
    [data-testid="stError"] {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.2), rgba(220, 53, 69, 0.1)) !important;
        border: 1px solid rgba(220, 53, 69, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 20, 25, 0.6);
        border-bottom: 2px solid rgba(100, 181, 246, 0.3);
        backdrop-filter: blur(5px);
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
        color: #b0bec5;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #64b5f6;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom-color: #1976d2;
        color: #64b5f6;
        text-shadow: 0 0 10px rgba(25, 118, 210, 0.6);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        color: #e0e0e0 !important;
        backdrop-filter: blur(5px);
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: rgba(100, 181, 246, 0.8) !important;
        box-shadow: 0 0 15px rgba(25, 118, 210, 0.3) !important;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div > input {
        background: rgba(100, 181, 246, 0.3);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        background: rgba(15, 20, 25, 0.6);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(100, 181, 246, 0.3);
    }
    
    .dataframe th {
        background: linear-gradient(90deg, rgba(25, 118, 210, 0.4), rgba(118, 75, 162, 0.4));
        color: #64b5f6;
        padding: 14px;
        text-align: left;
        font-weight: 700;
        border-bottom: 2px solid rgba(100, 181, 246, 0.5);
        text-shadow: 0 0 8px rgba(0, 0, 0, 0.5);
    }
    
    .dataframe td {
        padding: 12px 14px;
        border-bottom: 1px solid rgba(100, 181, 246, 0.2);
        color: #e0e0e0;
    }
    
    .dataframe tr:hover {
        background: rgba(100, 181, 246, 0.1);
    }
    
    /* Container borders */
    [data-testid="stContainer"] {
        border-radius: 8px;
        border: 1px solid rgba(100, 181, 246, 0.3);
        padding: 1rem;
        margin: 1rem 0;
        background: rgba(15, 20, 25, 0.4);
        backdrop-filter: blur(5px);
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(100, 181, 246, 0.3);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
    }
    
    /* Spacer elements */
    .spacer {
        height: 20px;
    }
    
    /* Code blocks */
    pre, code {
        background: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(100, 181, 246, 0.2) !important;
        border-radius: 6px !important;
        color: #81c784 !important;
    }
    
    /* Radio buttons and checkboxes */
    .stCheckbox, .stRadio {
        color: #e0e0e0 !important;
    }
    
    .stCheckbox > label, .stRadio > label {
        color: #e0e0e0 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 20, 25, 0.8);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1976d2, #7b1fa2);
        border-radius: 5px;
        border: 2px solid rgba(15, 20, 25, 0.8);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #2196f3, #9c27b0);
    }
    
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)


def create_metric_card(label, value, delta=None, icon=None):
    """Create a styled metric card."""
    col = st.columns([1])[0]
    
    with col:
        html_content = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <p style="margin: 0; font-size: 14px; opacity: 0.9;">{icon} {label}</p>
            <h3 style="margin: 10px 0 0 0; font-size: 28px; font-weight: bold;">{value}</h3>
            {f'<p style="margin: 5px 0 0 0; font-size: 12px;">{delta}</p>' if delta else ''}
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)


def create_status_badge(status, size="medium"):
    """Create a styled status badge."""
    status_colors = {
        "🔴 CRITICAL": "#FF4444",
        "🟠 HIGH": "#FF8844",
        "🟡 MODERATE": "#FFBB44",
        "🟢 NORMAL": "#44FF44",
    }
    
    color = status_colors.get(status, "#999999")
    
    sizes = {"small": "10px", "medium": "14px", "large": "18px"}
    font_size = sizes.get(size, "14px")
    
    html = f"""
    <span style="
        display: inline-block;
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: {font_size};
        font-weight: bold;
        margin: 2px;
    ">
        {status}
    </span>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def create_progress_bar(current, maximum, label="Progress"):
    """Create a styled progress bar."""
    percentage = (current / maximum) * 100
    
    html = f"""
    <div style="margin: 20px 0;">
        <p style="margin: 0 0 8px 0; font-weight: bold;">{label}</p>
        <div style="
            width: 100%;
            height: 25px;
            background-color: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 12px;
            ">
                {percentage:.0f}%
            </div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def create_info_box(title, content, box_type="info"):
    """Create a styled information box."""
    box_colors = {
        "info": ("#d1ecf1", "#0c5460"),
        "success": ("#d4edda", "#155724"),
        "warning": ("#fff3cd", "#856404"),
        "error": ("#f8d7da", "#721c24"),
    }
    
    bg_color, border_color = box_colors.get(box_type, box_colors["info"])
    
    html = f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    ">
        <h4 style="margin: 0 0 10px 0; color: {border_color};">{title}</h4>
        <p style="margin: 0; color: {border_color};">{content}</p>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def create_spacecraft_header():
    """Display an animated spacecraft header with parallax effect."""
    spacecraft_html = """
    <style>
    @keyframes float {
        0%, 100% { transform: translateY(0px) translateX(0px); }
        25% { transform: translateY(-15px) translateX(5px); }
        50% { transform: translateY(-30px) translateX(-5px); }
        75% { transform: translateY(-15px) translateX(5px); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes glow-pulse {
        0%, 100% { filter: drop-shadow(0 0 5px rgba(100, 181, 246, 0.5)); }
        50% { filter: drop-shadow(0 0 15px rgba(100, 181, 246, 0.8)); }
    }
    
    .spacecraft-container {
        position: relative;
        width: 100%;
        height: 220px;
        background: radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.15), transparent 4%),
                    radial-gradient(circle at 85% 35%, rgba(255, 255, 255, 0.12), transparent 3%),
                    radial-gradient(circle at 55% 75%, rgba(255, 255, 255, 0.1), transparent 3%),
                    linear-gradient(180deg, #02040d 0%, #08172f 35%, #050c21 100%);
        border-radius: 12px;
        border: 2px solid rgba(100, 181, 246, 0.5);
        overflow: hidden;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 30px rgba(25, 118, 210, 0.4), inset 0 1px 0 rgba(100, 181, 246, 0.2);
        background-size: cover;
    }
    
    .spacecraft-svg {
        animation: float 6s ease-in-out infinite, glow-pulse 3s ease-in-out infinite;
        filter: drop-shadow(0 0 15px rgba(100, 181, 246, 0.8));
    }
    
    .header-text {
        position: absolute;
        top: 15px;
        right: 25px;
        color: #64b5f6;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 15px rgba(100, 181, 246, 0.8);
    }
    
    .star {
        position: absolute;
        width: 1px;
        height: 1px;
        background: white;
        border-radius: 50%;
        animation: pulse 3s infinite;
    }
    </style>
    
    <div class="spacecraft-container">
        <div class="header-text">🛸 Advanced RUL Prediction System</div>
        
        <svg class="spacecraft-svg" width="140" height="110" viewBox="0 0 120 100" preserveAspectRatio="xMidYMid meet">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
                <radialGradient id="engineGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" style="stop-color:#ffa726;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#ff6f00;stop-opacity:0" />
                </radialGradient>
            </defs>
            
            <!-- Main spacecraft body -->
            <ellipse cx="60" cy="45" rx="38" ry="28" fill="#1e88e5" opacity="0.95" filter="url(#glow)"/>
            <path d="M 25 45 Q 25 68 60 75 Q 95 68 95 45" fill="#2196f3" opacity="0.9"/>
            
            <!-- Engine glow circle -->
            <circle cx="60" cy="72" r="10" fill="url(#engineGlow)" opacity="0.8"/>
            
            <!-- Engine flames -->
            <ellipse cx="60" cy="75" rx="7" ry="14" fill="#ff6f00" opacity="0.8" filter="url(#glow)"/>
            <ellipse cx="60" cy="80" rx="5" ry="10" fill="#ffa726" opacity="0.7"/>
            
            <!-- Energy particles -->
            <circle cx="48" cy="74" r="2.5" fill="#ffeb3b" opacity="0.9" filter="url(#glow)"/>
            <circle cx="72" cy="74" r="2.5" fill="#ffeb3b" opacity="0.9" filter="url(#glow)"/>
            <circle cx="60" cy="82" r="2" fill="#fff9c4" opacity="0.8"/>
            
            <!-- Stars around -->
            <circle cx="15" cy="25" r="2" fill="#81c784" opacity="0.8" filter="url(#glow)"/>
            <circle cx="105" cy="35" r="1.5" fill="#81c784" opacity="0.7"/>
            <circle cx="10" cy="65" r="1.5" fill="#81c784" opacity="0.7"/>
        </svg>
    </div>
    """
    
    st.markdown(spacecraft_html, unsafe_allow_html=True)


def create_feature_showcase(features_dict):
    """Create a showcase of features with icons."""
    cols = st.columns(len(features_dict))
    
    for idx, (feature_name, description) in enumerate(features_dict.items()):
        with cols[idx]:
            html = f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            " onmouseover="this.style.transform='translateY(-5px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                <h3 style="margin: 0 0 10px 0; font-size: 18px;">{feature_name}</h3>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">{description}</p>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)


def show_ui_components_demo():
    """Display UI components demo page."""
    st.header("🎨 Custom UI Components")
    st.write("Improved visual design with custom styling and enhanced components.")
    
    apply_custom_styling()
    
    # Tab for different components
    tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Badges", "Boxes", "Components"])
    
    with tab1:
        st.subheader("Metric Cards")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", "94.2%", "+2.1%")
        with col2:
            st.metric("⏱️ Response Time", "245ms", "-15ms")
        with col3:
            st.metric("📊 Data Points", "1,245", "+345")
    
    with tab2:
        st.subheader("Status Badges")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_status_badge("🔴 CRITICAL")
        with col2:
            create_status_badge("🟠 HIGH")
        with col3:
            create_status_badge("🟡 MODERATE")
        with col4:
            create_status_badge("🟢 NORMAL")
    
    with tab3:
        st.subheader("Information Boxes")
        
        create_info_box(
            "ℹ️ Information",
            "This is an information box with important details.",
            "info"
        )
        
        create_info_box(
            "✅ Success",
            "Operation completed successfully.",
            "success"
        )
        
        create_info_box(
            "⚠️ Warning",
            "Please review before proceeding.",
            "warning"
        )
        
        create_info_box(
            "❌ Error",
            "Something went wrong during processing.",
            "error"
        )
    
    with tab4:
        st.subheader("Progress Indicators")
        
        create_progress_bar(65, 100, "Model Training Progress")
        create_progress_bar(92, 100, "Data Processing")
        create_progress_bar(100, 100, "Evaluation Complete")
        
        st.subheader("Feature Showcase")
        
        features = {
            "⚡ Real-Time": "Instant predictions",
            "🧠 AI-Powered": "Advanced algorithms",
            "📊 Analytics": "Detailed insights",
            "🔒 Secure": "Data protection"
        }
        
        create_feature_showcase(features)
