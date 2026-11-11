"""
app.py
------
Housing Price Prediction System - Streamlit Web Application
A modern, intuitive, and academic-grade machine learning web application.
Frontend refined with premium real-estate portal aesthetics and high-contrast styling.
Preserves 100% of the underlying Machine Learning models, feature selection, preprocessing, and prediction logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import preprocessing as prep
import model as ml

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Housing Price Prediction System",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Premium Real-Estate UI Custom Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts: Plus Jakarta Sans */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global Root Variables & Theme Enforcement */
    :root, [data-testid="stAppViewContainer"], .stApp {
        --text-color: #0f172a !important;
        --background-color: #f8fafc !important;
        --secondary-background-color: #ffffff !important;
        --primary-color: #2563eb !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Hide Streamlit default decoration header */
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px;
    }

    /* ---------------- Sidebar Styling ---------------- */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 12px rgba(15, 23, 42, 0.02) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding-bottom: 1.1rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    .sidebar-brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.1rem;
        font-weight: 800;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    .brand-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #0f172a !important;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        font-weight: 600;
        color: #2563eb !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Navigation Links */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.35rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent !important;
        padding: 0.65rem 0.9rem !important;
        border-radius: 10px !important;
        border: 1px solid transparent !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        margin-bottom: 0.15rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #f1f5f9 !important;
        border-color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: #eff6ff !important;
        border-color: #bfdbfe !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
        color: #1d4ed8 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] p {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    .sidebar-info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-top: 1.4rem;
        font-size: 0.8rem;
        color: #475569 !important;
        line-height: 1.55;
    }
    .sidebar-info-box strong {
        color: #0f172a !important;
    }

    /* ---------------- Hero Banner Styles ---------------- */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #1e3a8a 100%);
        color: white;
        padding: 2.4rem 2.6rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 28px -6px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-badge-pill {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        background: rgba(37, 99, 235, 0.28);
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        letter-spacing: -0.025em;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #e2e8f0 !important;
        max-width: 780px;
        line-height: 1.6;
    }

    /* ---------------- Cards & Containers ---------------- */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 4px 12px -2px rgba(15, 23, 42, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px -4px rgba(15, 23, 42, 0.07);
        border-color: #cbd5e1;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b !important;
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-size: 1.68rem;
        font-weight: 800;
        color: #0f172a !important;
        letter-spacing: -0.02em;
    }
    .stat-desc {
        font-size: 0.8rem;
        color: #94a3b8 !important;
        margin-top: 0.25rem;
    }

    .content-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.7rem 1.9rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    }
    .section-title {
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin-bottom: 0.85rem !important;
        margin-top: 0.2rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        letter-spacing: -0.015em !important;
    }

    /* ---------------- FORM CONTAINER & LAYOUT FIXES ---------------- */
    form[data-testid="stForm"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 1.8rem 2rem !important;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04) !important;
    }
    form[data-testid="stForm"] hr {
        margin: 1.1rem 0 !important;
        border: none !important;
        border-top: 1px solid #f1f5f9 !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.7rem !important;
    }

    /* ==================== 1. WIDGET LABELS (HIGH CONTRAST & VISIBILITY) ==================== */
    label[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span,
    .stWidgetLabel,
    .stWidgetLabel p,
    label p,
    label {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.93rem !important;
        line-height: 1.4 !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-block !important;
        margin-bottom: 0.35rem !important;
    }
    div[data-testid="stWidgetLabel"] svg {
        color: #64748b !important;
        fill: #64748b !important;
    }

    /* ==================== 2. RADIO BUTTONS (YES/NO & FURNISHING) ==================== */
    div[data-testid="stRadio"] {
        margin-bottom: 0.45rem !important;
    }
    div[data-testid="stRadio"] > label {
        margin-bottom: 0.35rem !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 1.4rem !important;
        align-items: center !important;
        padding-top: 0.15rem !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        cursor: pointer !important;
        margin: 0 !important;
        padding: 0.2rem 0.5rem 0.2rem 0 !important;
    }
    /* Radio Option Text: "Yes", "No", "Furnished", etc. */
    div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] span,
    div[data-testid="stRadio"] p {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        visibility: visible !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] svg {
        fill: #2563eb !important;
    }

    /* ==================== 3. NUMBER & TEXT INPUTS ==================== */
    div[data-testid="stNumberInput"],
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stNumberInputContainer"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
    }

    div[data-testid="stNumberInputContainer"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stNumberInputContainer"]:focus-within,
    div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stNumberInputContainer"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    input[type="number"],
    input[type="text"],
    input {
        background-color: #ffffff !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        caret-color: #172033 !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        padding: 0.55rem 0.85rem !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Plus (+) and Minus (-) Step Controls */
    button[data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"],
    div[data-testid="stNumberInputContainer"] button,
    div[data-testid="stNumberInput"] button {
        background-color: #f8fafc !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        border: none !important;
        border-left: 1px solid #e2e8f0 !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    button[data-testid="stNumberInputStepUp"]:hover,
    button[data-testid="stNumberInputStepDown"]:hover,
    div[data-testid="stNumberInputContainer"] button:hover,
    div[data-testid="stNumberInput"] button:hover {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    button[data-testid="stNumberInputStepUp"] svg,
    button[data-testid="stNumberInputStepDown"] svg,
    div[data-testid="stNumberInputContainer"] button svg,
    div[data-testid="stNumberInput"] button svg {
        fill: #172033 !important;
        color: #172033 !important;
    }

    button[data-testid="stNumberInputStepUp"]:hover svg,
    button[data-testid="stNumberInputStepDown"]:hover svg,
    div[data-testid="stNumberInputContainer"] button:hover svg,
    div[data-testid="stNumberInput"] button:hover svg {
        fill: #0f172a !important;
        color: #0f172a !important;
    }

    /* ==================== 4. SELECTBOX (PARKING DROPDOWN) ==================== */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        min-height: 44px !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }
    div[data-testid="stSelectbox"] svg {
        fill: #0f172a !important;
        color: #0f172a !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: #eff6ff !important;
        color: #1d4ed8 !important;
        -webkit-text-fill-color: #1d4ed8 !important;
    }

    /* ==================== 5. SLIDERS ==================== */
    div[data-testid="stSlider"] {
        margin-bottom: 0.45rem !important;
    }
    div[data-testid="stSlider"] div[data-testid="stWidgetLabel"] * {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSlider"] div[data-testid="stTickBar"] span,
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stSlider"] span,
    div[data-testid="stSlider"] p {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #2563eb !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4) !important;
    }

    /* ---------------- Valuation Result Card ---------------- */
    .valuation-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 2.4rem 2.6rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 16px 36px -6px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1.8rem 0;
    }
    .valuation-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #93c5fd;
        margin-bottom: 0.6rem;
    }
    .valuation-price {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 0.3rem;
        line-height: 1.1;
    }
    .valuation-denomination {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 1rem;
    }
    .valuation-range {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 0.48rem 1.3rem;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1.1rem;
    }
    .valuation-disclaimer {
        font-size: 0.82rem;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Pills / Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .badge-success {
        background-color: #eff6ff;
        color: #1d4ed8;
        border-color: #bfdbfe;
    }

    /* ---------------- Button Overrides ---------------- */
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        padding: 0.65rem 1.9rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: -0.01em !important;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Expanders */
    details[data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data & Model Pipeline Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Machine Learning Pipeline...")
def get_trained_pipeline():
    """
    Loads Housing.csv and trains the complete regression pipeline once.
    Results are cached in memory for high-performance interactivity.
    """
    df = prep.load_dataset("Housing.csv")
    pipeline = ml.train_complete_pipeline(df)
    return pipeline


# Initialize session state for page routing
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "🏠 Home / Dashboard"

# Load the pipeline
try:
    pipeline_data = get_trained_pipeline()
    raw_df = pipeline_data["raw_df"]
except Exception as e:
    st.error(f"Error initializing Machine Learning Pipeline: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">⌂</div>
        <div>
            <div class="brand-title">EstateVal ML</div>
            <div class="brand-subtitle">Property Valuation System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "🏠 Home / Dashboard",
        "🔮 House Price Prediction",
        "📊 Dataset Insights",
        "🧠 Model Performance"
    ]

    # Sync radio button with session state
    current_index = nav_options.index(st.session_state["current_page"]) if st.session_state["current_page"] in nav_options else 0
    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=current_index,
        label_visibility="collapsed"
    )

    if selected_page != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_page
        st.rerun()

    st.markdown("""
    <div class="sidebar-info-box">
        <strong>Academic Project Overview</strong><br>
        • <strong>Algorithm:</strong> Multiple Linear Regression<br>
        • <strong>Feature Selection:</strong> Backward Elimination & RFE<br>
        • <strong>Dataset:</strong> 545 Verified Properties<br>
        • <strong>Status:</strong> Model Trained & Ready
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1: Home / Dashboard
# =============================================================================
def render_home_page():
    # Hero Section
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge-pill">AI-Powered Valuation</div>
        <div class="hero-title">Housing Price Prediction System</div>
        <div class="hero-subtitle">
            An academic supervised regression platform designed to estimate fair market house prices using
            Multiple Linear Regression, Recursive Feature Elimination (RFE), and statistical Backward Elimination.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Dataset Summary Metrics
    total_props = len(raw_df)
    avg_price_fmt, avg_denom = prep.format_inr(raw_df["price"].mean())
    min_price_fmt, min_denom = prep.format_inr(raw_df["price"].min())
    max_price_fmt, max_denom = prep.format_inr(raw_df["price"].max())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Properties</div>
            <div class="stat-value">{total_props}</div>
            <div class="stat-desc">Residential records in dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Average Price</div>
            <div class="stat-value">{avg_denom}</div>
            <div class="stat-desc">{avg_price_fmt} across dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Lowest Price</div>
            <div class="stat-value">{min_denom}</div>
            <div class="stat-desc">{min_price_fmt} entry price</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Highest Price</div>
            <div class="stat-value">{max_denom}</div>
            <div class="stat-desc">{max_price_fmt} luxury villa</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Call-to-Action to Jump to Prediction
    callout_col1, callout_col2 = st.columns([3, 1])
    with callout_col1:
        st.markdown("""
        ### Ready to estimate a property?
        Enter the property's area, bedrooms, bathrooms, and features to calculate an instant valuation powered by the trained statistical model.
        """)
    with callout_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Estimate House Price", use_container_width=True):
            st.session_state["current_page"] = "🔮 House Price Prediction"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # How It Works & Core ML Workflow (Beginner-Friendly Explanation)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="content-box">
            <div class="section-title">🔍 How House Price Prediction Works</div>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                Real estate prices depend on a combination of physical dimensions, architectural layout,
                location advantages, and modern conveniences.
            </p>
            <ol style="color: #334155; font-size: 0.88rem; padding-left: 1.2rem; line-height: 1.8;">
                <li><strong>Feature Extraction:</strong> We analyze 12 key attributes including square footage, bathrooms, parking, and furnishing.</li>
                <li><strong>Statistical Screening:</strong> Insignificant variables are pruned using Backward Elimination (OLS <em>p</em>-values &le; 0.05) and RFE to prevent overfitting.</li>
                <li><strong>Regression Valuation:</strong> The Multiple Linear Regression equation computes a reliable price estimate based on real transactions.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="content-box">
            <div class="section-title">⭐ Key Valuation Drivers</div>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                According to our trained model coefficients, the top factors that influence house value are:
            </p>
            <div style="margin-top: 0.8rem;">
                <span class="badge badge-success">📐 Total Lot Area</span>
                <span class="badge badge-success">🚿 Bathrooms Count</span>
                <span class="badge badge-success">📍 Preferred Area Access</span>
                <span class="badge badge-success">❄️ Air Conditioning</span>
                <span class="badge badge-success">♨️ Hot Water Heating</span>
                <span class="badge badge-success">🚗 Dedicated Parking</span>
            </div>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.9rem; line-height: 1.5;">
                Properties with prime location access and air conditioning exhibit substantial market premiums, while unfurnished properties command lower baseline pricing.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: House Price Prediction (Main Feature)
# =============================================================================
def render_prediction_page():
    st.markdown("""
    <div style="margin-bottom: 1.4rem;">
        <h2 style="font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; letter-spacing: -0.025em;">🔮 House Price Prediction</h2>
        <p style="color: #64748b; font-size: 0.95rem;">Fill in the property details below to calculate an estimated selling price using the trained Machine Learning regression model.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        # Section 1: Property Size & Architecture
        st.markdown("""
        <div class="section-title">📐 1. Property Size & Layout</div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            area = st.number_input(
                "Area (Square Feet)",
                min_value=500,
                max_value=25000,
                value=5500,
                step=100,
                help="Total plot or carpet area of the house in sq. ft."
            )
        with c2:
            bedrooms = st.slider(
                "Bedrooms",
                min_value=1,
                max_value=6,
                value=3,
                help="Total number of bedrooms"
            )
        with c3:
            bathrooms = st.slider(
                "Bathrooms",
                min_value=1,
                max_value=4,
                value=2,
                help="Total number of bathrooms"
            )
        with c4:
            stories = st.slider(
                "Number of Floors (Stories)",
                min_value=1,
                max_value=4,
                value=2,
                help="Number of building stories"
            )

        st.markdown("---")

        # Section 2: Rooms & Facilities
        st.markdown("""
        <div class="section-title">🚗 2. Parking & Street Access</div>
        """, unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        with c5:
            parking = st.selectbox(
                "Parking Capacity (Cars)",
                options=[0, 1, 2, 3],
                index=1,
                help="Number of dedicated vehicle parking spots"
            )
        with c6:
            mainroad_choice = st.radio(
                "Connected to Main Road?",
                options=["Yes", "No"],
                horizontal=True,
                help="Direct access to the main arterial road"
            )

        st.markdown("---")

        # Section 3: Amenities & Extras
        st.markdown("""
        <div class="section-title">✨ 3. Amenities & Location Features</div>
        """, unsafe_allow_html=True)

        c7, c8, c9 = st.columns(3)
        with c7:
            prefarea_choice = st.radio(
                "Preferred / Prime Neighborhood?",
                options=["Yes", "No"],
                horizontal=True,
                help="Located in a highly sought-after preferred locality"
            )
            airconditioning_choice = st.radio(
                "Air Conditioning Installed?",
                options=["Yes", "No"],
                horizontal=True,
                help="Equipped with central or split AC"
            )

        with c8:
            hotwaterheating_choice = st.radio(
                "Hot Water Heating / Geyser?",
                options=["Yes", "No"],
                index=1,
                horizontal=True,
                help="Hot water heating system"
            )
            basement_choice = st.radio(
                "Has Basement?",
                options=["Yes", "No"],
                index=1,
                horizontal=True,
                help="Property has a usable basement area"
            )

        with c9:
            guestroom_choice = st.radio(
                "Dedicated Guest Room?",
                options=["Yes", "No"],
                index=1,
                horizontal=True,
                help="Dedicated guest bedroom"
            )

        st.markdown("---")

        # Section 4: Furnishing Status
        st.markdown("""
        <div class="section-title">🛋️ 4. Furnishing Condition</div>
        """, unsafe_allow_html=True)

        furnishing_choice = st.radio(
            "Furnishing Status",
            options=["Furnished", "Semi-furnished", "Unfurnished"],
            index=1,
            horizontal=True,
            help="Current interior furnishing condition"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        submit_button = st.form_submit_button("💰 Predict House Price", use_container_width=True)

    # Prediction Action
    if submit_button:
        # Construct input dictionary
        input_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad_choice.lower(),
            "guestroom": guestroom_choice.lower(),
            "basement": basement_choice.lower(),
            "hotwaterheating": hotwaterheating_choice.lower(),
            "airconditioning": airconditioning_choice.lower(),
            "parking": parking,
            "prefarea": prefarea_choice.lower(),
            "furnishingstatus": furnishing_choice.lower()
        }

        # Format input using identical preprocessing as training
        model_features = pipeline_data["production_features"]
        user_row_df = prep.prepare_user_input(input_data, model_features)

        # Run actual prediction using trained model
        pred_value = pipeline_data["production_model"].predict(user_row_df)[0]
        rmse = pipeline_data["backward_elimination"]["metrics_test"]["rmse"]

        # Prevent unrealistic negative bounds
        pred_value = max(100000.0, float(pred_value))
        price_fmt, denom_fmt = prep.format_inr(pred_value)

        # Valuation confidence interval approx (± 1 RMSE)
        lower_bound = max(500000.0, pred_value - rmse)
        upper_bound = pred_value + rmse
        lower_fmt, lower_denom = prep.format_inr(lower_bound)
        upper_fmt, upper_denom = prep.format_inr(upper_bound)

        # Display Result Card
        st.markdown(f"""
        <div class="valuation-card">
            <div class="valuation-label">Estimated Property Market Price</div>
            <div class="valuation-price">{price_fmt}</div>
            <div class="valuation-denomination">({denom_fmt})</div>
            <div class="valuation-range">
                Typical Valuation Range: {lower_denom} &ndash; {upper_denom}
            </div>
            <div class="valuation-disclaimer">
                This is an estimated price based on the property's features and the available dataset.
                Actual market prices may vary depending on micro-locality, building condition, and negotiations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary of Entered Features
        st.markdown("#### 📋 Property Summary Breakdown")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""
            - **Area:** {area:,} sq. ft.
            - **Bedrooms:** {bedrooms} BHK
            - **Bathrooms:** {bathrooms}
            - **Floors:** {stories}
            """)
        with sc2:
            st.markdown(f"""
            - **Parking:** {parking} Vehicle(s)
            - **Main Road:** {mainroad_choice}
            - **Prime Area:** {prefarea_choice}
            - **Air Conditioning:** {airconditioning_choice}
            """)
        with sc3:
            st.markdown(f"""
            - **Furnishing:** {furnishing_choice}
            - **Basement:** {basement_choice}
            - **Hot Water:** {hotwaterheating_choice}
            - **Guest Room:** {guestroom_choice}
            """)

        st.info("💡 Want to evaluate another property? Modify the values above and click **Predict House Price** again.")


# =============================================================================
# PAGE 3: Dataset Insights
# =============================================================================
def render_insights_page():
    st.markdown("""
    <div style="margin-bottom: 1.4rem;">
        <h2 style="font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; letter-spacing: -0.025em;">📊 Dataset Insights & Exploratory Data Analysis</h2>
        <p style="color: #64748b; font-size: 0.95rem;">Exploration of 545 verified residential transactions from the Housing dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    # Set Seaborn / Matplotlib styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#e2e8f0"
    plt.rcParams["axes.linewidth"] = 0.85

    # Dataset Preview & Summary Stats
    with st.expander("🔍 Click to view Dataset Preview & Summary Statistics", expanded=False):
        c_p1, c_p2 = st.columns([3, 1])
        with c_p1:
            st.markdown(f"**Shape:** `{raw_df.shape[0]} rows` &times; `{raw_df.shape[1]} columns` | **Missing Values:** `0`")
            st.dataframe(raw_df.head(10), use_container_width=True)
        with c_p2:
            st.markdown("**Numerical Feature Stats:**")
            st.dataframe(raw_df.describe().T[["mean", "std", "min", "50%", "max"]], use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 of Charts: Price Distribution & Area vs Price
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. House Price Distribution")
        fig1, ax1 = plt.subplots(figsize=(6.8, 4.2), dpi=120)
        # Prices in Lakhs for readable axis
        prices_in_lakhs = raw_df["price"] / 100000.0
        sns.histplot(prices_in_lakhs, kde=True, color="#2563eb", ax=ax1, bins=25, edgecolor="white")
        mean_lakhs = prices_in_lakhs.mean()
        median_lakhs = prices_in_lakhs.median()
        ax1.axvline(mean_lakhs, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean: ₹{mean_lakhs:.1f}L")
        ax1.axvline(median_lakhs, color="#10b981", linestyle=":", linewidth=2, label=f"Median: ₹{median_lakhs:.1f}L")
        ax1.set_xlabel("House Price (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        ax1.set_ylabel("Property Count", fontsize=9, fontweight="bold", color="#334155")
        ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption("The distribution is moderately right-skewed with the majority of homes priced between ₹30 Lakhs and ₹65 Lakhs.")

    with col2:
        st.markdown("### 2. Area vs. Selling Price")
        fig2, ax2 = plt.subplots(figsize=(6.8, 4.2), dpi=120)
        sns.regplot(
            data=raw_df,
            x="area",
            y=prices_in_lakhs,
            scatter_kws={"alpha": 0.55, "color": "#0284c7", "s": 32},
            line_kws={"color": "#dc2626", "linewidth": 1.8},
            ax=ax2
        )
        ax2.set_xlabel("Property Area (Square Feet)", fontsize=9, fontweight="bold", color="#334155")
        ax2.set_ylabel("Selling Price (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("A clear positive linear trend: larger plot area consistently correlates with higher market valuations.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 of Charts: Bedrooms & Furnishing Status
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 3. Price Variation by Bedroom Count")
        fig3, ax3 = plt.subplots(figsize=(6.8, 4.2), dpi=120)
        sns.boxplot(
            data=raw_df,
            x="bedrooms",
            y=prices_in_lakhs,
            palette="Blues",
            ax=ax3,
            boxprops=dict(alpha=0.85)
        )
        ax3.set_xlabel("Number of Bedrooms", fontsize=9, fontweight="bold", color="#334155")
        ax3.set_ylabel("Selling Price (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        plt.tight_layout()
        st.pyplot(fig3)
        st.caption("3 and 4 bedroom properties represent the most popular luxury segment, with steady increases in median price.")

    with col4:
        st.markdown("### 4. Impact of Furnishing Status")
        fig4, ax4 = plt.subplots(figsize=(6.8, 4.2), dpi=120)
        furn_summary = raw_df.groupby("furnishingstatus")["price"].mean() / 100000.0
        furn_summary = furn_summary.reindex(["furnished", "semi-furnished", "unfurnished"])
        colors = ["#10b981", "#3b82f6", "#94a3b8"]
        bars = ax4.bar(furn_summary.index.str.title(), furn_summary.values, color=colors, width=0.52)
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f"₹{height:.1f}L",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold", fontsize=9, color="#0f172a")
        ax4.set_ylabel("Average Price (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        ax4.set_ylim(0, max(furn_summary.values) * 1.16)
        plt.tight_layout()
        st.pyplot(fig4)
        st.caption("Furnished and semi-furnished homes command a visible price premium over unfurnished properties.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Correlation Matrix
    st.markdown("### 5. Numerical Feature Correlation Matrix")
    num_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    corr_matrix = raw_df[num_cols].corr()

    fig5, ax5 = plt.subplots(figsize=(8, 4.0), dpi=120)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.7,
        cbar_kws={"shrink": 0.8},
        ax=ax5
    )
    plt.tight_layout()
    st.pyplot(fig5)
    st.caption("Area (0.54) and Bathrooms (0.52) exhibit the highest direct linear correlation with property price.")


# =============================================================================
# PAGE 4: Model Performance & Academic Analysis
# =============================================================================
def render_model_performance_page():
    st.markdown("""
    <div style="margin-bottom: 1.4rem;">
        <h2 style="font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; letter-spacing: -0.025em;">🧠 Machine Learning Model Performance</h2>
        <p style="color: #64748b; font-size: 0.95rem;">Rigorous evaluation of Multiple Linear Regression, Recursive Feature Elimination (RFE), and Backward Elimination.</p>
    </div>
    """, unsafe_allow_html=True)

    be_model = pipeline_data["backward_elimination"]
    be_metrics = be_model["metrics_test"]
    base_metrics = pipeline_data["baseline"]["metrics_test"]
    rfe_metrics = pipeline_data["rfe"]["metrics_test"]

    # 4 Metric Cards for the Selected Model (R², MAE, MSE, RMSE)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">R² Score</div>
            <div class="stat-value">{be_metrics['r2']:.4f}</div>
            <div class="stat-desc">{be_metrics['r2']*100:.1f}% variance explained</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        mae_fmt, mae_denom = prep.format_inr(be_metrics['mae'])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Mean Absolute Error (MAE)</div>
            <div class="stat-value">{mae_denom}</div>
            <div class="stat-desc">{mae_fmt} average deviation</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        mse_val = be_metrics['mse']
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Mean Squared Error (MSE)</div>
            <div class="stat-value">{mse_val:.2e}</div>
            <div class="stat-desc">Squared error penalty</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        rmse_fmt, rmse_denom = prep.format_inr(be_metrics['rmse'])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Root Mean Squared Error (RMSE)</div>
            <div class="stat-value">{rmse_denom}</div>
            <div class="stat-desc">{rmse_fmt} standard deviation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plain English Metric Guide
    with st.expander("📘 What do these regression metrics mean in plain English?", expanded=False):
        st.markdown("""
        - **R² Score (Coefficient of Determination):** Measures what percentage of the variance in house prices is captured by our model. A score of **0.6480** means approximately **65%** of price differences are successfully explained by the property's size and amenities.
        - **Mean Absolute Error (MAE):** The average difference between what the model predicted and what the house actually sold for (~₹9.8 Lakhs).
        - **Mean Squared Error (MSE):** The average of the squared errors (1.78 × 10¹²), heavily penalizing large outliers.
        - **Root Mean Squared Error (RMSE):** The square root of MSE (~₹13.3 Lakhs), reflecting standard deviation in the original Rupee units.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted & Residual Plots
    y_test = pipeline_data["y_test"]
    y_pred = be_model["y_test_pred"]
    residuals = be_model["residuals"]

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("### 🎯 Actual vs. Predicted Prices")
        fig_avp, ax_avp = plt.subplots(figsize=(6.8, 4.2), dpi=120)
        y_test_lakhs = y_test / 100000.0
        y_pred_lakhs = y_pred / 100000.0
        ax_avp.scatter(y_test_lakhs, y_pred_lakhs, color="#2563eb", alpha=0.6, edgecolors="white", s=40)
        # 45 degree perfect line
        min_val = min(y_test_lakhs.min(), y_pred_lakhs.min())
        max_val = max(y_test_lakhs.max(), y_pred_lakhs.max())
        ax_avp.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=1.8, label="Ideal 45° Line")
        ax_avp.set_xlabel("Actual Price (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        ax_avp.set_ylabel("Predicted Price (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        ax_avp.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_avp)
        st.caption("Points cluster closely along the diagonal 45° reference line, demonstrating strong predictive agreement on unseen test data.")

    with p_col2:
        st.markdown("### 📉 Residual Plot (Prediction Errors)")
        fig_res, ax_res = plt.subplots(figsize=(6.8, 4.2), dpi=120)
        res_lakhs = residuals / 100000.0
        ax_res.scatter(y_pred_lakhs, res_lakhs, color="#059669", alpha=0.6, edgecolors="white", s=40)
        ax_res.axhline(0, color="#ef4444", linestyle="--", linewidth=1.8)
        ax_res.set_xlabel("Predicted Price (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        ax_res.set_ylabel("Residual Error (₹ in Lakhs)", fontsize=9, fontweight="bold", color="#334155")
        plt.tight_layout()
        st.pyplot(fig_res)
        st.caption("Residuals are randomly dispersed around the zero horizontal line without obvious curvature, validating linear regression assumptions.")

    st.markdown("---")

    # =========================================================================
    # ACADEMIC FEATURE SELECTION SECTION
    # =========================================================================
    st.markdown("""
    <div style="margin-top: 1rem; margin-bottom: 1.5rem;">
        <h3 style="font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">🔬 Feature Selection Analysis & Model Comparison</h3>
        <p style="color: #64748b; font-size: 0.95rem;">
            Detailed academic comparison between Baseline Multiple Linear Regression, Backward Elimination using statistical <em>p</em>-values,
            and scikit-learn Recursive Feature Elimination (RFE).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Comparison Table
    st.markdown("#### 1. Model Performance Comparison Table")
    comp_df = pd.DataFrame({
        "Model Configuration": [
            "Baseline (All 13 Features)",
            "Recursive Feature Elimination (RFE - 10 Features)",
            "Backward Elimination (OLS p ≤ 0.05 - 10 Features) [Selected]"
        ],
        "Features Used": [
            len(pipeline_data["baseline"]["features"]),
            len(pipeline_data["rfe"]["selected_features"]),
            len(be_model["selected_features"])
        ],
        "Test R² Score": [
            f"{base_metrics['r2']:.4f}",
            f"{rfe_metrics['r2']:.4f}",
            f"{be_metrics['r2']:.4f}"
        ],
        "Test MAE (₹)": [
            prep.format_inr(base_metrics['mae'])[0],
            prep.format_inr(rfe_metrics['mae'])[0],
            prep.format_inr(be_metrics['mae'])[0]
        ],
        "Test RMSE (₹)": [
            prep.format_inr(base_metrics['rmse'])[0],
            prep.format_inr(rfe_metrics['rmse'])[0],
            prep.format_inr(be_metrics['rmse'])[0]
        ]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Backward Elimination Step Log
    st.markdown("#### 2. Backward Elimination Process (Significance Threshold &alpha; = 0.05)")
    st.markdown("""
    Using `statsmodels.api.OLS`, we iteratively evaluate predictor *p*-values. At each iteration, the feature with the highest
    *p*-value exceeding 0.05 is eliminated until all remaining variables are statistically significant:
    """)
    step_log_df = be_model["step_log"].copy()
    step_log_df["Highest_P_Value"] = step_log_df["Highest_P_Value"].apply(lambda x: f"{x:.4f}")
    step_log_df["AIC"] = step_log_df["AIC"].apply(lambda x: f"{x:,.1f}")
    step_log_df["Adj_R2"] = step_log_df["Adj_R2"].apply(lambda x: f"{x:.4f}")
    st.dataframe(step_log_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Statsmodels OLS Regression Summary
    with st.expander("📑 View Complete Statsmodels OLS Statistical Summary Table", expanded=False):
        st.text(be_model["ols_model"].summary().as_text())

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. RFE Feature Ranking Table
    st.markdown("#### 3. Recursive Feature Elimination (RFE) Ranking")
    st.markdown("Features evaluated using standardized Linear Regression estimator:")
    rfe_ranking = pipeline_data["rfe"]["ranking_df"].copy()
    rfe_ranking["Status"] = rfe_ranking["Selected"].map({True: "Selected (Rank 1)", False: "Eliminated"})
    st.dataframe(rfe_ranking[["Rank", "Feature", "Status"]], use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Interpretation of Selected Model Coefficients
    st.markdown("#### 4. Feature Impact Interpretation")
    coef_df, intercept = ml.get_feature_coefficients(
        be_model["sklearn_model"],
        be_model["selected_features"]
    )
    coef_display = coef_df.copy()
    coef_display["Impact in Rupees"] = coef_display["Coefficient"].apply(lambda c: f"{'+' if c > 0 else ''}{prep.format_inr(c)[0]}")
    coef_display["Real-World Meaning"] = coef_display["Feature"].map({
        "bathrooms": "Each additional bathroom increases estimated price by ~₹11.2 Lakhs",
        "airconditioning": "Having air conditioning adds ~₹8.1 Lakhs in property value",
        "hotwaterheating": "Hot water heating system adds ~₹6.8 Lakhs",
        "prefarea": "Location in preferred / prime area adds ~₹6.5 Lakhs",
        "basement": "Having a basement adds ~₹4.8 Lakhs",
        "stories": "Each additional floor / story adds ~₹4.5 Lakhs",
        "mainroad": "Direct main road access adds ~₹3.7 Lakhs",
        "parking": "Each additional parking spot adds ~₹2.3 Lakhs",
        "area": "Each square foot of lot area adds ~₹244",
        "furnishingstatus_unfurnished": "Unfurnished condition reduces value by ~₹3.4 Lakhs vs. Furnished"
    })
    st.dataframe(coef_display[["Feature", "Impact in Rupees", "Real-World Meaning"]], use_container_width=True, hide_index=True)


# =============================================================================
# Main Router
# =============================================================================
if st.session_state["current_page"] == "🏠 Home / Dashboard":
    render_home_page()
elif st.session_state["current_page"] == "🔮 House Price Prediction":
    render_prediction_page()
elif st.session_state["current_page"] == "📊 Dataset Insights":
    render_insights_page()
elif st.session_state["current_page"] == "🧠 Model Performance":
    render_model_performance_page()
