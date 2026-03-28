import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud
import numpy as np
import sqlite3
from textblob import TextBlob

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="ReviewSense • AI Feedback Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- GLOBAL STYLES --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080B14;
    color: #E8EAF0;
}

[data-testid="stAppViewContainer"] {
    background: #080B14;
}

[data-testid="stHeader"] { background: transparent; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D1120; }
::-webkit-scrollbar-thumb { background: #2A3556; border-radius: 2px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1120 0%, #0A0F1E 100%);
    border-right: 1px solid rgba(99,120,195,0.15);
}

[data-testid="stSidebar"] .block-container { padding: 2rem 1.2rem; }

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 2rem 0;
    border-bottom: 1px solid rgba(99,120,195,0.15);
    margin-bottom: 2rem;
}

.sidebar-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #4F6EF7, #9B5CF6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}

.sidebar-logo-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    color: #E8EAF0;
    letter-spacing: -0.02em;
}

.sidebar-logo-badge {
    font-size: 0.6rem;
    color: #6378C3;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    display: block;
    margin-top: -2px;
}

/* ── Sidebar Radio ── */
[data-testid="stRadio"] > div {
    gap: 6px !important;
    flex-direction: column;
}

[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
    font-size: 0.875rem !important;
    color: #8B93B8 !important;
    cursor: pointer;
}

[data-testid="stRadio"] label:hover {
    background: rgba(79,110,247,0.08) !important;
    border-color: rgba(79,110,247,0.2) !important;
    color: #E8EAF0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4F6EF7 0%, #9B5CF6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(79,110,247,0.25);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(79,110,247,0.4);
    opacity: 0.95;
}

.stButton > button:active { transform: translateY(0); }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: #111827 !important;
    border: 1px solid rgba(99,120,195,0.25) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease;
    padding: 0.6rem 1rem !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(79,110,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(79,110,247,0.12) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0F1526;
    border: 1px solid rgba(99,120,195,0.15);
    border-radius: 14px;
    padding: 1.4rem 1.5rem !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(79,110,247,0.35);
    transform: translateY(-2px);
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4F6EF7, #9B5CF6);
    border-radius: 14px 14px 0 0;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6378C3 !important;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #E8EAF0 !important;
    line-height: 1.1 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    color: #6378C3 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1120 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,120,195,0.15) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    padding: 8px 20px !important;
    color: #8B93B8 !important;
    font-size: 0.875rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F6EF7, #9B5CF6) !important;
    color: white !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,120,195,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0F1526 !important;
    border: 1px solid rgba(99,120,195,0.15) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    color: #8B93B8 !important;
}

/* ── Select ── */
[data-testid="stMultiSelect"] > div > div,
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid rgba(99,120,195,0.25) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
}

/* ── Download Button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid rgba(79,110,247,0.4) !important;
    color: #7B93F5 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: none !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: rgba(79,110,247,0.1) !important;
    border-color: rgba(79,110,247,0.7) !important;
    transform: translateY(-1px) !important;
}

/* ── Chart containers ── */
.chart-card {
    background: #0F1526;
    border: 1px solid rgba(99,120,195,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #C8CEDE;
    letter-spacing: -0.01em;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99,120,195,0.15);
    margin-left: 8px;
}

/* ── Alert / Success ── */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.875rem !important;
}

/* ── Sidebar filter labels ── */
.sidebar-filter-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6378C3;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Page title block ── */
.page-title-block {
    margin-bottom: 2.5rem;
}

.page-title-block h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #E8EAF0;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem 0;
    line-height: 1.15;
}

.page-title-block p {
    color: #6378C3;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 300;
}

/* ── Sentiment badge ── */
.sentiment-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-positive { background: rgba(34,197,94,0.12); color: #4ADE80; border: 1px solid rgba(34,197,94,0.25); }
.badge-negative { background: rgba(239,68,68,0.12); color: #F87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-neutral  { background: rgba(148,163,184,0.12); color: #94A3B8; border: 1px solid rgba(148,163,184,0.25); }

/* ── Stat row ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
}

.stat-pill {
    flex: 1;
    background: #111827;
    border: 1px solid rgba(99,120,195,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}

.stat-pill-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #E8EAF0;
    display: block;
}

.stat-pill-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6378C3;
    font-weight: 500;
}

/* ── Login card ── */
.auth-card {
    background: #0F1526;
    border: 1px solid rgba(99,120,195,0.2);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    box-shadow: 0 30px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(79,110,247,0.05);
}

/* ── Landing hero ── */
.hero-block {
    background: linear-gradient(135deg, #0D1528 0%, #111E40 50%, #0D1528 100%);
    border: 1px solid rgba(79,110,247,0.2);
    border-radius: 20px;
    padding: 80px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 3rem;
}

.hero-block::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 300px;
    background: radial-gradient(ellipse, rgba(79,110,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-block;
    background: rgba(79,110,247,0.12);
    border: 1px solid rgba(79,110,247,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7B93F5;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.04em;
    color: #E8EAF0;
    margin: 0 0 1rem 0;
}

.hero-title span {
    background: linear-gradient(135deg, #4F6EF7, #9B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #6B7FA8;
    font-weight: 300;
    margin: 0;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Feature cards ── */
.feature-card {
    background: #0F1526;
    border: 1px solid rgba(99,120,195,0.15);
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(79,110,247,0.5), transparent);
    opacity: 0;
    transition: opacity 0.25s ease;
}

.feature-card:hover::before { opacity: 1; }

.feature-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    margin: 0 auto 1rem;
}

.fi-blue  { background: rgba(79,110,247,0.12); border: 1px solid rgba(79,110,247,0.2); }
.fi-purple{ background: rgba(155,92,246,0.12); border: 1px solid rgba(155,92,246,0.2); }
.fi-teal  { background: rgba(20,184,166,0.12); border: 1px solid rgba(20,184,166,0.2); }
.fi-amber { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.2); }

.feature-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #C8CEDE;
    margin: 0 0 0.5rem 0;
}

.feature-desc {
    font-size: 0.82rem;
    color: #6378C3;
    line-height: 1.5;
    margin: 0;
}

/* ── User avatar chip ── */
.user-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(79,110,247,0.08);
    border: 1px solid rgba(79,110,247,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 2rem;
}

.user-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F6EF7, #9B5CF6);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}

.user-name {
    font-size: 0.85rem;
    font-weight: 500;
    color: #C8CEDE;
}

.user-role {
    font-size: 0.7rem;
    color: #6378C3;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Insight row ── */
.insight-card {
    background: #0F1526;
    border: 1px solid rgba(99,120,195,0.15);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.insight-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.8rem;
}

.insight-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6378C3;
}

/* ── Progress bar ── */
.progress-container { margin: 0.5rem 0; }
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 5px;
    color: #8B93B8;
}
.progress-track {
    height: 6px;
    background: rgba(99,120,195,0.12);
    border-radius: 6px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* ── Keyword chip ── */
.keyword-chip {
    display: inline-block;
    background: rgba(79,110,247,0.08);
    border: 1px solid rgba(79,110,247,0.2);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #7B93F5;
    margin: 3px;
}

/* Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0F1526',
    'axes.facecolor': '#0F1526',
    'axes.edgecolor': '#1E2A45',
    'axes.labelcolor': '#8B93B8',
    'xtick.color': '#8B93B8',
    'ytick.color': '#8B93B8',
    'text.color': '#C8CEDE',
    'grid.color': '#1A2540',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 10,
})

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
conn = sqlite3.connect("reviews.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT
)
""")
conn.commit()
cursor.execute("INSERT OR IGNORE INTO users VALUES ('admin','admin123','admin')")
conn.commit()


# ─────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────
def landing_page():
    col_left, col_right = st.columns([8, 2])
    with col_right:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                st.session_state["page"] = "login"
                st.rerun()
        with c2:
            if st.button("Register", use_container_width=True):
                st.session_state["page"] = "login"
                st.rerun()

    st.markdown("""
    <div class="hero-block">
        <div class="hero-eyebrow">◈ AI-Powered Analytics Platform</div>
        <h1 class="hero-title">Turn Reviews Into<br><span>Revenue Intelligence</span></h1>
        <p class="hero-subtitle">Advanced sentiment analysis and NLP insights that transform raw customer feedback into strategic decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Platform Capabilities</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("fi-blue",   "😊", "Sentiment Detection", "Real-time polarity scoring with confidence metrics across positive, negative, and neutral classes."),
        ("fi-purple", "🧠", "Deep NLP Analysis",   "Entity extraction, noun phrase detection, and subjectivity scoring powered by TextBlob."),
        ("fi-teal",   "📈", "Trend Intelligence",  "Monthly sentiment shifts and product performance heatmaps for longitudinal analysis."),
        ("fi-amber",  "☁️", "Keyword Mapping",     "Frequency-weighted word clouds and top-term charts reveal what customers talk about most."),
    ]
    for col, (cls, icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon {cls}">{icon}</div>
                <p class="feature-title">{title}</p>
                <p class="feature-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Why ReviewSense?</p>', unsafe_allow_html=True)

    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown("""
        <div class="insight-card">
            <p style="color:#8B93B8; font-size:0.92rem; line-height:1.75; margin:0;">
            ReviewSense helps product teams, marketing analysts, and customer success managers make sense of 
            thousands of reviews instantly. Instead of reading every piece of feedback manually, our AI pipeline 
            extracts polarity, emotion, key themes, and statistical patterns — delivering the signal you need 
            to act, without the noise.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        for label, pct, color in [("Accuracy", 94, "#4F6EF7"), ("Speed", 99, "#9B5CF6"), ("Coverage", 87, "#14B8A6")]:
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-label"><span>{label}</span><span>{pct}%</span></div>
                <div class="progress-track">
                    <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTH PAGE
# ─────────────────────────────────────────────
def auth_page():
    st.markdown("""
    <div style="text-align:center; margin: 3rem 0 2rem;">
        <div style="font-size:2rem; font-family:'Syne',sans-serif; font-weight:800; letter-spacing:-0.03em; color:#E8EAF0;">
            ◈ ReviewSense
        </div>
        <div style="color:#6378C3; font-size:0.85rem; margin-top:4px;">Sign in to continue to your workspace</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="Enter your username", key="li_user")
            p = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", use_container_width=True, key="login_btn"):
                user = cursor.execute(
                    "SELECT * FROM users WHERE username=? AND password=?", (u, p)
                ).fetchone()
                if user:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = u
                    st.session_state["page"] = "app"
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            u2 = st.text_input("Choose a username", placeholder="username", key="reg_user")
            p2 = st.text_input("Choose a password", type="password", placeholder="password", key="reg_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, key="reg_btn"):
                if cursor.execute("SELECT * FROM users WHERE username=?", (u2,)).fetchone():
                    st.warning("Username already taken.")
                else:
                    cursor.execute("INSERT INTO users VALUES (?,?,?)", (u2, p2, "user"))
                    conn.commit()
                    st.success("Account created! Switch to Sign In.")

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ANALYZE PAGE
# ─────────────────────────────────────────────
def analyze_review_page():
    st.markdown("""
    <div class="page-title-block">
        <h1>Analyze Review</h1>
        <p>Paste any customer review to extract sentiment, emotion, and key insights in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Input</p>', unsafe_allow_html=True)
        review = st.text_area(
            "Customer Review",
            placeholder="Paste a customer review here...",
            height=200,
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("Run Analysis →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if analyze and review:
            blob = TextBlob(review)
            polarity    = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0:   sentiment, badge_cls, emoji = "Positive", "badge-positive", "↑"
            elif polarity < 0: sentiment, badge_cls, emoji = "Negative", "badge-negative", "↓"
            else:               sentiment, badge_cls, emoji = "Neutral",  "badge-neutral",  "→"

            if polarity > 0.3:   emotion = "😊 Happy"
            elif polarity < -0.3: emotion = "😠 Frustrated"
            else:                  emotion = "😐 Neutral"

            noun_phrases = list(blob.noun_phrases[:6])
            words        = blob.words
            word_freq    = pd.Series(words).value_counts().head(8)
            word_count   = len(words)
            char_count   = len(review)

            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-header">
                    <span class="insight-title">Sentiment Result</span>
                    <span class="sentiment-badge {badge_cls}">{emoji} {sentiment}</span>
                </div>
                <div class="stat-row">
                    <div class="stat-pill">
                        <span class="stat-pill-value">{polarity:+.2f}</span>
                        <span class="stat-pill-label">Polarity</span>
                    </div>
                    <div class="stat-pill">
                        <span class="stat-pill-value">{subjectivity:.2f}</span>
                        <span class="stat-pill-label">Subjectivity</span>
                    </div>
                    <div class="stat-pill">
                        <span class="stat-pill-value">{word_count}</span>
                        <span class="stat-pill-label">Words</span>
                    </div>
                    <div class="stat-pill">
                        <span class="stat-pill-value">{char_count}</span>
                        <span class="stat-pill-label">Chars</span>
                    </div>
                </div>
                <div style="margin-top:1rem;">
                    <span class="insight-title" style="font-size:0.75rem;">DETECTED EMOTION</span>
                    <p style="color:#C8CEDE; font-size:1.1rem; margin:6px 0 1rem;">{emotion}</p>
                    <span class="insight-title" style="font-size:0.75rem;">KEY PHRASES</span>
                    <div style="margin-top:6px;">
                        {''.join(f'<span class="keyword-chip">{kp}</span>' for kp in noun_phrases) if noun_phrases else '<span style="color:#6378C3;font-size:0.82rem;">None detected</span>'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-header" style="margin-top:1.5rem;">Top Word Frequencies</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3))
            bars = ax.barh(word_freq.index[::-1], word_freq.values[::-1],
                           color=[f"#{int(79+i*20):02X}{int(110+i*5):02X}F7" for i in range(len(word_freq))])
            ax.set_xlabel("Frequency")
            for bar, val in zip(bars, word_freq.values[::-1]):
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, str(val),
                        va='center', color='#8B93B8', fontsize=9)
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.set_xlim(0, word_freq.values.max() * 1.2)
            fig.tight_layout()
            st.markdown('<div class="chart-card" style="padding:1rem;">', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        elif analyze and not review:
            st.warning("Please enter a review to analyze.")
        else:
            st.markdown("""
            <div style="background:#0F1526; border:1px dashed rgba(99,120,195,0.2); border-radius:14px; padding:3rem; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.8rem;">🔍</div>
                <p style="color:#6378C3; font-size:0.875rem; margin:0;">Enter a review and click <strong style="color:#8B93B8;">Run Analysis</strong> to see results here.</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION ROUTING
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

if st.session_state["page"] == "landing":
    landing_page()
    st.stop()

if "logged_in" not in st.session_state:
    auth_page()
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR (Post-login)
# ─────────────────────────────────────────────
username = st.session_state.get("username", "User")
initials = username[:2].upper()

st.sidebar.markdown(f"""
<div class="sidebar-logo">
    <div class="sidebar-logo-icon">◈</div>
    <div>
        <div class="sidebar-logo-text">ReviewSense</div>
        <span class="sidebar-logo-badge">Analytics Platform</span>
    </div>
</div>
<div class="user-chip">
    <div class="user-avatar">{initials}</div>
    <div>
        <div class="user-name">{username}</div>
        <div class="user-role">Analyst</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<span class="sidebar-filter-label">Navigation</span>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Dashboard", "Analyze Review"], label_visibility="collapsed")


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
if page == "Dashboard":

    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("Module2_Sentiment_Results_new.csv")
            df['sentiment'] = df['sentiment'].str.capitalize()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()

    @st.cache_data
    def load_keywords():
        try:
            return pd.read_csv("Module3_Keyword_Insights.csv")
        except Exception as e:
            st.error(f"Error loading keywords: {e}")
            return pd.DataFrame()

    df = load_data()
    keywords_df = load_keywords()

    # ── Sidebar filters ──
    st.sidebar.markdown("---")
    st.sidebar.markdown('<span class="sidebar-filter-label">Filters</span>', unsafe_allow_html=True)

    sentiment_filter = st.sidebar.multiselect(
        "Sentiment", options=["Positive", "Negative", "Neutral"],
        default=["Positive", "Negative", "Neutral"]
    )
    product_filter = st.sidebar.multiselect(
        "Product",
        options=sorted(df["product"].unique()) if not df.empty else [],
        default=sorted(df["product"].unique()) if not df.empty else []
    )

    st.sidebar.markdown('<span class="sidebar-filter-label" style="margin-top:0.5rem; display:block;">Date Range</span>', unsafe_allow_html=True)
    if not df.empty:
        min_d = df['date'].min().date()
        max_d = df['date'].max().date()
        start_date = st.sidebar.date_input("From", value=min_d)
        end_date   = st.sidebar.date_input("To",   value=max_d)
    else:
        start_date = end_date = datetime.now()

    if not df.empty:
        filtered_df = df[
            (df["sentiment"].isin(sentiment_filter)) &
            (df["product"].isin(product_filter)) &
            (df["date"].dt.date >= start_date) &
            (df["date"].dt.date <= end_date)
        ].copy()
    else:
        filtered_df = pd.DataFrame()

    # ── Page Header ──
    st.markdown("""
    <div class="page-title-block">
        <h1>Feedback Dashboard</h1>
        <p>Customer sentiment analytics across all products and time ranges.</p>
    </div>
    """, unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No data found. Adjust your filters or check your CSV files.")
    else:
        total = len(filtered_df)
        pos   = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
        neg   = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
        neu   = len(filtered_df[filtered_df['sentiment'] == 'Neutral'])

        # ── KPI Row ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews",   f"{total:,}")
        c2.metric("Positive",  f"{pos/total*100:.1f}%",  f"{pos:,} reviews")
        c3.metric("Negative",  f"{neg/total*100:.1f}%",  f"{neg:,} reviews")
        c4.metric("Neutral",   f"{neu/total*100:.1f}%",  f"{neu:,} reviews")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sentiment Breakdown ──
        col_bar, col_pie = st.columns([3, 2], gap="large")

        with col_bar:
            st.markdown('<p class="section-header">Sentiment Distribution</p>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                counts = filtered_df["sentiment"].value_counts()
                palette = {'Positive': '#4ADE80', 'Negative': '#F87171', 'Neutral': '#94A3B8'}
                fig, ax = plt.subplots(figsize=(7, 3.5))
                bars = ax.bar(counts.index, counts.values,
                              color=[palette.get(s, '#4F6EF7') for s in counts.index],
                              width=0.5, zorder=2)
                ax.set_ylabel("Reviews", fontsize=9)
                ax.yaxis.grid(True, zorder=0)
                ax.spines[['top','right','left','bottom']].set_visible(False)
                ax.tick_params(axis='x', length=0)
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                            f"{val:,}", ha='center', va='bottom', fontsize=9, color='#8B93B8')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_pie:
            st.markdown('<p class="section-header">Sentiment Share</p>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                sizes  = [pos, neg, neu]
                colors = ['#4ADE80', '#F87171', '#94A3B8']
                labels = ['Positive', 'Negative', 'Neutral']
                wedges, texts, autotexts = ax2.pie(
                    sizes, labels=None, colors=colors,
                    autopct='%1.0f%%', startangle=90,
                    wedgeprops=dict(width=0.65, edgecolor='#0F1526', linewidth=2),
                    pctdistance=0.75
                )
                for t in autotexts: t.set(color='#0F1526', fontsize=9, fontweight='bold')
                legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
                ax2.legend(handles=legend_patches, loc='lower center', ncol=3,
                           bbox_to_anchor=(0.5, -0.08), frameon=False,
                           fontsize=8, labelcolor='#8B93B8')
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Product Performance ──
        st.markdown('<p class="section-header" style="margin-top:0.5rem;">Product Performance</p>', unsafe_allow_html=True)
        col_table, col_heat = st.columns([1, 1], gap="large")
        product_stats = filtered_df.groupby('product')['sentiment'].value_counts().unstack(fill_value=0)

        with col_table:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#6378C3; margin:0 0 0.8rem;">Summary Table</p>', unsafe_allow_html=True)
            st.dataframe(product_stats, use_container_width=True, height=220)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_heat:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#6378C3; margin:0 0 0.8rem;">Sentiment Heatmap</p>', unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(product_stats, annot=True, fmt="d",
                        cmap=sns.color_palette("mako", as_cmap=True),
                        linewidths=1, linecolor='#080B14',
                        ax=ax3, cbar_kws={"shrink": 0.7})
            ax3.tick_params(axis='both', length=0, labelsize=8)
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Monthly Trends ──
        st.markdown('<p class="section-header">Monthly Sentiment Trends</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        filtered_df['Month'] = filtered_df['date'].dt.to_period('M').astype(str)
        trend_data = filtered_df.groupby(['Month', 'sentiment']).size().unstack(fill_value=0)
        
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        colors_trend = {'Positive': '#4ADE80', 'Negative': '#F87171', 'Neutral': '#94A3B8'}
        for col_name in trend_data.columns:
            ax4.plot(trend_data.index, trend_data[col_name],
                     color=colors_trend.get(col_name, '#4F6EF7'),
                     linewidth=2, marker='o', markersize=4, label=col_name,
                     alpha=0.9)
            ax4.fill_between(trend_data.index, trend_data[col_name], alpha=0.07,
                             color=colors_trend.get(col_name, '#4F6EF7'))
        ax4.legend(frameon=False, fontsize=9, labelcolor='#8B93B8')
        ax4.spines[['top','right']].set_visible(False)
        ax4.set_xlabel("")
        ax4.set_ylabel("Review Count", fontsize=9)
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.yaxis.grid(True)
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Keywords ──
        if not keywords_df.empty:
            st.markdown('<p class="section-header">Keyword Insights</p>', unsafe_allow_html=True)
            col_kw1, col_kw2 = st.columns([1, 1], gap="large")

            with col_kw1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<p style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#6378C3; margin:0 0 0.8rem;">Top Keywords</p>', unsafe_allow_html=True)
                top_kw = keywords_df.head(12)
                fig5, ax5 = plt.subplots(figsize=(7, 4))
                cmap = plt.cm.get_cmap('cool', len(top_kw))
                colors_kw = [cmap(i) for i in range(len(top_kw))]
                bars5 = ax5.barh(top_kw['keyword'][::-1], top_kw['frequency'][::-1],
                                  color=colors_kw[::-1], zorder=2)
                ax5.spines[['top','right','left','bottom']].set_visible(False)
                ax5.tick_params(axis='y', length=0, labelsize=8)
                ax5.xaxis.grid(True, zorder=0)
                fig5.tight_layout()
                st.pyplot(fig5, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_kw2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<p style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#6378C3; margin:0 0 0.8rem;">Word Cloud</p>', unsafe_allow_html=True)
                word_freq_dict = dict(zip(keywords_df['keyword'], keywords_df['frequency']))
                wc = WordCloud(
                    width=800, height=420,
                    background_color='#0F1526',
                    colormap='cool',
                    max_words=80,
                    prefer_horizontal=0.85
                ).generate_from_frequencies(word_freq_dict)
                fig6, ax6 = plt.subplots(figsize=(7, 4))
                ax6.imshow(wc, interpolation='bilinear')
                ax6.axis('off')
                fig6.tight_layout(pad=0)
                st.pyplot(fig6, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Confidence Distribution ──
        st.markdown('<p class="section-header">Confidence Score Distribution</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig7, ax7 = plt.subplots(figsize=(12, 3.5))
        sns.histplot(filtered_df['confidence_score'], bins=25, kde=True,
                     color='#4F6EF7', alpha=0.5, ax=ax7, line_kws={'linewidth': 2, 'color': '#9B5CF6'})
        ax7.set_xlabel("Confidence Score", fontsize=9)
        ax7.set_ylabel("Count", fontsize=9)
        ax7.spines[['top','right']].set_visible(False)
        ax7.yaxis.grid(True)
        fig7.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Data Preview ──
        with st.expander("📋 View Filtered Data Preview"):
            st.dataframe(filtered_df.head(20), use_container_width=True)

        # ── Export ──
        st.markdown('<p class="section-header" style="margin-top:0.5rem;">Export</p>', unsafe_allow_html=True)
        c_dl1, c_dl2, _ = st.columns([1, 1, 2])
        with c_dl1:
            st.download_button("⬇ Filtered Reviews", filtered_df.to_csv(index=False),
                               "Filtered_Reviews.csv", "text/csv", use_container_width=True)
        with c_dl2:
            if not keywords_df.empty:
                st.download_button("⬇ Keyword Insights", keywords_df.to_csv(index=False),
                                   "Keyword_Insights.csv", "text/csv", use_container_width=True)

# ─────────────────────────────────────────────
# ANALYZE PAGE
# ─────────────────────────────────────────────
elif page == "Analyze Review":
    analyze_review_page()

# ─────────────────────────────────────────────
# SIDEBAR LOGOUT
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("Sign Out", use_container_width=True):
    st.session_state.clear()
    st.rerun()
