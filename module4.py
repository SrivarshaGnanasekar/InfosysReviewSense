import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud
import numpy as np

st.set_page_config(
    page_title="ReviewSense Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { 
        background-color: #262730; 
        padding: 1.2rem; 
        border-radius: 10px; 
        text-align: center; 
        border: 1px solid #464b5d;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Module2_Sentiment_Results_new.csv") 
        df['sentiment'] = df['sentiment'].str.capitalize()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading Primary Data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_keywords():
    try:
        return pd.read_csv("Module3_Keyword_Insights.csv")
    except Exception as e:
        st.error(f"Error loading Keyword Data: {e}")
        return pd.DataFrame()

df = load_data()
keywords_df = load_keywords()

st.sidebar.header("🔍 Dashboard Filters")

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=["Positive", "Negative", "Neutral"],
    default=["Positive", "Negative", "Neutral"]
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    options=sorted(df["product"].unique()) if not df.empty else [],
    default=sorted(df["product"].unique()) if not df.empty else []
)

st.sidebar.subheader("📅 Date Range")
if not df.empty:
    min_d = df['date'].min().date()
    max_d = df['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", value=min_d)
    end_date = st.sidebar.date_input("End Date", value=max_d)
else:
    start_date, end_date = datetime.now(), datetime.now()


if not df.empty:
    filtered_df = df[
        (df["sentiment"].isin(sentiment_filter)) &
        (df["product"].isin(product_filter)) &
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date)
    ].copy()
else:
    filtered_df = pd.DataFrame()

st.markdown('<h1 class="main-header">📊 ReviewSense – Customer Feedback Dashboard</h1>', unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("No data found. Please adjust your filters or check your CSV files.")
else:
    total = len(filtered_df)
    pos = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
    neg = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
    neu = len(filtered_df[filtered_df['sentiment'] == 'Neutral'])

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Reviews", total)
    with c2: st.metric("Positive %", f"{(pos/total*100):.1f}%", f"{pos} reviews")
    with c3: st.metric("Negative %", f"{(neg/total*100):.1f}%", f"{neg} reviews")
    with c4: st.metric("Neutral %", f"{(neu/total*100):.1f}%", f"{neu} reviews")

    st.subheader("😊 Overall Sentiment Breakdown")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    counts = filtered_df["sentiment"].value_counts()
    colors = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#9E9E9E'}
    sns.barplot(x=counts.index, y=counts.values, palette=[colors.get(x) for x in counts.index], ax=ax_bar)
    ax_bar.set_ylabel("Number of Reviews")
    st.pyplot(fig_bar)

    st.subheader("📱 Product-wise Performance")
    col_table, col_heat = st.columns([1, 1])
    
    product_stats = filtered_df.groupby('product')['sentiment'].value_counts().unstack(fill_value=0)
    
    with col_table:
        st.write("Summary Table")
        st.dataframe(product_stats, use_container_width=True)

    with col_heat:
        st.write("Sentiment Heatmap")
        fig_hm, ax_hm = plt.subplots()
        sns.heatmap(product_stats, annot=True, fmt="d", cmap="YlGnBu", ax=ax_hm)
        st.pyplot(fig_hm)

    st.subheader("📈 Monthly Sentiment Trends")
    filtered_df['Month'] = filtered_df['date'].dt.to_period('M').astype(str)
    trend_data = filtered_df.groupby(['Month', 'sentiment']).size().unstack(fill_value=0)
    st.line_chart(trend_data)

    st.subheader("🔑 Keyword Insights")
    if not keywords_df.empty:
        col_kw1, col_kw2 = st.columns(2)
        with col_kw1:
            st.write("Top Keyword Frequencies")
            st.bar_chart(keywords_df.set_index('keyword').head(12))
        with col_kw2:
            st.write("Keyword Word Cloud")
            word_freq = dict(zip(keywords_df['keyword'], keywords_df['frequency']))
            wc = WordCloud(width=800, height=450, background_color='black', colormap='viridis').generate_from_frequencies(word_freq)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

    st.subheader("🎯 Sentiment Confidence Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered_df['confidence_score'], bins=20, kde=True, color='skyblue', ax=ax_hist)
    ax_hist.set_xlabel("Confidence Score")
    st.pyplot(fig_hist)

    with st.expander("📋 View Filtered Data Preview"):
        st.dataframe(filtered_df.head(15), use_container_width=True)

    st.subheader("💾 Export Data")
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button("⬇️ Download Filtered Reviews", filtered_df.to_csv(index=False), "Filtered_Reviews.csv", "text/csv")
    with c_dl2:
        if not keywords_df.empty:
            st.download_button("⬇️ Download Keyword Insights", keywords_df.to_csv(index=False), "Keyword_Insights.csv", "text/csv")

st.success("Dashboard ready! ")