import os
import sys
import time
import threading
import pandas as pd
import yaml
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Adjust path to import from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from database_manager import DBManager
from sentiment_engine import SentimentEngine
from data_generator import DataGenerator

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Social Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Dark CSS theme injection for rich aesthetics
st.markdown("""
    <style>
        .main {
            background-color: #0f111a;
            color: #ffffff;
        }
        .stMetric {
            background-color: #1e2235;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border-left: 5px solid #6366f1;
        }
        .stMetric div[data-testid="metric-container"] {
            color: #ffffff;
        }
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            color: #ffffff !important;
        }
        .sentiment-positive {
            color: #10b981;
            font-weight: bold;
        }
        .sentiment-negative {
            color: #ef4444;
            font-weight: bold;
        }
        .sentiment-neutral {
            color: #9ca3af;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------

# Load Config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

db = DBManager()
db.initialize_db()

# Initialize Sentiment Engine in Session State (to avoid reloading weights on every action)
if "active_model_name" not in st.session_state:
    st.session_state.active_model_name = config.get("active_model", "tfidf_logistic")

if "sentiment_engine" not in st.session_state:
    st.session_state.sentiment_engine = SentimentEngine()

# Thread control for Data Generator Simulator
if "simulator_running" not in st.session_state:
    st.session_state.simulator_running = False
if "stop_simulator_event" not in st.session_state:
    st.session_state.stop_simulator_event = threading.Event()
if "simulator_thread" not in st.session_state:
    st.session_state.simulator_thread = None

# ----------------- SIMULATOR WORKER THREAD -----------------

def run_simulator_loop(stop_event, delay):
    """Background thread target that simulates social media stream writes."""
    gen = DataGenerator()
    while not stop_event.is_set():
        post = gen.generate_single_post()
        db.insert_raw_posts([post])
        time.sleep(delay)

def toggle_simulator():
    """Starts or stops the background data generation thread."""
    if st.session_state.simulator_running:
        # Stop
        st.session_state.stop_simulator_event.set()
        if st.session_state.simulator_thread:
            st.session_state.simulator_thread.join(timeout=1.0)
        st.session_state.simulator_running = False
        st.toast("Real-time Simulator Stopped", icon="⏹️")
    else:
        # Start
        st.session_state.stop_simulator_event.clear()
        delay = config["simulation"].get("delay_seconds", 1.5)
        st.session_state.simulator_thread = threading.Thread(
            target=run_simulator_loop, 
            args=(st.session_state.stop_simulator_event, delay),
            daemon=True
        )
        st.session_state.simulator_thread.start()
        st.session_state.simulator_running = True
        st.toast("Real-time Simulator Started", icon="🚀")

# ----------------- BACKGROUND INGESTION ENGINE -----------------

def run_sentiment_pipeline():
    """Pulls unprocessed tweets from SQLite, runs active model predictions, and updates SQLite."""
    unprocessed = db.get_unprocessed_posts(limit=15)
    if not unprocessed:
        return
        
    engine = st.session_state.sentiment_engine
    for post in unprocessed:
        pred = engine.predict(post["text"])
        db.update_post_sentiment(
            post["id"], 
            pred["label"], 
            pred["score"], 
            engine.model_type
        )

# ----------------- APP SIDEBAR SETUP -----------------

st.sidebar.title("📊 Control Panel")

# 1. Simulator Controls
st.sidebar.subheader("📡 Simulated Ingest Stream")
sim_btn_label = "Stop Simulator" if st.session_state.simulator_running else "Start Simulator"
st.sidebar.button(sim_btn_label, on_click=toggle_simulator, type="primary")
st.sidebar.write(f"Status: {'🟢 Active Streaming' if st.session_state.simulator_running else '🔴 Stopped'}")

# 2. Model Selection Dropdown
st.sidebar.subheader("🧠 NLP Core Model")
model_options = {
    "VADER (Rule-Based Baseline)": "vader",
    "TF-IDF + Logistic Regression (Custom ML)": "tfidf_logistic",
    "RoBERTa (Deep Learning Transformer)": "roberta"
}
selected_model_display = st.sidebar.selectbox(
    "Active Sentiment Classifier",
    options=list(model_options.keys()),
    index=list(model_options.values()).index(st.session_state.active_model_name)
)
selected_model_val = model_options[selected_model_display]

# If model selection changes, re-instantiate engine
if selected_model_val != st.session_state.active_model_name:
    # Update yaml configuration dynamically
    config["active_model"] = selected_model_val
    with open("config/config.yaml", "w") as f:
        yaml.safe_dump(config, f)
        
    st.session_state.active_model_name = selected_model_val
    with st.spinner("Reloading sentiment model weights..."):
        st.session_state.sentiment_engine = SentimentEngine()
    st.toast(f"Swapped model to: {selected_model_display}", icon="🧠")

# 3. Database Utility Actions
st.sidebar.subheader("⚙️ Database Operations")
if st.sidebar.button("Clear DB Records"):
    db.clear_database()
    st.toast("Database cleared successfully", icon="🧹")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Portfolio Hack:** Switch models using the dropdown, and click the **Start Simulator** button. The dashboard will process incoming social media feed items in real time!"
)

# ----------------- MAIN DASHBOARD GRID -----------------

st.title("📈 Real-Time Social Media Sentiment Analytics")
st.write("Analyze and visualize corporate brand reputation using lexicon, machine learning, or deep learning architectures.")

# Execute sentiment processing logic
run_sentiment_pipeline()

# Load processed data
processed_data = db.get_processed_data(limit=500)
df = pd.DataFrame(processed_data)

if df.empty:
    st.warning("⚠️ No processed posts found in database. Start the simulator in the sidebar to stream data!")
    
    # Showcase "Try it Yourself" when DB is empty
    st.subheader("🔮 Model Playpen (Interactive Text Test)")
    user_input = st.text_input("Enter a custom comment or review to test the active NLP model:")
    if user_input:
        res = st.session_state.sentiment_engine.predict(user_input)
        st.write(f"**Predicted Sentiment:** {res['label']} (Confidence Score: {res['score']:.4f})")
else:
    # 1. Metrics Cards Row
    kpis = db.get_kpi_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Comments Analyzed", value=kpis["total_posts"])
    with col2:
        st.metric(label="Positive Sentiment %", value=f"{kpis['pos_pct']}%")
    with col3:
        st.metric(label="Average Sentiment Score", value=kpis["avg_score"])
    with col4:
        st.metric(label="Active Model Engine", value=st.session_state.active_model_name.upper())

    st.markdown("---")

    # 2. Main Visualization Columns
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("🍩 Sentiment Ratio Distribution")
        sentiment_counts = df["sentiment_label"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        
        # Color mapping matching standard design rules
        color_map = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#9ca3af"}
        
        fig = px.pie(
            sentiment_counts, 
            names="Sentiment", 
            values="Count", 
            hole=0.4,
            color="Sentiment",
            color_discrete_map=color_map,
            template="plotly_dark"
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with viz_col2:
        st.subheader("📈 Sentiment Trends Over Time")
        # Ensure timestamp is datetime and sort
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_sorted = df.sort_values(by="timestamp")
        
        # Resample or take rolling average of sentiment scores to smooth trends
        # Map sentiment labels to scores for lines: Positive=1, Neutral=0, Negative=-1
        val_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
        df_sorted["numeric_val"] = df_sorted["sentiment_label"].map(val_map)
        df_sorted["rolling_score"] = df_sorted["numeric_val"].rolling(window=10, min_periods=1).mean()
        
        fig_line = px.line(
            df_sorted,
            x="timestamp",
            y="rolling_score",
            color="topic",
            title="Rolling Average Sentiment Score (-1.0 to +1.0)",
            template="plotly_dark",
            labels={"rolling_score": "Sentiment Value", "timestamp": "Timeline"}
        )
        fig_line.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_line, use_container_width=True)

    # 3. Brand Metrics and Wordcloud Row
    bot_col1, bot_col2 = st.columns(2)
    
    with bot_col1:
        st.subheader("🏢 Brand Net Sentiment Comparison")
        # Net Sentiment Score = Pos % - Neg %
        brand_stats = []
        for brand in df["topic"].unique():
            brand_df = df[df["topic"] == brand]
            total = len(brand_df)
            if total > 0:
                pos = len(brand_df[brand_df["sentiment_label"] == "Positive"])
                neg = len(brand_df[brand_df["sentiment_label"] == "Negative"])
                nss = ((pos - neg) / total) * 100
                brand_stats.append({"Brand": brand, "Net Sentiment Score (NSS)": round(nss, 1)})
        
        nss_df = pd.DataFrame(brand_stats)
        if not nss_df.empty:
            # Color code bars based on positive/negative NSS
            nss_df["Color"] = nss_df["Net Sentiment Score (NSS)"].apply(lambda x: "Positive" if x >= 0 else "Negative")
            fig_bar = px.bar(
                nss_df,
                x="Brand",
                y="Net Sentiment Score (NSS)",
                color="Color",
                color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444"},
                template="plotly_dark",
                labels={"Net Sentiment Score (NSS)": "NSS (%)"}
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Insufficient data to compute comparative brand stats.")

    with bot_col2:
        st.subheader("☁️ Discussion Topic Word Cloud")
        all_words = " ".join(df["text"].astype(str))
        
        # Preprocess text to keep only informative words for wordcloud
        clean_tokens = st.session_state.sentiment_engine.preprocessor.preprocess(all_words)
        cleaned_words_str = " ".join(clean_tokens)
        
        if cleaned_words_str.strip():
            wc = WordCloud(
                width=800, 
                height=300, 
                background_color="#101321", 
                colormap="cool",
                max_words=80
            ).generate(cleaned_words_str)
            
            fig_wc, ax = plt.subplots(figsize=(8, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_facecolor("#101321")
            st.pyplot(fig_wc)
            plt.close(fig_wc)
        else:
            st.info("Insufficient word token vocabulary to generate Word Cloud.")

    st.markdown("---")

    # 4. Interactive Sandbox & Live Feed
    col_play, col_feed = st.columns([1, 2])
    
    with col_play:
        st.subheader("🔮 Model Playpen")
        st.write("Test predictions instantly using the active model:")
        sandbox_input = st.text_input("Type a custom sentence to analyze:", placeholder="e.g., The food was alright, but the delivery took forever.")
        if sandbox_input:
            result = st.session_state.sentiment_engine.predict(sandbox_input)
            
            # Label rendering styles
            lbl = result["label"]
            score = result["score"]
            if lbl == "Positive":
                st.markdown(f"**Sentiment Result:** <span class='sentiment-positive'>{lbl}</span> (Confidence: **{score:.4f}**)", unsafe_allow_html=True)
            elif lbl == "Negative":
                st.markdown(f"**Sentiment Result:** <span class='sentiment-negative'>{lbl}</span> (Confidence: **{score:.4f}**)", unsafe_allow_html=True)
            else:
                st.markdown(f"**Sentiment Result:** <span class='sentiment-neutral'>{lbl}</span> (Confidence: **{score:.4f}**)", unsafe_allow_html=True)

    with col_feed:
        st.subheader("📻 Live Social Media Feed Stream")
        
        # Display search filter
        search_query = st.text_input("🔍 Search posts content or username:")
        display_df = df.copy()
        
        if search_query:
            display_df = display_df[
                display_df["text"].str.contains(search_query, case=False) | 
                display_df["username"].str.contains(search_query, case=False)
            ]
            
        # Select columns to display
        feed_cols = ["timestamp", "platform", "username", "topic", "text", "sentiment_label", "sentiment_score"]
        display_df = display_df[feed_cols]
        display_df.columns = ["Timestamp", "Platform", "Username", "Brand", "Post Content", "Sentiment", "Confidence"]
        
        # Style dataframe cells
        st.dataframe(
            display_df.style.map(
                lambda x: "color: #10b981; font-weight: bold;" if x == "Positive" else (
                    "color: #ef4444; font-weight: bold;" if x == "Negative" else (
                        "color: #9ca3af; font-weight: bold;" if x == "Neutral" else ""
                    )
                ),
                subset=["Sentiment"]
            ),
            use_container_width=True,
            height=300
        )

    # 5. Dynamic Brand Actionable Insights & Downloader
    st.markdown("---")
    st.subheader("📋 Brand Reputation Insights & Actionable Reports")
    st.write("Generate and download detailed sentiment audits detailing specific product strengths and customer pain points.")
    
    # Selected brand option
    brand_options = list(df["topic"].unique())
    selected_brand = st.selectbox("Select Brand for Reputation Report:", options=brand_options)
    
    if selected_brand:
        from collections import Counter
        
        # Filter data for chosen brand
        df_brand = df[df["topic"] == selected_brand]
        total_brand = len(df_brand)
        
        pos_brand_df = df_brand[df_brand["sentiment_label"] == "Positive"]
        neg_brand_df = df_brand[df_brand["sentiment_label"] == "Negative"]
        neut_brand_df = df_brand[df_brand["sentiment_label"] == "Neutral"]
        
        pos_count = len(pos_brand_df)
        neg_count = len(neg_brand_df)
        neut_count = len(neut_brand_df)
        
        pos_pct = round((pos_count / total_brand * 100), 1) if total_brand > 0 else 0.0
        neg_pct = round((neg_count / total_brand * 100), 1) if total_brand > 0 else 0.0
        neut_pct = round((neut_count / total_brand * 100), 1) if total_brand > 0 else 0.0
        nss = round((pos_pct - neg_pct), 1)
        
        # Extract keywords for Strengths (Positive)
        pos_tokens = []
        for text in pos_brand_df["text"]:
            pos_tokens.extend(st.session_state.sentiment_engine.preprocessor.preprocess(text))
        pos_freq = Counter(pos_tokens).most_common(5)
        
        # Extract keywords for Weaknesses (Negative)
        neg_tokens = []
        for text in neg_brand_df["text"]:
            neg_tokens.extend(st.session_state.sentiment_engine.preprocessor.preprocess(text))
        neg_freq = Counter(neg_tokens).most_common(5)
        
        # Display side-by-side columns
        col_good, col_bad = st.columns(2)
        
        with col_good:
            st.markdown("### 🌟 Key Strengths (Praise Points)")
            if pos_freq:
                for idx, (word, freq) in enumerate(pos_freq, 1):
                    st.write(f"**{idx}. {word.capitalize()}** (mentioned {freq} times)")
            else:
                st.write("*No positive keywords detected yet.*")
                
            st.markdown("#### 🗣️ Recent Praise Quotes:")
            recent_pos = pos_brand_df.head(3)
            if not recent_pos.empty:
                for _, row in recent_pos.iterrows():
                    st.markdown(f"> *\"{row['text']}\"* — @{row['username']} ({row['platform']})")
            else:
                st.write("*No recent positive quotes available.*")
                
        with col_bad:
            st.markdown("### ⚠️ Key Weaknesses (Pain Points)")
            if neg_freq:
                for idx, (word, freq) in enumerate(neg_freq, 1):
                    st.write(f"**{idx}. {word.capitalize()}** (mentioned {freq} times)")
            else:
                st.write("*No negative keywords detected yet.*")
                
            st.markdown("#### 🗣️ Recent Critical Complaints:")
            recent_neg = neg_brand_df.head(3)
            if not recent_neg.empty:
                for _, row in recent_neg.iterrows():
                    st.markdown(f"> *\"{row['text']}\"* — @{row['username']} ({row['platform']})")
            else:
                st.write("*No recent negative quotes available.*")
                
        # Generate string report
        report_text = f"==================================================\n"
        report_text += f"BRAND SENTIMENT REPUTATION REPORT: {selected_brand.upper()}\n"
        report_text += f"Generated on: {pd.Timestamp.now()}\n"
        report_text += f"Classifier Engine: {st.session_state.active_model_name.upper()}\n"
        report_text += f"==================================================\n\n"
        
        report_text += f"SUMMARY METRICS:\n"
        report_text += f"--------------------------------------------------\n"
        report_text += f"- Total Mentions: {total_brand}\n"
        report_text += f"- Positive Sentiment Ratio: {pos_pct}%\n"
        report_text += f"- Negative Sentiment Ratio: {neg_pct}%\n"
        report_text += f"- Neutral Sentiment Ratio: {neut_pct}%\n"
        report_text += f"- Net Sentiment Score (NSS): {nss}%\n\n"
        
        report_text += f"🌟 CUSTOMER PRAISES & KEY STRENGTHS:\n"
        report_text += f"--------------------------------------------------\n"
        if pos_freq:
            for idx, (word, freq) in enumerate(pos_freq, 1):
                report_text += f"{idx}. {word.capitalize()} (Count: {freq})\n"
        else:
            report_text += "No positive features tracked.\n"
        report_text += "\n"
        
        report_text += f"🌟 RAW POSITIVE COMMENT EXAMPLES (DIRECT QUOTES):\n"
        report_text += f"--------------------------------------------------\n"
        if not recent_pos.empty:
            for _, row in recent_pos.iterrows():
                report_text += f"- \"{row['text']}\" (Posted by @{row['username']} on {row['platform']})\n"
        else:
            report_text += "No positive quotes tracked.\n"
        report_text += "\n"
        
        report_text += f"⚠️ CUSTOMER COMPLAINTS & KEY WEAKNESSES:\n"
        report_text += f"--------------------------------------------------\n"
        if neg_freq:
            for idx, (word, freq) in enumerate(neg_freq, 1):
                report_text += f"{idx}. {word.capitalize()} (Count: {freq})\n"
        else:
            report_text += "No negative complaints tracked.\n"
        report_text += "\n"
        
        report_text += f"⚠️ RAW NEGATIVE COMMENT EXAMPLES (DIRECT QUOTES):\n"
        report_text += f"--------------------------------------------------\n"
        if not recent_neg.empty:
            for _, row in recent_neg.iterrows():
                report_text += f"- \"{row['text']}\" (Posted by @{row['username']} on {row['platform']})\n"
        else:
            report_text += "No negative quotes tracked.\n"
        report_text += "\n"
        
        report_text += f"ACTIONABLE STRATEGY PROMPT:\n"
        report_text += f"--------------------------------------------------\n"
        if nss < 0:
            report_text += "Reputation Alert: Negative discussions dominate. Investigate the key complaints above immediately.\n"
        elif nss < 30:
            report_text += "Reputation Stable: Moderate customer satisfaction. Focus on resolving key weaknesses to build satisfaction.\n"
        else:
            report_text += "Reputation Excellent: Strong brand advocacy. Continue leveraging key strengths in marketing campaigns.\n"
            
        # Download Button
        st.download_button(
            label=f"📥 Download {selected_brand} Insights Audit Report",
            data=report_text,
            file_name=f"{selected_brand.lower()}_sentiment_report.txt",
            mime="text/plain"
        )

# Auto-refresh mechanism (if simulator is running) to update graphs dynamically
if st.session_state.simulator_running:
    time.sleep(1.5)
    st.rerun()
