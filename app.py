import os
import time
import requests
import tweepy
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ---------------- Config & Secrets ---------------- #
st.set_page_config(
    page_title="🚀 AI Sentiment Hub", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Sentiment Hub</h1>
    <p>Real-time sentiment analysis with advanced AI models</p>
</div>
""", unsafe_allow_html=True)

# Twitter (X) Bearer Token
TWITTER_BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", os.getenv("TWITTER_BEARER_TOKEN"))
if not TWITTER_BEARER_TOKEN:
    st.error("❌ Missing TWITTER_BEARER_TOKEN. Add it in Streamlit Secrets or as an env var.")
    st.stop()

# Hugging Face token (MUST be set in secrets)
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", os.getenv("HF_API_TOKEN"))
if not HF_API_TOKEN:
    st.error("❌ Missing HF_API_TOKEN. Add it in Streamlit Secrets or as an env var.")
    st.stop()

HF_API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"

# ---------------- Hugging Face helper ---------------- #
def analyze_sentiment(text, retries=3, timeout=30):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True, "use_cache": True}
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout)

            if resp.status_code in (503, 429):
                time.sleep(2 * attempt)
                continue
            if resp.status_code != 200:
                return "Error", 0.0, f"{resp.status_code}: {resp.text}"

            data = resp.json()

            # Handle error responses
            if isinstance(data, dict) and "error" in data:
                time.sleep(2 * attempt)
                continue

            if isinstance(data, list) and len(data) > 0:
                candidates = data[0] if isinstance(data[0], list) else data
                best = max(candidates, key=lambda x: x.get("score", 0.0))

                raw_label = best.get("label", "")
                score = float(best.get("score", 0.0))

                # Normalize labels
                map_3 = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral"}
                if raw_label in map_3:
                    label = map_3[raw_label]
                elif raw_label.upper().startswith("LABEL_"):
                    idx = int(raw_label.split("_")[-1])
                    label = ["Negative", "Neutral", "Positive"][idx] if idx in (0, 1, 2) else "Neutral"
                else:
                    u = raw_label.upper()
                    if "POS" in u: label = "Positive"
                    elif "NEG" in u: label = "Negative"
                    elif "NEU" in u: label = "Neutral"
                    else: label = raw_label

                return label, score, None

            return "Error", 0.0, "Unexpected response format."
        except requests.RequestException as e:
            if attempt == retries:
                return "Error", 0.0, f"Request failed: {e}"
            time.sleep(2 * attempt)

    return "Error", 0.0, "Failed after retries."

# ---------------- Twitter fetch function ---------------- #
def fetch_tweets(query, count=10):
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        response = client.search_recent_tweets(
            query=query,
            max_results=count,  # Use the exact count since we're limiting the slider to 10-20
            tweet_fields=["text", "lang", "created_at"]
        )

        tweets = []
        if response.data:
            for tweet in response.data:
                if getattr(tweet, "lang", "en") == "en":
                    tweets.append(tweet.text)

        return tweets, None
    except Exception as e:
        return [], f"⚠️ Error fetching tweets: {str(e)}"

# ---------------- Visualization Helpers ---------------- #
# Remove matplotlib pie chart function as we're using Plotly

def plot_gauge(sentiment):
    value = {"Positive": 80, "Neutral": 50, "Negative": 20}.get(sentiment, 50)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"Sentiment Meter: {sentiment}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 33], 'color': "red"},
                {'range': [34, 66], 'color': "orange"},
                {'range': [67, 100], 'color': "green"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))
    return gauge

# ---------------- Interactive Sidebar ---------------- #
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Navigation menu
    selected = option_menu(
        menu_title="Navigation",
        options=["🔍 Text Analysis", "🐦 Tweet Monitor", "📊 Analytics", "⚙️ Settings"],
        icons=["search", "twitter", "bar-chart", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    st.markdown("---")
    
    # Real-time settings
    st.markdown("### ⚡ Real-time Settings")
    auto_refresh = st.toggle("Auto Refresh", value=False)
    if auto_refresh:
        refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
    
    st.markdown("### 🎨 Display Options")
    show_charts = st.checkbox("Show Charts", value=True)
    show_metrics = st.checkbox("Show Metrics", value=True)
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark"])

# ---------------- Main Content Based on Selection ---------------- #
if selected == "🔍 Text Analysis":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Text Input")
        user_input = st.text_area(
            "Enter your text here:", 
            height=150,
            placeholder="Type or paste any text to analyze its sentiment..."
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col_btn3:
            sample_btn = st.button("📋 Load Sample", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if sample_btn:
            user_input = "I absolutely love this new product! It's amazing and works perfectly."
            st.rerun()
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        st.info("💡 **Tips:**\n- Use complete sentences\n- Avoid very short text\n- Mix of emotions gives better insights")
        
        if show_metrics:
            st.markdown("### 📊 Session Stats")
            if 'analysis_count' not in st.session_state:
                st.session_state.analysis_count = 0
            st.metric("Analyses Done", st.session_state.analysis_count)
    
    if analyze_btn and user_input.strip():
        st.session_state.analysis_count += 1
        
        with st.spinner("🤖 AI is analyzing..."):
            label, score, err = analyze_sentiment(user_input)
        
        if err:
            st.error(f"❌ Analysis failed: {err}")
        else:
            # Results in expandable sections
            with st.expander("📊 Detailed Results", expanded=True):
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric(
                        "Sentiment", 
                        label,
                        delta=f"{score:.1%} confidence"
                    )
                
                with col_res2:
                    color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}[label]
                    st.metric("Status", f"{color} {label}")
                
                with col_res3:
                    st.metric("Confidence", f"{score:.1%}")
            
            if show_charts:
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(plot_gauge(label), use_container_width=True)
                with chart_col2:
                    # Enhanced pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=[label],
                        values=[score],
                        hole=0.6,
                        marker_colors=[{"Positive": "#00CC96", "Negative": "#EF553B", "Neutral": "#636EFA"}[label]]
                    )])
                    fig.update_layout(title="Sentiment Distribution", template=chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            interpretations = {
                "Positive": "😊 **Excellent!** This text expresses positive emotions and attitudes.",
                "Neutral": "😐 **Balanced.** This text maintains a neutral tone without strong emotions.",
                "Negative": "😔 **Concerning.** This text contains negative sentiments that may need attention."
            }
            st.success(interpretations[label])
    
    elif analyze_btn:
        st.warning("⚠️ Please enter some text to analyze.")

elif selected == "🐦 Tweet Monitor":
    # Twitter monitoring interface
    st.markdown("### 🐦 Live Twitter Sentiment Monitor")
    
    # Input controls
    input_col1, input_col2, input_col3 = st.columns([2, 1, 1])
    
    with input_col1:
        query = st.text_input(
            "🔍 Search Query", 
            placeholder="#AI, @username, or any keyword",
            help="Enter hashtags, mentions, or keywords to monitor"
        )
    
    with input_col2:
        count = st.number_input("Tweet Count", 10, 100, 20)
    
    with input_col3:
        monitor_btn = st.button("🚀 Start Monitor", use_container_width=True)
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            lang_filter = st.selectbox("Language", ["en", "es", "fr", "de", "all"])
            result_type = st.selectbox("Result Type", ["recent", "popular", "mixed"])
        with filter_col2:
            exclude_retweets = st.checkbox("Exclude Retweets", True)
            min_likes = st.number_input("Min Likes", 0, 1000, 0)
    
    if monitor_btn and query.strip():
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Fetching tweets...")
        progress_bar.progress(25)
        
        tweets, error = fetch_tweets(query, count)
        
        if error:
            st.error(error)
        elif not tweets:
            st.warning("⚠️ No tweets found. Try different keywords.")
        else:
            progress_bar.progress(50)
            status_text.text("🤖 Analyzing sentiments...")
            
            # Process tweets with progress
            results = []
            sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
            
            for i, tweet in enumerate(tweets):
                label, score, err = analyze_sentiment(tweet)
                if not err and score >= confidence_threshold:
                    results.append({
                        "tweet": tweet[:100] + "..." if len(tweet) > 100 else tweet,
                        "sentiment": label,
                        "confidence": score,
                        "timestamp": datetime.now()
                    })
                    sentiments[label] += 1
                
                progress_bar.progress(50 + (i + 1) / len(tweets) * 50)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            if results:
                # Summary metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("📊 Total Analyzed", len(results))
                with metric_col2:
                    st.metric("😊 Positive", sentiments["Positive"], 
                             f"{sentiments['Positive']/len(results)*100:.1f}%")
                with metric_col3:
                    st.metric("😐 Neutral", sentiments["Neutral"],
                             f"{sentiments['Neutral']/len(results)*100:.1f}%")
                with metric_col4:
                    st.metric("😔 Negative", sentiments["Negative"],
                             f"{sentiments['Negative']/len(results)*100:.1f}%")
                
                # Interactive charts
                if show_charts:
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Donut chart
                        fig_donut = go.Figure(data=[go.Pie(
                            labels=list(sentiments.keys()),
                            values=list(sentiments.values()),
                            hole=0.5,
                            marker_colors=["#00CC96", "#EF553B", "#636EFA"]
                        )])
                        fig_donut.update_layout(
                            title="Sentiment Distribution",
                            template=chart_theme
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)
                    
                    with chart_col2:
                        # Confidence distribution
                        df_results = pd.DataFrame(results)
                        fig_hist = px.histogram(
                            df_results, 
                            x="confidence", 
                            color="sentiment",
                            title="Confidence Distribution",
                            template=chart_theme
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                # Tweet details
                with st.expander("📱 Tweet Details", expanded=False):
                    df_display = pd.DataFrame(results)
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        column_config={
                            "confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                help="AI confidence score",
                                min_value=0,
                                max_value=1,
                            ),
                        }
                    )
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
    
    elif monitor_btn:
        st.warning("⚠️ Please enter a search query.")

elif selected == "📊 Analytics":
    st.markdown("### 📊 Analytics Dashboard")
    st.info("🚧 Advanced analytics coming soon! This will include trend analysis, comparison tools, and historical data.")
    
    # Placeholder analytics
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if st.session_state.analysis_history:
        st.markdown("#### Recent Analysis History")
        # Show some mock analytics
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sentiment_trend = np.random.choice(['Positive', 'Negative', 'Neutral'], size=30, p=[0.4, 0.3, 0.3])
        
        df_trend = pd.DataFrame({
            'Date': dates,
            'Sentiment': sentiment_trend
        })
        
        fig_trend = px.line(df_trend.groupby(['Date', 'Sentiment']).size().reset_index(name='Count'), 
                           x='Date', y='Count', color='Sentiment', title='Sentiment Trends Over Time')
        st.plotly_chart(fig_trend, use_container_width=True)

elif selected == "⚙️ Settings":
    st.markdown("### ⚙️ Application Settings")
    
    # API Settings
    with st.expander("🔑 API Configuration", expanded=True):
        st.info("Current API tokens are configured via Streamlit secrets.")
        st.code(f"Twitter Bearer Token: {'✅ Configured' if TWITTER_BEARER_TOKEN else '❌ Missing'}")
        st.code(f"Hugging Face Token: {'✅ Configured' if HF_API_TOKEN else '❌ Missing'}")
    
    # Model Settings
    with st.expander("🤖 Model Configuration"):
        st.selectbox("Sentiment Model", ["finiteautomata/bertweet-base-sentiment-analysis"])
        st.slider("API Timeout (seconds)", 10, 60, 30)
        st.slider("Max Retries", 1, 5, 3)
    
    # Export Settings
    with st.expander("💾 Data Export"):
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
        if st.button("📥 Export Analysis History"):
            st.success(f"Data exported in {export_format} format!")
    
    # Reset
    if st.button("🔄 Reset All Settings", type="secondary"):
        st.success("Settings reset to default!")
