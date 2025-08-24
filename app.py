import os
import time
import requests
import tweepy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# ---------------- Config & Secrets ---------------- #
st.set_page_config(page_title="📊 Social Media Sentiment Analyzer", page_icon="📈", layout="centered")
st.title("📊 Social Media Sentiment Analyzer")
st.caption("Analyze sentiments from manual text or live tweets (Twitter API) using a Hugging Face model.")

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
def plot_sentiment_pie(sentiment):
    labels = ["Positive", "Negative", "Neutral"]
    sizes = [1 if sentiment == l else 0 for l in labels]
    colors = ["green", "red", "orange"]

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=[l if s > 0 else "" for l, s in zip(labels, sizes)],
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        startangle=90,
        colors=colors
    )
    ax.axis("equal")
    return fig

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

# ---------------- UI ---------------- #
option = st.radio("Choose input method:", ["Manual Text", "Fetch Tweets"])

# Manual
if option == "Manual Text":
    user_input = st.text_area("✍️ Enter text to analyze:")

    if st.button("🔍 Analyze"):
        if user_input.strip():
            label, score, err = analyze_sentiment(user_input)
            st.subheader("📌 Sentiment Result")

            if err:
                st.error(f"❌ Hugging Face API error: {err}")
                st.write("Confidence Score: 0.00")
            else:
                st.write(f"**Sentiment:** {label}")
                st.write(f"**Confidence Score:** {score:.2f}")
                st.pyplot(plot_sentiment_pie(label))
                st.plotly_chart(plot_gauge(label))
                st.markdown({
                    "Positive": "😊 **Great! People like this.**",
                    "Neutral":  "😐 **It's okay, neutral vibes.**",
                    "Negative": "😡 **Oops! Negative reaction detected.**"
                }[label])
        else:
            st.warning("⚠️ Please enter some text.")

# Tweets
else:
    query = st.text_input("🔑 Enter a keyword or hashtag (e.g., #AI)")
    count = st.slider("Number of tweets to fetch", 10, 20, 10)  # Changed to 10-20 range
    st.caption("Twitter API requires between 10-100 tweets per request")

    if st.button("📥 Fetch & Analyze Tweets"):
        if not query.strip():
            st.warning("⚠️ Please enter a valid keyword or hashtag.")
        else:
            with st.spinner("Fetching tweets... ⏳"):
                tweets, error = fetch_tweets(query, count)

            if error:
                st.error(error)
            elif not tweets:
                st.error("⚠️ No tweets found for this query.")
            else:
                st.success(f"✅ Fetched {len(tweets)} recent tweets for '{query}'")
                st.subheader("📌 Sample Tweets")
                for i, t in enumerate(tweets[:5], 1):
                    st.write(f"**Tweet {i}:** {t}")

                # Batch sentiment
                sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
                errors = 0
                for t in tweets:
                    label, _, err = analyze_sentiment(t)
                    if err:
                        errors += 1
                        continue
                    if label in sentiments:
                        sentiments[label] += 1

                st.subheader("🧾 Sentiment Summary")
                st.json(sentiments)
                if errors:
                    st.info(f"ℹ️ {errors} tweet(s) could not be analyzed due to API throttling or loading. Try again.")

                fig2, ax2 = plt.subplots()
                ax2.pie(
                    sentiments.values(),
                    labels=sentiments.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=["green", "red", "orange"]
                )
                ax2.axis("equal")
                st.pyplot(fig2)
