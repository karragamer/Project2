# src/collect_data.py
import tweepy
import pandas as pd
import os
from datetime import datetime
from src import config

CSV_FILE = "data/tweets.csv"

class TweetStreamer(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        # Prepare tweet data
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "text": tweet.text
        }
        df = pd.DataFrame([data])

        # If file doesn't exist, create it with header
        if not os.path.isfile(CSV_FILE):
            df.to_csv(CSV_FILE, mode='w', header=True, index=False)
        else:
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)

        print(f"[NEW TWEET] {tweet.text}")

def start_stream():
    streamer = TweetStreamer(config.BEARER_TOKEN)

    # Clean old rules before adding a new one
    existing_rules = streamer.get_rules().data
    if existing_rules:
        rule_ids = [rule.id for rule in existing_rules]
        streamer.delete_rules(rule_ids)

    # Add new hashtag rule
    streamer.add_rules(tweepy.StreamRule(config.HASHTAG))
    print(f"📡 Streaming tweets for {config.HASHTAG}...")
    streamer.filter(tweet_fields=["created_at", "lang"])

if __name__ == "__main__":
    start_stream()

