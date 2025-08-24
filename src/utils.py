# src/utils.py
import re
from typing import List
import pandas as pd

HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")

def extract_hashtags(text: str) -> List[str]:
    return [h.lower() for h in HASHTAG_RE.findall(text or "")]

def extract_mentions(text: str) -> List[str]:
    return [m.lower() for m in MENTION_RE.findall(text or "")]

def load_sentiment_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["timestamp","text","sentiment","confidence"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["timestamp"] = pd.NaT
    # Normalize sentiment labels
    if "sentiment" not in df.columns:
        df["sentiment"] = "NEUTRAL"
    df["sentiment"] = df["sentiment"].astype(str).str.upper()
    # Confidence to float [0,1]
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").clip(lower=0, upper=1)
    else:
        df["confidence"] = 0.0
    # Features
    df["hashtags"] = df["text"].fillna("").apply(extract_hashtags)
    df["mentions"] = df["text"].fillna("").apply(extract_mentions)
    return df

