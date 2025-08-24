# src/sentiment.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import os

# Load Hugging Face sentiment model (distilbert)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

def analyze_sentiment(input_text):
    # Check if input is a CSV file
    if os.path.exists(input_text) and input_text.endswith(".csv"):
        df = pd.read_csv(input_text, names=["timestamp","text"], header=0)
        df["sentiment"] = df["text"].apply(lambda x: pipeline(x)[0]["label"])
        return df
    else:
        # Input is normal text
        result = pipeline(input_text)[0]
        return {"text": input_text, "label": result["label"], "score": result["score"]}


