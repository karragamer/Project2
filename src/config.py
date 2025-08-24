# src/config.py
import os

# Prefer environment variables (great for Docker),
# but fall back to hardcoded values if not set.
BEARER_TOKEN = os.getenv("AAAAAAAAAAAAAAAAAAAAAPMA3wEAAAAA5F6z0b7PjaVkmXO15aJiELMxI64%3Dn0DbsnMWtCqhhtSqnclUavSMbkLdF7Ga8tsnrsdM1kNdUqsbdp")
HASHTAG = os.getenv("HASHTAG", "#YourCampaignHashtag")

