"""
Configuration settings for the Movie Recommender Backend
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Index Configuration
INDEX_NAME = "movies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search Defaults
DEFAULT_NUM_RESULTS = 5
DEFAULT_DISTANCE_THRESHOLD = 0.5
DEFAULT_HYBRID_ALPHA = 0.5

# Index Schema for Redis Vector Search
INDEX_SCHEMA = {
    "index": {
        "name": INDEX_NAME,
        "prefix": "movie:",
    },
    "fields": [
        {"name": "title", "type": "text"},
        {"name": "genre", "type": "tag"},
        {"name": "rating", "type": "numeric"},
        {"name": "description", "type": "text"},
        {
            "name": "vector",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": 384,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
    ],
}

