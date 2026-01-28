# Movie Recommender with Redis RAG & Semantic Cache Workshop

A hands-on workshop to build a movie recommendation engine using **Redis Cloud**, **Vector Search**, **RAG (Retrieval Augmented Generation)**, and **Semantic Caching**. Learn how to implement various search techniques including vector similarity search, hybrid search, full-text search, and more! Also includes a **Help Center** with guardrails and PII protection.

![Redis Cloud](https://img.shields.io/badge/Redis_Cloud-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Part 1: Redis Cloud Setup](#part-1-redis-cloud-setup)
- [Part 2: Environment Setup](#part-2-environment-setup)
- [Part 3: Understanding the Data](#part-3-understanding-the-data)
- [Part 4: Building the Search Engine](#part-4-building-the-search-engine)
- [Part 5: Running the Application](#part-5-running-the-application)
- [Part 6: Exploring Search Types](#part-6-exploring-search-types)
- [Part 7: Semantic Caching](#part-7-semantic-caching)
- [Part 8: Help Center with RAG](#part-8-help-center-with-rag)
- [Part 9: Guardrails & PII Protection](#part-9-guardrails--pii-protection)
- [Topics Covered](#topics-covered)

---

## Overview

This workshop guides you through building a complete movie recommendation system that leverages:

- **Redis Cloud** - Fully managed Redis database with vector search capabilities
- **Vector Similarity Search** - Find movies by semantic meaning
- **Full-Text Search** - Traditional keyword-based search with BM25 scoring
- **Hybrid Search** - Combine vector and text search for best results
- **Filtered Search** - Apply metadata filters (genre, rating) to vector results
- **Range Queries** - Find results within a semantic distance threshold
- **Semantic Caching** - Cache LLM responses for faster repeated queries
- **Help Center RAG** - AI-powered customer support with article retrieval
- **Semantic Router Guardrails** - Block off-topic queries using semantic routing
- **PII Protection** - Prevent caching of personally identifiable information

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│      Movie Search UI  │  Help Center Chat  │  http://localhost:3000
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       NGINX Reverse Proxy                        │
│                     Routes /api/* to backend                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│                   http://localhost:8000                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    MovieSearchEngine                         │ │
│  │  • Vector Search    • Hybrid Search    • Range Search        │ │
│  │  • Filtered Search  • Keyword Search   • Embeddings Cache    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    HelpCenterEngine                          │ │
│  │  • RAG Pipeline     • Semantic Cache   • OpenAI LLM          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      Guardrails                              │ │
│  │  • Semantic Router (topic filtering)                         │ │
│  │  • PII Detection (email, phone, SSN, credit card)            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                 │
│                    ┌───────────┴───────────┐                    │
│                    ▼                       ▼                    │
│  ┌─────────────────────────────┐ ┌─────────────────────────────┐ │
│  │   HuggingFace Vectorizer    │ │    OpenAI GPT-4o-mini       │ │
│  │   (all-MiniLM-L6-v2)        │ │    (Response Generation)    │ │
│  └─────────────────────────────┘ └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Redis Cloud                             │
│              redis://default:***@your-endpoint:port              │
│                                                                  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐│
│  │  Movie Index   │ │  Help Articles │ │   Semantic Cache       ││
│  │  (HNSW/FLAT)   │ │  Index         │ │   (LLM Responses)      ││
│  └────────────────┘ └────────────────┘ └────────────────────────┘│
│  ┌────────────────┐ ┌────────────────┐                          │
│  │  Embeddings    │ │  Router Index  │                          │
│  │  Cache         │ │  (Guardrails)  │                          │
│  └────────────────┘ └────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before starting this workshop, ensure you have:

- **Python 3.11+** installed
- **Node.js 20+** installed
- **RIOT CLI** - Redis Input/Output Tools for data import
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Redis Cloud account** - Free tier available at [redis.io/try-free](https://redis.io/try-free)

### Installing RIOT

RIOT (Redis Input/Output Tools) is used to import the movie dataset into Redis. Install it using one of these methods:

**macOS (Homebrew):**

```bash
brew install redis/tap/riot
```

**Docker:**

```bash
# Run RIOT commands via Docker
docker run riotx/riot --help
```

**Manual Download:**

Download from [https://github.com/redis/riot](https://github.com/redis/riot)

---

## Part 1: Redis Cloud Setup

### Estimated time: **10 minutes**

### Task 1: Create a Redis Cloud Account

1. Visit [https://redis.io/try-free](https://redis.io/try-free)
2. Create a free account (no credit card required)
3. You can sign up using your GitHub or Google account

Once you create your account, you will be redirected to the Redis Cloud console.

### Task 2: Create a Free Database

1. Click the **New Database** button
2. Select the **Essentials** subscription (free tier)
3. Choose your preferred cloud provider (AWS, GCP, or Azure) and region
4. Set a memorable database name (e.g., `movie-recommender`)
5. Ensure the database size is **30MB** (free tier)
6. Click **Create Database**

> It usually takes about 30 seconds for your database to be created.

### Task 3: Get Your Connection Details

1. Once your database is ready, find the **General** section
2. Click the **Connect** button
3. Select the **Redis CLI** option
4. Click the eye icon to reveal the password
5. Copy the Redis URL in this format:

```
redis://default:YOUR_PASSWORD@YOUR_ENDPOINT:PORT
```

> **Keep this URL handy!** You'll need it in the next section.

---

## Part 2: Environment Setup

### Estimated time: **10 minutes**

### Task 1: Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommender-rag-semantic-cache-workshop.git
cd movie-recommender-rag-semantic-cache-workshop
```

### Task 2: Set Up the Backend

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Task 3: Configure Environment Variables

Create a `.env` file in the project root with your configuration:

```bash
# Redis Cloud Configuration (Required)
REDIS_URL=redis://default:YOUR_PASSWORD@YOUR_ENDPOINT:PORT

# OpenAI API Key (Required for Help Center)
OPENAI_API_KEY=your_openai_key_here
```

> **Important:** 
> - Replace `YOUR_PASSWORD`, `YOUR_ENDPOINT`, and `PORT` with your actual Redis Cloud credentials from Part 1.
> - Get your OpenAI API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (required for Help Center RAG features).

### Task 4: Set Up the Frontend

```bash
cd frontend
npm install
cd ..
```

Next, create a file `.env.local` in `/frontend`:

```
# If running in codespaces
VITE_API_URL=https://${CODESPACE_NAME}-8000.app.github.dev/api

# Else, if running on your own machine
VITE_API_URL=http:localhost:8000/api
```

---

## Part 3: Understanding the Data

### Estimated time: **5 minutes**

The workshop uses a curated dataset of Bollywood movies located in `resources/movies.json`.

### Movie Schema

Each movie has the following structure:

```json
{
    "title": "3 Idiots",
    "genre": "comedy",
    "rating": 9,
    "description": "Two friends embark on a quest for a lost buddy..."
}
```

### Index Schema (Redis Vector Search)

The movies are indexed with the following schema defined in `backend/config.py`:

```python
INDEX_SCHEMA = {
    "index": {
        "name": "movies",
        "prefix": "movie:",
    },
    "fields": [
        {"name": "title", "type": "text"},           # Full-text searchable
        {"name": "genre", "type": "tag"},            # Filterable tag
        {"name": "rating", "type": "numeric"},       # Numeric filter
        {"name": "description", "type": "text"},     # Full-text searchable
        {
            "name": "vector",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": 384,                         # MiniLM embedding dimension
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
    ],
}
```

---

## Part 4: Building the Search Engine

### Estimated time: **20 minutes**

### Understanding the MovieSearchEngine Class

The core search logic is in `backend/search_engine.py`. Let's explore each component:

### Task 1: Initializing the Search Engine

```python
class MovieSearchEngine:
    def __init__(self):
        # Connect to Redis Cloud
        self.client = Redis.from_url(REDIS_URL)
        self.schema = IndexSchema.from_dict(INDEX_SCHEMA)
        self.index = SearchIndex(self.schema, self.client)
        
        # Initialize embeddings with cache
        self.vectorizer = HFTextVectorizer(
            model=EMBEDDING_MODEL,
            cache=EmbeddingsCache(
                name="embedcache",
                ttl=600,  # 10 minutes
                redis_client=self.client,
            )
        )
```

### Task 2: Implementing Vector Search

Vector search finds semantically similar movies using KNN:

```python
def vector_search(self, query: str, num_results: int = 5):
    embedded_query = self._embed_query(query)
    
    vec_query = VectorQuery(
        vector=embedded_query,
        vector_field_name="vector",
        num_results=num_results,
        return_fields=["title", "genre", "rating", "description"],
        return_score=True,
    )
    
    return self.index.query(vec_query)
```

### Task 3: Implementing Filtered Search

Combine vector search with metadata filters:

```python
def filtered_search(self, query: str, genre: str = None, min_rating: int = None):
    embedded_query = self._embed_query(query)
    
    # Build filter expression
    filter_expression = None
    
    if genre and genre.lower() != "all":
        filter_expression = Tag("genre") == genre.lower()
    
    if min_rating is not None and min_rating > 0:
        num_filter = Num("rating") >= min_rating
        if filter_expression:
            filter_expression = filter_expression & num_filter
        else:
            filter_expression = num_filter
    
    vec_query = VectorQuery(
        vector=embedded_query,
        vector_field_name="vector",
        filter_expression=filter_expression,
        # ... other params
    )
```

### Task 4: Implementing Hybrid Search

Combine vector similarity with BM25 text scoring:

```python
def hybrid_search(self, query: str, alpha: float = 0.5):
    embedded_query = self._embed_query(query)
    
    hybrid_query = AggregateHybridQuery(
        text=query,
        text_field_name="description",
        text_scorer="BM25",
        vector=embedded_query,
        vector_field_name="vector",
        alpha=alpha,  # 1.0 = pure vector, 0.0 = pure text
        # ... other params
    )
```

### Task 5: Implementing Range Search

Find movies within a semantic distance threshold:

```python
def range_search(self, query: str, distance_threshold: float = 0.5):
    embedded_query = self._embed_query(query)
    
    range_query = RangeQuery(
        vector=embedded_query,
        vector_field_name="vector",
        distance_threshold=distance_threshold,
        return_score=True,
    )
```

---

## Part 5: Running the Application

### Estimated time: **15 minutes**

### Step 1: Import Data with RIOT

First, import the movie dataset into Redis Cloud using RIOT:

```bash
# Set your Redis Cloud URL
export REDIS_URL=redis://default:YOUR_PASSWORD@YOUR_ENDPOINT:PORT

# Run the RIOT import script
./scripts/import_data.sh
```

This imports the raw movie data (title, genre, rating, description) into Redis with key prefix `movie:`.

### Step 2: Create Embeddings and Search Index

After importing data with RIOT, generate embeddings and create the search index:

```bash
curl -X POST http://localhost:8000/api/create-index
```

This reads the RIOT-imported data from Redis, generates vector embeddings for each movie description, and creates the RediSearch index.

### Option A: Run Locally (Development)

**Terminal 1 - Backend:**

```bash
cd movie-recommender-rag-semantic-cache-workshop
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm run dev
```

**Import data and create index:**

```bash
# Step 1: Import raw data with RIOT
./scripts/import_data.sh

# Step 2: Generate embeddings and create search index
curl -X POST http://localhost:8000/api/create-index
```

Access the application at: `http://localhost:5173`

### Option B: Run with Docker Compose

First, set your Redis Cloud URL as an environment variable:

```bash
export REDIS_URL=redis://default:YOUR_PASSWORD@YOUR_ENDPOINT:PORT
```

Then build and start all services:

```bash
# Pull required tool images (RIOT for data import)
docker-compose --profile tools pull

# Build and start all services
docker-compose up --build

# Step 1: Import raw data with RIOT
./scripts/import_data.sh

# Step 2: Generate embeddings and create search index
curl -X POST http://localhost:8000/api/create-index
```

Access the application at: `http://localhost:3000`

### Running on Github Codespaces

Once your frontend and backend services are up and running, go to the **ports** tab on your codespaces IDE, and change the visibility of the ports 3000 and 5173 from `private` to `public`.

Then, you can access the application UI at the URL provided by Codespaces.

---

## Part 6: Exploring Search Types

### Estimated time: **15 minutes**

Try these searches in the UI to understand each search type:

### Vector Search

**Best for:** Finding movies by concept/meaning

```
Query: "friends on an adventure"
Returns: Zindagi Na Milegi Dobara, 3 Idiots, etc.
```

### Keyword Search (BM25)

**Best for:** Finding movies with specific keywords

```
Query: "gangster police"
Returns: Movies with exact keyword matches
```

### Hybrid Search

**Best for:** Combining semantic understanding with keyword precision

```
Query: "revenge action"
Alpha: 0.5 (balanced)
Returns: Best of both worlds
```

### Filtered Search

**Best for:** Narrowing results by metadata

```
Query: "emotional story"
Genre: Romance
Min Rating: 8
Returns: Highly-rated romantic movies
```

### Range Search

**Best for:** Finding only highly relevant results

```
Query: "comedy with friends"
Distance Threshold: 0.3
Returns: Only movies very close semantically
```

---

## Part 7: Semantic Caching

### Estimated time: **10 minutes**

The application uses **Semantic Caching** via RedisVL's `EmbeddingsCache` to optimize performance. The cache is stored in your Redis Cloud database alongside your movie data.

### How It Works

```python
self.vectorizer = HFTextVectorizer(
    model=EMBEDDING_MODEL,
    cache=EmbeddingsCache(
        name="embedcache",
        ttl=600,  # Cache for 10 minutes
        redis_client=self.client,
    )
)
```

### Benefits

1. **Faster Responses** - Repeated queries don't need re-embedding
2. **Reduced Compute** - ML model inference is expensive
3. **Cost Savings** - Less compute = lower costs
4. **Cloud-Native** - Cache persists in Redis Cloud, shared across instances

### Try It!

1. Search for "action movies with revenge"
2. Note the response time
3. Search for the same query again
4. The second query should be faster!

### Verify in Redis Cloud

You can see your cached embeddings in the Redis Cloud console:

1. Go to your database in Redis Cloud
2. Click on **Data Browser**
3. Look for keys prefixed with `embedcache:`

---

## Part 8: Help Center with RAG

### Estimated time: **15 minutes**

The Help Center demonstrates a complete RAG (Retrieval Augmented Generation) pipeline for customer support.

### How It Works

```
User Question → Guardrails Check → Cache Check → Article Search → LLM Response
                     │                  │              │              │
                     ▼                  ▼              ▼              ▼
              Block off-topic    Return cached    Vector search   Generate with
              queries            response         help articles   GPT-4o-mini
```

### Key Components

**1. Help Article Ingestion** (`/api/help/ingest`)

Ingests help articles from `resources/help_articles.json` into Redis with vector embeddings:

```bash
curl -X POST http://localhost:8000/api/help/ingest
```

**2. Chat Endpoint** (`/api/help/chat`)

The main chat endpoint that processes user questions:

```bash
curl -X POST http://localhost:8000/api/help/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?", "use_cache": true}'
```

**3. HelpCenterEngine** (`backend/help_center.py`)

The core RAG engine that:
- Checks semantic cache for similar previous questions
- Searches help articles using vector similarity
- Generates responses using OpenAI GPT-4o-mini
- Stores responses in cache for future queries

### Try It!

1. Navigate to the Help Center in the UI
2. Ask questions like:
   - "How do I reset my password?"
   - "Why is my video buffering?"
   - "How to set up parental controls?"
3. Notice the response badges:
   - **LLM** - Fresh response from the language model
   - **Cached** - Retrieved from semantic cache
   - **Off-topic** - Blocked by guardrails

---

## Part 9: Guardrails & PII Protection

### Estimated time: **10 minutes**

The application includes two important safety features to protect users and ensure quality responses.

### Semantic Router Guardrails

Uses RedisVL's `SemanticRouter` to detect and block off-topic queries.

**How It Works:**

```python
# Define allowed topics with reference phrases
STREAMFLIX_ROUTE = Route(
    name="streamflix_support",
    references=[
        "reset password", "video buffering", "cancel subscription",
        "billing issues", "playback quality", "device support",
        # ... 50+ reference phrases
    ],
    distance_threshold=0.5,
)
```

**Allowed Queries** (processed normally):
- "How do I reset my password?"
- "Why is my video buffering?"
- "Cancel my subscription"

**Blocked Queries** (returns helpful redirect):
- "What's the weather today?"
- "Tell me about aliens"
- "Write Python code for sorting"

### PII Detection

Prevents caching of queries or responses containing personally identifiable information.

**Detected PII Types:**
- Email addresses (`user@example.com`)
- Phone numbers (`555-123-4567`)
- Social Security Numbers (`123-45-6789`)
- Credit card numbers (`4111-1111-1111-1111`)
- Account/member numbers
- IP addresses
- Dates of birth

**How It Works:**

```python
def should_cache(query: str, response: str) -> Tuple[bool, str]:
    """Check both query and response for PII before caching."""
    query_has_pii, query_pii_types = detect_pii(query)
    if query_has_pii:
        return False, f"PII detected in query: {query_pii_types}"
    
    response_has_pii, response_pii_types = detect_pii(response)
    if response_has_pii:
        return False, f"PII detected in response: {response_pii_types}"
    
    return True, "No PII detected"
```

**Examples:**
- `"My email is john@example.com"` → **Not cached** (email detected)
- `"How do I reset my password?"` → **Cached** (no PII)

### Guardrails Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Semantic Router Check                          │
│            Does query match StreamFlix topics?                   │
└─────────────────────────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
            ┌───────────┐           ┌───────────────┐
            │  Allowed  │           │   Blocked     │
            │           │           │               │
            │ Continue  │           │ Return help   │
            │ to RAG    │           │ message       │
            └───────────┘           └───────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                                │
│         Cache → Search → Generate → PII Check → Store            │
└─────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
            ┌───────────────┐                       ┌───────────────┐
            │   No PII      │                       │   PII Found   │
            │               │                       │               │
            │ Store in      │                       │ Skip cache    │
            │ cache         │                       │ (log reason)  │
            └───────────────┘                       └───────────────┘
```

---

## Topics Covered

| Topic | Description |
|-------|-------------|
| **Redis Cloud** | Fully managed Redis database in the cloud |
| **Vector Similarity Search** | KNN search using cosine distance |
| **Full-Text Search** | BM25-based keyword matching |
| **Hybrid Search** | Combining vector and text scoring |
| **Filtered Search** | Tag and numeric metadata filters |
| **Range Queries** | Distance threshold filtering |
| **Semantic Caching** | Redis-backed LLM response cache |
| **RAG Pipeline** | Retrieval Augmented Generation for Help Center |
| **Semantic Router** | Topic-based query routing with RedisVL |
| **PII Detection** | Protect sensitive data from caching |
| **RedisVL** | Redis Vector Library for Python |
| **OpenAI Integration** | GPT-4o-mini for response generation |
| **FastAPI** | Modern Python web framework |
| **React + TypeScript** | Frontend development |
| **Docker** | Containerized deployment |
| **RIOT** | Redis Input/Output Tools for data import |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Database | **Redis Cloud** (Essentials - Free Tier) |
| Backend | Python 3.11, FastAPI, Uvicorn |
| Vector Search | RedisVL, RediSearch |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI GPT-4o-mini |
| Guardrails | RedisVL SemanticRouter |
| Frontend | React 19, TypeScript, Vite |
| Deployment | Docker, Docker Compose, NGINX |

---

## Project Structure

```
movie-recommender-rag-semantic-cache-workshop/
├── backend/
│   ├── __init__.py
│   ├── config.py           # Configuration and schema
│   ├── main.py             # FastAPI application
│   ├── search_engine.py    # Movie search engine
│   ├── help_center.py      # Help Center RAG engine
│   ├── semantic_cache.py   # LLM response caching
│   ├── guardrails.py       # Semantic router & PII detection
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── api/            # API client
│   │   ├── components/     # React components
│   │   │   ├── HelpChat.tsx    # Help Center chat UI
│   │   │   ├── MovieCard.tsx   # Movie result card
│   │   │   └── ...
│   │   ├── pages/          # Page components
│   │   │   └── HelpCenter.tsx  # Help Center page
│   │   └── styles/         # CSS styles
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   └── import_data.sh      # RIOT data import script
├── resources/
│   ├── movies.json         # Movie dataset
│   └── help_articles.json  # Help Center articles
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
└── README.md
```

---

## Troubleshooting

### Connection Issues

If you see connection errors:

1. Verify your Redis Cloud database is running (check the dashboard)
2. Ensure your `REDIS_URL` is correct in the `.env` file
3. Check that your IP is whitelisted in Redis Cloud security settings

### RIOT Import Issues

If RIOT import fails:

1. Ensure RIOT is installed: `riot --version`
2. Verify your `REDIS_URL` environment variable is set correctly
3. Check that the `resources/movies.json` file exists

### Index Creation Issues

If `/api/create-index` fails:

1. Ensure RIOT import was run first (check for `movie:*` keys in Redis)
2. Check the backend logs for detailed error messages
3. Verify Redis Cloud connectivity with `/api/health`
4. Ensure your database has enough memory (30MB free tier should be sufficient)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Resources

- [Redis Cloud - Free Trial](https://redis.io/try-free)
- [Redis Vector Library (RedisVL)](https://github.com/redis/redis-vl-python)
- [RedisVL Semantic Router](https://docs.redis.com/latest/redisvl/user_guide/semantic_router/)
- [RedisVL LLM Semantic Cache](https://docs.redis.com/latest/redisvl/user_guide/llmcache/)
- [RediSearch Documentation](https://redis.io/docs/stack/search/)
- [RIOT - Redis Input/Output Tools](https://github.com/redis/riot)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)

---
