"""
FastAPI Backend for Movie Recommender
Provides REST API endpoints for various search methods
"""
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .search_engine import get_search_engine, MovieSearchEngine
from .config import DEFAULT_NUM_RESULTS, DEFAULT_HYBRID_ALPHA, DEFAULT_DISTANCE_THRESHOLD

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="Redis Vector Search powered movie recommendation engine",
    version="1.0.0",
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    num_results: int = Field(DEFAULT_NUM_RESULTS, ge=1, le=50, description="Number of results")


class FilteredSearchRequest(SearchRequest):
    genre: Optional[str] = Field(None, description="Genre filter (action, comedy, romance)")
    min_rating: Optional[int] = Field(None, ge=1, le=10, description="Minimum rating filter")


class HybridSearchRequest(SearchRequest):
    alpha: float = Field(DEFAULT_HYBRID_ALPHA, ge=0, le=1, description="Balance between vector (1) and text (0)")


class RangeSearchRequest(SearchRequest):
    distance_threshold: float = Field(DEFAULT_DISTANCE_THRESHOLD, ge=0, le=1, description="Max distance threshold")


class MovieResult(BaseModel):
    title: str
    genre: str
    rating: Any
    description: str
    distance: Optional[float] = None
    similarity: Optional[float] = None
    score: Optional[float] = None
    hybrid_score: Optional[float] = None
    vector_similarity: Optional[float] = None
    text_score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[MovieResult]
    count: int
    search_type: str


class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    index_exists: bool
    index_info: Dict[str, Any]


# Dependency to get search engine
def get_engine() -> MovieSearchEngine:
    try:
        return get_search_engine()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search engine unavailable: {str(e)}")


# API Endpoints
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and Redis connection health"""
    try:
        engine = get_engine()
        redis_connected = engine.check_connection()
        index_exists = engine.check_index_exists()
        index_info = engine.get_index_info()
        
        return HealthResponse(
            status="healthy" if redis_connected else "degraded",
            redis_connected=redis_connected,
            index_exists=index_exists,
            index_info=index_info,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            redis_connected=False,
            index_exists=False,
            index_info={"error": str(e)},
        )


@app.post("/api/search/vector", response_model=SearchResponse, tags=["Search"])
async def vector_search(request: SearchRequest):
    """
    Semantic vector search using KNN
    Returns movies most similar to the query meaning
    """
    engine = get_engine()
    results = engine.vector_search(request.query, request.num_results)
    
    return SearchResponse(
        results=results,
        count=len(results),
        search_type="vector",
    )


@app.post("/api/search/filtered", response_model=SearchResponse, tags=["Search"])
async def filtered_search(request: FilteredSearchRequest):
    """
    Vector search with genre and rating filters
    Combines semantic similarity with metadata filtering
    """
    engine = get_engine()
    results = engine.filtered_search(
        query=request.query,
        genre=request.genre,
        min_rating=request.min_rating,
        num_results=request.num_results,
    )
    
    return SearchResponse(
        results=results,
        count=len(results),
        search_type="filtered",
    )


@app.post("/api/search/keyword", response_model=SearchResponse, tags=["Search"])
async def keyword_search(request: SearchRequest):
    """
    Full-text keyword search using BM25
    Returns movies matching exact keywords in description
    """
    engine = get_engine()
    results = engine.keyword_search(request.query, request.num_results)
    
    return SearchResponse(
        results=results,
        count=len(results),
        search_type="keyword",
    )


@app.post("/api/search/hybrid", response_model=SearchResponse, tags=["Search"])
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search combining vector and keyword
    Alpha controls balance: 1.0 = pure vector, 0.0 = pure text
    """
    engine = get_engine()
    results = engine.hybrid_search(
        query=request.query,
        alpha=request.alpha,
        num_results=request.num_results,
    )
    
    return SearchResponse(
        results=results,
        count=len(results),
        search_type="hybrid",
    )


@app.post("/api/search/range", response_model=SearchResponse, tags=["Search"])
async def range_search(request: RangeSearchRequest):
    """
    Range query with distance threshold
    Only returns results within semantic distance threshold
    """
    engine = get_engine()
    results = engine.range_search(
        query=request.query,
        distance_threshold=request.distance_threshold,
        num_results=request.num_results,
    )
    
    return SearchResponse(
        results=results,
        count=len(results),
        search_type="range",
    )


@app.post("/api/load-data", tags=["Admin"])
async def load_data():
    """Load movie data into Redis index"""
    engine = get_engine()
    success = engine.load_data("resources/movies.json")
    
    if success:
        return {"status": "success", "message": "Data loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load data")


@app.get("/api/genres", tags=["Metadata"])
async def get_genres():
    """Get available genre options"""
    return {
        "genres": ["all", "action", "comedy", "romance"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

