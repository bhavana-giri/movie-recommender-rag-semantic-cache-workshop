"""
Search Engine for Redis RAG Demo
Implements various search methods using redisvl
"""
import os
import warnings
from typing import List, Dict, Any, Optional

import pandas as pd
from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, RangeQuery, TextQuery
from redisvl.query.aggregate import AggregateHybridQuery
from redisvl.query.filter import Tag, Num, Text
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache

from config import (
    REDIS_URL,
    INDEX_NAME,
    INDEX_SCHEMA,
    EMBEDDING_MODEL,
    DEFAULT_NUM_RESULTS,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_HYBRID_ALPHA,
)

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MovieSearchEngine:
    """Search engine for movie recommendations using Redis Vector Search"""
    
    def __init__(self):
        """Initialize the search engine with Redis connection and embeddings"""
        self.client = Redis.from_url(REDIS_URL)
        self.schema = IndexSchema.from_dict(INDEX_SCHEMA)
        self.index = SearchIndex(self.schema, self.client)
        
        # Initialize the HuggingFace text vectorizer with embedding cache
        self.vectorizer = HFTextVectorizer(
            model=EMBEDDING_MODEL,
            cache=EmbeddingsCache(
                name="embedcache",
                ttl=600,
                redis_client=self.client,
            )
        )
    
    def check_connection(self) -> bool:
        """Check if Redis connection is working"""
        try:
            return self.client.ping()
        except Exception:
            return False
    
    def check_index_exists(self) -> bool:
        """Check if the movies index exists"""
        try:
            return self.index.exists()
        except Exception:
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index"""
        try:
            info = self.index.info()
            return {
                "name": info.get("index_name", INDEX_NAME),
                "num_docs": info.get("num_docs", 0),
                "indexing": info.get("indexing", 0)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string"""
        return self.vectorizer.embed(query)
    
    def vector_search(
        self,
        query: str,
        num_results: int = DEFAULT_NUM_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Standard KNN vector search
        Returns movies most semantically similar to the query
        """
        embedded_query = self._embed_query(query)
        
        vec_query = VectorQuery(
            vector=embedded_query,
            vector_field_name="vector",
            num_results=num_results,
            return_fields=["title", "genre", "rating", "description"],
            return_score=True,
        )
        
        results = self.index.query(vec_query)
        return self._format_results(results, "vector")
    
    def filtered_search(
        self,
        query: str,
        genre: Optional[str] = None,
        min_rating: Optional[int] = None,
        num_results: int = DEFAULT_NUM_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Vector search with tag and numeric filters
        Filters results by genre and/or minimum rating
        """
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
            num_results=num_results,
            return_fields=["title", "genre", "rating", "description"],
            return_score=True,
            filter_expression=filter_expression,
        )
        
        results = self.index.query(vec_query)
        return self._format_results(results, "filtered")
    
    def keyword_search(
        self,
        query: str,
        num_results: int = DEFAULT_NUM_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Full-text search using BM25 scoring
        Searches through title and description fields
        """
        text_query = TextQuery(
            text=query,
            text_field_name="description",
            text_scorer="BM25STD",
            num_results=num_results,
            return_fields=["title", "genre", "rating", "description"],
        )
        
        results = self.index.query(text_query)
        return self._format_results(results, "keyword")
    
    def hybrid_search(
        self,
        query: str,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        num_results: int = DEFAULT_NUM_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and text search
        Alpha controls the weight: higher = more vector, lower = more text
        """
        embedded_query = self._embed_query(query)
        
        hybrid_query = AggregateHybridQuery(
            text=query,
            text_field_name="description",
            text_scorer="BM25",
            vector=embedded_query,
            vector_field_name="vector",
            alpha=alpha,
            num_results=num_results,
            return_fields=["title", "genre", "rating", "description"],
        )
        
        results = self.index.query(hybrid_query)
        return self._format_results(results, "hybrid")
    
    def range_search(
        self,
        query: str,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        num_results: int = 20  # Get more results for range filtering
    ) -> List[Dict[str, Any]]:
        """
        Range query with distance threshold
        Returns only results within the specified semantic distance
        """
        embedded_query = self._embed_query(query)
        
        range_query = RangeQuery(
            vector=embedded_query,
            vector_field_name="vector",
            return_fields=["title", "genre", "rating", "description"],
            return_score=True,
            distance_threshold=distance_threshold,
        )
        
        results = self.index.query(range_query)
        return self._format_results(results, "range")
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        search_type: str
    ) -> List[Dict[str, Any]]:
        """Format search results for display"""
        formatted = []
        
        for result in results:
            formatted_result = {
                "title": result.get("title", "Unknown"),
                "genre": result.get("genre", "Unknown"),
                "rating": result.get("rating", "N/A"),
                "description": result.get("description", ""),
            }
            
            # Add score/distance based on search type
            if search_type in ["vector", "filtered", "range"]:
                distance = result.get("vector_distance", None)
                if distance:
                    formatted_result["distance"] = float(distance)
                    formatted_result["similarity"] = 1 - float(distance)
            
            elif search_type == "keyword":
                score = result.get("score", None)
                if score:
                    formatted_result["score"] = float(score)
            
            elif search_type == "hybrid":
                formatted_result["hybrid_score"] = float(result.get("hybrid_score", 0))
                formatted_result["vector_similarity"] = float(result.get("vector_similarity", 0))
                formatted_result["text_score"] = float(result.get("text_score", 0))
            
            formatted.append(formatted_result)
        
        return formatted
    
    def load_data(self, data_path: str = "resources/movies.json") -> bool:
        """
        Load movie data into Redis index
        Returns True if successful, False otherwise
        """
        try:
            # Read movie data
            df = pd.read_json(data_path)
            
            # Generate embeddings for descriptions
            df["vector"] = self.vectorizer.embed_many(
                df["description"].tolist(),
                as_buffer=True
            )
            
            # Create or overwrite index
            self.index.create(overwrite=True, drop=True)
            
            # Load data into index
            self.index.load(df.to_dict(orient="records"))
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False


# Singleton instance for the application
_search_engine = None


def get_search_engine() -> MovieSearchEngine:
    """Get or create the search engine singleton"""
    global _search_engine
    if _search_engine is None:
        _search_engine = MovieSearchEngine()
    return _search_engine

