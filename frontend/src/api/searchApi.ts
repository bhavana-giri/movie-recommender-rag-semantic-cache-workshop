/**
 * API client for Movie Recommender search endpoints
 */

// Use relative path in production (nginx proxies /api to backend)
// Use localhost:8000 in development
const API_BASE = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api';

export interface MovieResult {
  title: string;
  genre: string;
  rating: number | string;
  description: string;
  distance?: number;
  similarity?: number;
  score?: number;
  hybrid_score?: number;
  vector_similarity?: number;
  text_score?: number;
}

export interface SearchResponse {
  results: MovieResult[];
  count: number;
  search_type: string;
}

export interface HealthResponse {
  status: string;
  redis_connected: boolean;
  index_exists: boolean;
  index_info: Record<string, unknown>;
}

export type SearchType = 'vector' | 'filtered' | 'keyword' | 'hybrid' | 'range';

interface BaseSearchParams {
  query: string;
  num_results?: number;
}

interface FilteredSearchParams extends BaseSearchParams {
  genre?: string;
  min_rating?: number;
}

interface HybridSearchParams extends BaseSearchParams {
  alpha?: number;
}

interface RangeSearchParams extends BaseSearchParams {
  distance_threshold?: number;
}

async function fetchApi<T>(endpoint: string, body: object): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function vectorSearch(params: BaseSearchParams): Promise<SearchResponse> {
  return fetchApi('/search/vector', params);
}

export async function filteredSearch(params: FilteredSearchParams): Promise<SearchResponse> {
  return fetchApi('/search/filtered', params);
}

export async function keywordSearch(params: BaseSearchParams): Promise<SearchResponse> {
  return fetchApi('/search/keyword', params);
}

export async function hybridSearch(params: HybridSearchParams): Promise<SearchResponse> {
  return fetchApi('/search/hybrid', params);
}

export async function rangeSearch(params: RangeSearchParams): Promise<SearchResponse> {
  return fetchApi('/search/range', params);
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

export async function loadData(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/load-data`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Load data failed: ${response.status}`);
  }
  return response.json();
}

export async function getGenres(): Promise<{ genres: string[] }> {
  const response = await fetch(`${API_BASE}/genres`);
  if (!response.ok) {
    throw new Error(`Get genres failed: ${response.status}`);
  }
  return response.json();
}

