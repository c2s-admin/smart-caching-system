# main.py - FastAPI application for Smart Caching
import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import redis
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
import httpx

from config import *
from cache_manager import SmartCacheManager
from monitoring import MetricsCollector

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query_text: str = Field(..., description="The search query text")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    use_cache: bool = Field(True, description="Whether to use caching")


class SearchResponse(BaseModel):
    source: str = Field(..., description="Data source (L1_cache, L2_cache, or qdrant_db)")
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float


class InvalidateRequest(BaseModel):
    vector_ids: List[str] = Field(..., description="List of vector IDs to invalidate")


class InvalidateResponse(BaseModel):
    status: str
    invalidated_queries_count: int
    processing_time: float


class CacheStatsResponse(BaseModel):
    stats: Dict[str, Any]
    timestamp: float


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    cache_stats: Dict[str, Any]


class WarmupRequest(BaseModel):
    queries: List[str] = Field(..., description="List of queries to pre-cache")


# --- Global instances ---
redis_client: Optional[redis.Redis] = None
qdrant_client: Optional[QdrantClient] = None
cache_manager: Optional[SmartCacheManager] = None
metrics_collector: Optional[MetricsCollector] = None
http_client: Optional[httpx.AsyncClient] = None


# --- Lifespan context manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global redis_client, qdrant_client, cache_manager, metrics_collector, http_client
    
    # Startup
    logger.info("Starting Smart Caching Service...")
    
    # Initialize Redis
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        redis_client.ping()
        logger.info("Successfully connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None
    
    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
        # Test connection
        qdrant_client.get_collections()
        logger.info("Successfully connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        qdrant_client = None
    
    # Initialize cache manager
    if redis_client and qdrant_client:
        cache_manager = SmartCacheManager(redis_client, qdrant_client)
        logger.info("Cache manager initialized")
    else:
        logger.error("Cannot initialize cache manager without Redis and Qdrant")
    
    # Initialize metrics collector
    if ENABLE_METRICS and redis_client:
        metrics_collector = MetricsCollector(redis_client, cache_manager)
        asyncio.create_task(metrics_collector.start())
        logger.info("Metrics collector started")
    
    # Initialize HTTP client for embedding service
    http_client = httpx.AsyncClient(timeout=30.0)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Smart Caching Service...")
    
    if metrics_collector:
        metrics_collector.stop()
    
    if http_client:
        await http_client.aclose()
    
    if redis_client:
        redis_client.close()
    
    logger.info("Shutdown complete")


# --- FastAPI app ---
app = FastAPI(
    title="Smart Caching API",
    description="Intelligent caching layer for vector search with LFU/LRU eviction and semantic similarity",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Helper functions ---
async def get_vector_embedding(text: str) -> List[float]:
    """Get vector embedding from the embedding service."""
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")
    
    try:
        response = await http_client.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json={"text": text, "model": EMBEDDING_MODEL}
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")


def extract_vector_ids(results: List[Dict]) -> List[str]:
    """Extract vector IDs from search results."""
    vector_ids = []
    for result in results:
        if isinstance(result, dict):
            # Handle different result formats
            if "id" in result:
                vector_ids.append(str(result["id"]))
            elif "point" in result and isinstance(result["point"], dict) and "id" in result["point"]:
                vector_ids.append(str(result["point"]["id"]))
    return vector_ids


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health status of all services."""
    services = {}
    
    # Check Redis
    try:
        redis_client.ping()
        services["redis"] = "healthy"
    except Exception:
        services["redis"] = "unhealthy"
    
    # Check Qdrant
    try:
        qdrant_client.get_collections()
        services["qdrant"] = "healthy"
    except Exception:
        services["qdrant"] = "unhealthy"
    
    # Check embedding service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{EMBEDDING_SERVICE_URL}/health", timeout=5.0)
            services["embedding"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        services["embedding"] = "unhealthy"
    
    # Get cache stats
    cache_stats = {}
    if cache_manager and services["redis"] == "healthy":
        cache_stats = cache_manager.get_cache_stats()
    
    overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        services=services,
        cache_stats=cache_stats
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform a vector search with intelligent caching.
    
    The search follows this strategy:
    1. L1 Cache: Exact query match
    2. L2 Cache: Semantic similarity match
    3. Qdrant: Direct database query
    """
    start_time = time.time()
    
    if not all([redis_client, qdrant_client, cache_manager]):
        raise HTTPException(status_code=503, detail="Backend services unavailable")
    
    # Skip cache if requested
    if not request.use_cache:
        logger.info("Cache disabled for this request")
        query_vector = await get_vector_embedding(request.query_text)
        
        # Direct Qdrant query
        try:
            qdrant_results = qdrant_client.search(
                collection_name=QDRANT_MAIN_COLLECTION,
                query_vector=query_vector,
                limit=request.top_k,
                query_filter=request.filters,
            )
            
            results = [result.model_dump() for result in qdrant_results]
            
            return SearchResponse(
                source="qdrant_db_direct",
                results=results,
                metadata={"cache_disabled": True},
                processing_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            raise HTTPException(status_code=500, detail="Search failed")
    
    # Normal cached search flow
    query_hash = cache_manager.get_query_hash(request.query_text)
    
    # 1. Check L1 exact cache
    cached_result = cache_manager.check_exact_cache(query_hash)
    if cached_result:
        return SearchResponse(
            source=cached_result["source"],
            results=cached_result["results"],
            metadata=cached_result.get("metadata", {}),
            processing_time=time.time() - start_time
        )
    
    # Get query vector for L2 check and potential Qdrant query
    query_vector = await get_vector_embedding(request.query_text)
    
    # 2. Check L2 semantic cache
    semantic_result = cache_manager.check_semantic_cache(query_vector, request.query_text)
    if semantic_result:
        return SearchResponse(
            source=semantic_result["source"],
            results=semantic_result["results"],
            metadata=semantic_result.get("metadata", {}),
            processing_time=time.time() - start_time
        )
    
    # 3. Cache miss - query Qdrant
    logger.info(f"Cache MISS for query: '{request.query_text[:50]}...'")
    
    try:
        qdrant_results = qdrant_client.search(
            collection_name=QDRANT_MAIN_COLLECTION,
            query_vector=query_vector,
            limit=request.top_k,
            query_filter=request.filters,
        )
        
        # Convert results to serializable format
        results = [result.model_dump() for result in qdrant_results]
        
        # Extract vector IDs for invalidation tracking
        vector_ids = extract_vector_ids(results)
        
        # Populate cache
        cache_metadata = cache_manager.populate_cache(
            query_hash=query_hash,
            query_text=request.query_text,
            query_vector=query_vector,
            results=results,
            vector_ids=vector_ids
        )
        
        return SearchResponse(
            source="qdrant_db",
            results=results,
            metadata=cache_metadata,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/invalidate", response_model=InvalidateResponse)
async def invalidate(request: InvalidateRequest):
    """
    Invalidate cached queries containing the specified vector IDs.
    
    This should be called whenever vectors are updated or deleted.
    """
    start_time = time.time()
    
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager unavailable")
    
    invalidated_count = cache_manager.invalidate_vectors(request.vector_ids)
    
    logger.info(f"Invalidated {invalidated_count} cached queries for {len(request.vector_ids)} vectors")
    
    return InvalidateResponse(
        status="success",
        invalidated_queries_count=invalidated_count,
        processing_time=time.time() - start_time
    )


@app.get("/stats", response_model=CacheStatsResponse)
async def get_stats():
    """Get comprehensive cache statistics."""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager unavailable")
    
    stats = cache_manager.get_cache_stats()
    
    return CacheStatsResponse(
        stats=stats,
        timestamp=time.time()
    )


@app.post("/cache/warmup")
async def warmup_cache(request: WarmupRequest, background_tasks: BackgroundTasks):
    """
    Pre-populate cache with specified queries.
    
    Useful for warming up cache with popular queries.
    """
    if not all([cache_manager, http_client]):
        raise HTTPException(status_code=503, detail="Required services unavailable")
    
    # Add warmup task to background
    background_tasks.add_task(perform_cache_warmup, request.queries)
    
    return {
        "status": "warmup_started",
        "query_count": len(request.queries)
    }


async def perform_cache_warmup(queries: List[str]):
    """Background task to warm up cache."""
    logger.info(f"Starting cache warmup with {len(queries)} queries")
    
    success_count = 0
    for query_text in queries:
        try:
            # Use the search endpoint to populate cache
            search_request = SearchRequest(query_text=query_text, top_k=10)
            await search(search_request)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to warmup query '{query_text[:50]}...': {e}")
    
    logger.info(f"Cache warmup completed. Success: {success_count}/{len(queries)}")


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached entries."""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager unavailable")
    
    # Get all cache entries
    all_hashes = redis_client.zrange(FREQUENCY_KEY, 0, -1)
    
    if all_hashes:
        cache_manager._evict_hashes(all_hashes)
    
    # Reset stats
    cache_manager.stats.reset_stats()
    
    return {
        "status": "cache_cleared",
        "entries_removed": len(all_hashes)
    }


@app.get("/cache/top-queries")
async def get_top_queries(limit: int = Query(10, ge=1, le=100)):
    """Get the most frequently cached queries."""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager unavailable")
    
    top_hashes = redis_client.zrevrange(FREQUENCY_KEY, 0, limit - 1, withscores=True)
    
    results = []
    for query_hash, frequency in top_hashes:
        cache_key = f"{CACHE_PREFIX}{query_hash}"
        query_text = redis_client.hget(cache_key, "query_text")
        if query_text:
            results.append({
                "query_text": query_text,
                "frequency": int(frequency),
                "hash": query_hash[:8]
            })
    
    return {"top_queries": results}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL.lower(),
        reload=False
    )