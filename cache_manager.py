# cache_manager.py - Core caching logic
import hashlib
import json
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Any
import redis
from redis import Redis
from redis.client import Pipeline
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np

from config import *

logger = logging.getLogger(__name__)


class SmartCacheManager:
    """Manages the smart caching layer for vector search queries."""
    
    def __init__(self, redis_client: Redis, qdrant_client: QdrantClient):
        self.redis = redis_client
        self.qdrant = qdrant_client
        self.stats = CacheStats(redis_client)
        
        # Ensure meta collection exists
        self._ensure_meta_collection()
        
    def _ensure_meta_collection(self):
        """Ensure the cached queries meta collection exists in Qdrant."""
        try:
            self.qdrant.get_collection(collection_name=QDRANT_META_COLLECTION)
            logger.info(f"Meta collection '{QDRANT_META_COLLECTION}' already exists.")
        except Exception:
            self.qdrant.recreate_collection(
                collection_name=QDRANT_META_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant meta collection: {QDRANT_META_COLLECTION}")
    
    def get_query_hash(self, query_text: str) -> str:
        """Generate a unique hash for a query."""
        return hashlib.sha256(query_text.encode()).hexdigest()
    
    def calculate_ttl(self, frequency: int, last_access: float) -> int:
        """Calculate adaptive TTL based on frequency and recency."""
        # Logarithmic scaling for frequency
        log_freq = math.log(frequency + 1)  # +1 to avoid log(0)
        ttl = BASE_TTL + (log_freq * FREQUENCY_MULTIPLIER)
        
        # Recency boost if accessed recently
        time_since_access = time.time() - last_access
        if time_since_access < RECENCY_WINDOW_SECONDS:
            ttl += RECENCY_BOOST
        
        # Cap at MAX_TTL
        return min(int(ttl), MAX_TTL)
    
    def check_exact_cache(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Check L1 exact match cache."""
        cache_key = f"{CACHE_PREFIX}{query_hash}"
        cached_data = self.redis.hgetall(cache_key)
        
        if cached_data:
            logger.info(f"L1 Cache HIT for hash: {query_hash[:8]}...")
            self.stats.record_hit("L1")
            
            # Update access patterns
            pipe = self.redis.pipeline()
            current_freq = int(cached_data.get("frequency", "0"))
            new_freq = current_freq + 1
            now = time.time()
            
            # Update cache entry
            pipe.hincrby(cache_key, "frequency", 1)
            pipe.hset(cache_key, "last_access", now)
            
            # Update tracking sorted sets
            pipe.zadd(FREQUENCY_KEY, {query_hash: new_freq})
            pipe.zadd(RECENCY_KEY, {query_hash: now})
            
            # Refresh TTL
            new_ttl = self.calculate_ttl(new_freq, now)
            pipe.expire(cache_key, new_ttl)
            
            pipe.execute()
            
            return {
                "source": "L1_cache",
                "results": json.loads(cached_data["results"]),
                "metadata": {
                    "frequency": new_freq,
                    "ttl": new_ttl
                }
            }
        
        return None
    
    def check_semantic_cache(self, query_vector: List[float], query_text: str) -> Optional[Dict[str, Any]]:
        """Check L2 semantic similarity cache."""
        try:
            similar_queries = self.qdrant.search(
                collection_name=QDRANT_META_COLLECTION,
                query_vector=query_vector,
                limit=1,
                score_threshold=SEMANTIC_SIMILARITY_THRESHOLD,
            )
            
            if similar_queries:
                similar_hash = similar_queries[0].id
                similar_score = similar_queries[0].score
                logger.info(f"L2 Semantic match found with score {similar_score:.4f}")
                
                # Fetch the cached results for the similar query
                return self._fetch_and_update_similar_cache(similar_hash)
                
        except Exception as e:
            logger.error(f"Error during semantic cache check: {e}")
            
        return None
    
    def _fetch_and_update_similar_cache(self, similar_hash: str) -> Optional[Dict[str, Any]]:
        """Fetch and update stats for a semantically similar cached query."""
        cache_key = f"{CACHE_PREFIX}{similar_hash}"
        cached_data = self.redis.hgetall(cache_key)
        
        if cached_data:
            self.stats.record_hit("L2")
            
            # Update stats for the similar query
            pipe = self.redis.pipeline()
            current_freq = int(cached_data.get("frequency", "0"))
            new_freq = current_freq + 1
            now = time.time()
            
            pipe.hincrby(cache_key, "frequency", 1)
            pipe.hset(cache_key, "last_access", now)
            pipe.zadd(FREQUENCY_KEY, {similar_hash: new_freq})
            pipe.zadd(RECENCY_KEY, {similar_hash: now})
            
            new_ttl = self.calculate_ttl(new_freq, now)
            pipe.expire(cache_key, new_ttl)
            
            pipe.execute()
            
            return {
                "source": "L2_cache",
                "results": json.loads(cached_data["results"]),
                "metadata": {
                    "frequency": new_freq,
                    "ttl": new_ttl,
                    "similar_hash": similar_hash[:8]
                }
            }
        
        return None
    
    def populate_cache(self, query_hash: str, query_text: str, query_vector: List[float], 
                      results: List[Dict], vector_ids: List[str]) -> Dict[str, Any]:
        """Add new query results to cache."""
        # Check if we need to evict
        current_size = self.redis.zcard(FREQUENCY_KEY)
        if current_size >= MAX_CACHE_SIZE:
            evicted_count = self.evict_entries()
            logger.info(f"Evicted {evicted_count} entries before adding new cache entry")
        
        # Prepare cache data
        now = time.time()
        initial_ttl = self.calculate_ttl(1, now)
        results_json = json.dumps(results)
        
        # Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        cache_key = f"{CACHE_PREFIX}{query_hash}"
        
        # Store main cache entry
        pipe.hset(cache_key, mapping={
            "results": results_json,
            "frequency": 1,
            "last_access": now,
            "query_text": query_text,
            "created_at": now
        })
        pipe.expire(cache_key, initial_ttl)
        
        # Update tracking sets
        pipe.zadd(FREQUENCY_KEY, {query_hash: 1})
        pipe.zadd(RECENCY_KEY, {query_hash: now})
        
        # Update invalidation index
        for vector_id in vector_ids:
            pipe.sadd(f"{INVALIDATION_PREFIX}{vector_id}", query_hash)
        
        # Update stats
        pipe.incr(f"{STATS_PREFIX}total_cached")
        
        pipe.execute()
        
        # Add to Qdrant meta collection
        try:
            self.qdrant.upsert(
                collection_name=QDRANT_META_COLLECTION,
                points=[PointStruct(
                    id=query_hash,
                    vector=query_vector,
                    payload={
                        "query_text": query_text,
                        "created_at": now
                    }
                )],
                wait=True
            )
        except Exception as e:
            logger.error(f"Error adding to meta collection: {e}")
        
        self.stats.record_miss()
        
        return {
            "cached": True,
            "ttl": initial_ttl,
            "cache_size": current_size + 1
        }
    
    def evict_entries(self, count: int = None) -> int:
        """Evict least valuable cache entries using hybrid LFU/LRU strategy."""
        if count is None:
            count = EVICTION_BATCH_SIZE
        
        # Get all cache entries with their scores
        all_entries = self._get_all_cache_entries_with_scores()
        
        if not all_entries:
            return 0
        
        # Sort by hybrid score (lower is worse)
        all_entries.sort(key=lambda x: x[1])
        
        # Select entries to evict
        entries_to_evict = all_entries[:count]
        hashes_to_evict = [entry[0] for entry in entries_to_evict]
        
        # Perform eviction
        self._evict_hashes(hashes_to_evict)
        
        return len(hashes_to_evict)
    
    def _get_all_cache_entries_with_scores(self) -> List[Tuple[str, float]]:
        """Calculate hybrid scores for all cache entries."""
        # Get frequency scores
        freq_scores = self.redis.zrange(FREQUENCY_KEY, 0, -1, withscores=True)
        
        # Get recency scores
        recency_scores = {}
        for query_hash, _ in freq_scores:
            score = self.redis.zscore(RECENCY_KEY, query_hash)
            if score:
                recency_scores[query_hash] = score
        
        # Calculate hybrid scores
        now = time.time()
        max_freq = max(score for _, score in freq_scores) if freq_scores else 1
        
        results = []
        for query_hash, freq in freq_scores:
            if query_hash in recency_scores:
                # Normalize frequency (0-1)
                norm_freq = freq / max_freq
                
                # Normalize recency (0-1, where 1 is most recent)
                age = now - recency_scores[query_hash]
                norm_recency = 1.0 / (1.0 + age / 3600)  # Decay over hours
                
                # Calculate hybrid score
                hybrid_score = (LFU_WEIGHT * norm_freq) + (LRU_WEIGHT * norm_recency)
                results.append((query_hash, hybrid_score))
        
        return results
    
    def _evict_hashes(self, query_hashes: List[str]):
        """Remove cache entries and all associated data."""
        if not query_hashes:
            return
        
        pipe = self.redis.pipeline()
        
        for query_hash in query_hashes:
            # Remove main cache entry
            cache_key = f"{CACHE_PREFIX}{query_hash}"
            
            # Get vector IDs for invalidation cleanup
            cached_data = self.redis.hget(cache_key, "results")
            if cached_data:
                try:
                    results = json.loads(cached_data)
                    # Extract vector IDs from results (adjust based on actual format)
                    for result in results:
                        if isinstance(result, dict) and "id" in result:
                            pipe.srem(f"{INVALIDATION_PREFIX}{result['id']}", query_hash)
                except Exception as e:
                    logger.error(f"Error parsing cached results for cleanup: {e}")
            
            # Remove cache entry
            pipe.delete(cache_key)
            
            # Remove from tracking sets
            pipe.zrem(FREQUENCY_KEY, query_hash)
            pipe.zrem(RECENCY_KEY, query_hash)
        
        pipe.execute()
        
        # Remove from Qdrant meta collection
        try:
            self.qdrant.delete(
                collection_name=QDRANT_META_COLLECTION,
                points_selector=models.PointIdsList(points=query_hashes),
                wait=True
            )
        except Exception as e:
            logger.error(f"Error removing from meta collection: {e}")
        
        logger.info(f"Evicted {len(query_hashes)} cache entries")
    
    def invalidate_vectors(self, vector_ids: List[str]) -> int:
        """Invalidate all cached queries containing the specified vectors."""
        all_hashes_to_invalidate = set()
        
        # Collect all affected query hashes
        pipe = self.redis.pipeline()
        for vector_id in vector_ids:
            pipe.smembers(f"{INVALIDATION_PREFIX}{vector_id}")
        
        results = pipe.execute()
        
        for query_hashes in results:
            if query_hashes:
                all_hashes_to_invalidate.update(query_hashes)
        
        # Remove the queries
        if all_hashes_to_invalidate:
            self._evict_hashes(list(all_hashes_to_invalidate))
        
        # Clean up invalidation sets
        pipe = self.redis.pipeline()
        for vector_id in vector_ids:
            pipe.delete(f"{INVALIDATION_PREFIX}{vector_id}")
        pipe.execute()
        
        return len(all_hashes_to_invalidate)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        pipe = self.redis.pipeline()
        pipe.zcard(FREQUENCY_KEY)
        pipe.get(f"{STATS_PREFIX}total_cached")
        pipe.get(f"{STATS_PREFIX}total_hits")
        pipe.get(f"{STATS_PREFIX}total_misses")
        pipe.get(f"{STATS_PREFIX}l1_hits")
        pipe.get(f"{STATS_PREFIX}l2_hits")
        
        results = pipe.execute()
        
        current_size = results[0] or 0
        total_cached = int(results[1] or 0)
        total_hits = int(results[2] or 0)
        total_misses = int(results[3] or 0)
        l1_hits = int(results[4] or 0)
        l2_hits = int(results[5] or 0)
        
        total_requests = total_hits + total_misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Get top queries
        top_queries = self.redis.zrevrange(FREQUENCY_KEY, 0, 9, withscores=True)
        
        return {
            "current_size": current_size,
            "max_size": MAX_CACHE_SIZE,
            "utilization": (current_size / MAX_CACHE_SIZE * 100),
            "total_cached": total_cached,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "l1_hits": l1_hits,
            "l2_hits": l2_hits,
            "hit_rate": hit_rate,
            "top_queries": [
                {"hash": h[:8], "frequency": int(f)} 
                for h, f in top_queries
            ]
        }


class CacheStats:
    """Helper class for tracking cache statistics."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    def record_hit(self, cache_level: str):
        """Record a cache hit."""
        pipe = self.redis.pipeline()
        pipe.incr(f"{STATS_PREFIX}total_hits")
        pipe.incr(f"{STATS_PREFIX}{cache_level.lower()}_hits")
        pipe.execute()
    
    def record_miss(self):
        """Record a cache miss."""
        self.redis.incr(f"{STATS_PREFIX}total_misses")
    
    def reset_stats(self):
        """Reset all statistics."""
        keys = self.redis.keys(f"{STATS_PREFIX}*")
        if keys:
            self.redis.delete(*keys)