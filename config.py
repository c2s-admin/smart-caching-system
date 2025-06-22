# config.py - Configuration for Smart Caching System
import os
from typing import Optional

# --- Connection Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "10.10.180.170")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "1"))  # Use DB 1 for cache
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

QDRANT_HOST = os.getenv("QDRANT_HOST", "10.10.180.172")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

API_HOST = os.getenv("API_HOST", "10.10.180.173")
API_PORT = int(os.getenv("API_PORT", "8006"))  # New port for smart caching

# --- Qdrant Collections ---
QDRANT_MAIN_COLLECTION = os.getenv("QDRANT_MAIN_COLLECTION", "memory_vectors")
QDRANT_META_COLLECTION = os.getenv("QDRANT_META_COLLECTION", "cached_queries_meta")

# --- Vector Configuration ---
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "384"))  # For MiniLM-L6-v2
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://10.10.180.173:8000")

# --- Caching Strategy Parameters ---
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "10000"))  # Max number of queries to cache
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.98"))

# TTL Configuration
BASE_TTL = int(os.getenv("BASE_TTL", "3600"))  # 1 hour
FREQUENCY_MULTIPLIER = int(os.getenv("FREQUENCY_MULTIPLIER", "600"))
RECENCY_WINDOW_SECONDS = int(os.getenv("RECENCY_WINDOW_SECONDS", "600"))  # 10 minutes
RECENCY_BOOST = int(os.getenv("RECENCY_BOOST", "1800"))  # 30 minutes
MAX_TTL = int(os.getenv("MAX_TTL", "86400"))  # 24 hours max

# --- Eviction Strategy ---
EVICTION_BATCH_SIZE = int(os.getenv("EVICTION_BATCH_SIZE", "100"))  # Evict multiple at once when full
LFU_WEIGHT = float(os.getenv("LFU_WEIGHT", "0.7"))  # Weight for frequency in hybrid score
LRU_WEIGHT = float(os.getenv("LRU_WEIGHT", "0.3"))  # Weight for recency in hybrid score

# --- Performance Settings ---
REDIS_PIPELINE_SIZE = int(os.getenv("REDIS_PIPELINE_SIZE", "100"))
BATCH_INVALIDATION_SIZE = int(os.getenv("BATCH_INVALIDATION_SIZE", "1000"))
CACHE_WARMUP_SIZE = int(os.getenv("CACHE_WARMUP_SIZE", "100"))  # Popular queries to pre-cache

# --- Monitoring ---
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_UPDATE_INTERVAL = int(os.getenv("METRICS_UPDATE_INTERVAL", "60"))  # seconds

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Cache Key Prefixes ---
CACHE_PREFIX = "cache:query:"
FREQUENCY_KEY = "cache:frequency_lfu"
RECENCY_KEY = "cache:recency_lru"
INVALIDATION_PREFIX = "invalidation:vector:"
STATS_PREFIX = "cache:stats:"
METRICS_PREFIX = "cache:metrics:"

# --- Health Check ---
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))