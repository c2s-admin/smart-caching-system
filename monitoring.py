# monitoring.py - Metrics collection and monitoring
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
import redis
from redis import Redis

from config import *
from cache_manager import SmartCacheManager

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and stores cache performance metrics."""
    
    def __init__(self, redis_client: Redis, cache_manager: SmartCacheManager):
        self.redis = redis_client
        self.cache_manager = cache_manager
        self.running = False
        self.task = None
        
    async def start(self):
        """Start the metrics collection loop."""
        self.running = True
        self.task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")
        
    def stop(self):
        """Stop the metrics collection."""
        self.running = False
        if self.task:
            self.task.cancel()
        logger.info("Metrics collector stopped")
        
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(METRICS_UPDATE_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(METRICS_UPDATE_INTERVAL)
                
    async def _collect_metrics(self):
        """Collect current metrics and store in Redis."""
        timestamp = time.time()
        
        # Get cache stats
        stats = self.cache_manager.get_cache_stats()
        
        # Calculate additional metrics
        metrics = {
            "timestamp": timestamp,
            "cache_size": stats["current_size"],
            "cache_utilization": stats["utilization"],
            "hit_rate": stats["hit_rate"],
            "total_requests": stats["total_hits"] + stats["total_misses"],
            "l1_hit_rate": (stats["l1_hits"] / (stats["total_hits"] + stats["total_misses"]) * 100) 
                          if (stats["total_hits"] + stats["total_misses"]) > 0 else 0,
            "l2_hit_rate": (stats["l2_hits"] / (stats["total_hits"] + stats["total_misses"]) * 100)
                          if (stats["total_hits"] + stats["total_misses"]) > 0 else 0,
        }
        
        # Store current metrics
        self._store_metrics(metrics)
        
        # Store time series data
        self._store_time_series(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store current metrics in Redis."""
        pipe = self.redis.pipeline()
        
        for key, value in metrics.items():
            pipe.set(f"{METRICS_PREFIX}current:{key}", value)
            pipe.expire(f"{METRICS_PREFIX}current:{key}", 300)  # 5 min TTL
            
        pipe.execute()
        
    def _store_time_series(self, metrics: Dict[str, Any]):
        """Store time series data for historical analysis."""
        timestamp = metrics["timestamp"]
        
        pipe = self.redis.pipeline()
        
        # Store each metric as a sorted set with timestamp as score
        for key in ["hit_rate", "cache_utilization", "total_requests"]:
            if key in metrics:
                pipe.zadd(
                    f"{METRICS_PREFIX}timeseries:{key}",
                    {f"{timestamp}:{metrics[key]}": timestamp}
                )
                # Keep only last 24 hours
                pipe.zremrangebyscore(
                    f"{METRICS_PREFIX}timeseries:{key}",
                    "-inf",
                    timestamp - 86400
                )
                
        pipe.execute()
        
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and raise alerts."""
        alerts = []
        
        # Low hit rate alert
        if metrics["hit_rate"] < 50 and metrics["total_requests"] > 100:
            alerts.append({
                "level": "warning",
                "metric": "hit_rate",
                "value": metrics["hit_rate"],
                "threshold": 50,
                "message": f"Cache hit rate low: {metrics['hit_rate']:.1f}%"
            })
            
        # High cache utilization alert
        if metrics["cache_utilization"] > 90:
            alerts.append({
                "level": "warning",
                "metric": "cache_utilization",
                "value": metrics["cache_utilization"],
                "threshold": 90,
                "message": f"Cache nearly full: {metrics['cache_utilization']:.1f}% utilized"
            })
            
        # Store alerts
        if alerts:
            for alert in alerts:
                logger.warning(f"Alert: {alert['message']}")
                self.redis.lpush(f"{METRICS_PREFIX}alerts", str(alert))
                self.redis.ltrim(f"{METRICS_PREFIX}alerts", 0, 99)  # Keep last 100 alerts
                
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        pipe = self.redis.pipeline()
        
        # Get all current metrics
        keys = self.redis.keys(f"{METRICS_PREFIX}current:*")
        for key in keys:
            pipe.get(key)
            
        values = pipe.execute()
        
        # Build summary
        summary = {}
        for i, key in enumerate(keys):
            metric_name = key.split(":")[-1]
            try:
                value = float(values[i])
                summary[metric_name] = value
            except (ValueError, TypeError):
                summary[metric_name] = values[i]
                
        # Add recent alerts
        recent_alerts = self.redis.lrange(f"{METRICS_PREFIX}alerts", 0, 9)
        summary["recent_alerts"] = [eval(alert) for alert in recent_alerts]
        
        return summary
        
    def get_time_series(self, metric: str, hours: int = 1) -> List[Dict[str, float]]:
        """Get time series data for a specific metric."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Get data from sorted set
        data = self.redis.zrangebyscore(
            f"{METRICS_PREFIX}timeseries:{metric}",
            start_time,
            end_time,
            withscores=True
        )
        
        # Parse and format results
        results = []
        for item, score in data:
            parts = item.split(":")
            if len(parts) >= 2:
                results.append({
                    "timestamp": float(parts[0]),
                    "value": float(parts[1])
                })
                
        return results


class PerformanceTracker:
    """Tracks performance metrics for individual operations."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        
    def track_operation(self, operation: str, duration: float, success: bool = True):
        """Track an operation's performance."""
        timestamp = time.time()
        
        # Update operation stats
        pipe = self.redis.pipeline()
        
        # Increment counters
        pipe.hincrby(f"{METRICS_PREFIX}ops:{operation}", "total", 1)
        if success:
            pipe.hincrby(f"{METRICS_PREFIX}ops:{operation}", "success", 1)
        else:
            pipe.hincrby(f"{METRICS_PREFIX}ops:{operation}", "failure", 1)
            
        # Track duration
        pipe.lpush(f"{METRICS_PREFIX}ops:{operation}:durations", duration)
        pipe.ltrim(f"{METRICS_PREFIX}ops:{operation}:durations", 0, 999)  # Keep last 1000
        
        # Update percentiles periodically (every 100 operations)
        total = int(pipe.hget(f"{METRICS_PREFIX}ops:{operation}", "total") or 0)
        if total % 100 == 0:
            self._update_percentiles(operation)
            
        pipe.execute()
        
    def _update_percentiles(self, operation: str):
        """Update percentile calculations for an operation."""
        # Get recent durations
        durations = self.redis.lrange(f"{METRICS_PREFIX}ops:{operation}:durations", 0, -1)
        if not durations:
            return
            
        # Convert to floats and sort
        durations = sorted([float(d) for d in durations])
        
        # Calculate percentiles
        p50_idx = int(len(durations) * 0.5)
        p95_idx = int(len(durations) * 0.95)
        p99_idx = int(len(durations) * 0.99)
        
        pipe = self.redis.pipeline()
        pipe.hset(f"{METRICS_PREFIX}ops:{operation}", "p50", durations[p50_idx])
        pipe.hset(f"{METRICS_PREFIX}ops:{operation}", "p95", durations[p95_idx])
        pipe.hset(f"{METRICS_PREFIX}ops:{operation}", "p99", durations[p99_idx])
        pipe.hset(f"{METRICS_PREFIX}ops:{operation}", "avg", sum(durations) / len(durations))
        pipe.execute()
        
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        stats = self.redis.hgetall(f"{METRICS_PREFIX}ops:{operation}")
        
        # Convert to appropriate types
        result = {}
        for key, value in stats.items():
            try:
                result[key] = float(value) if "." in value else int(value)
            except ValueError:
                result[key] = value
                
        return result