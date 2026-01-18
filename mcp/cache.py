"""
═══════════════════════════════════════════════════════════════════════════════
JOB AUTOMATION MCP - CACHE & METRICS MODULE
═══════════════════════════════════════════════════════════════════════════════

Performance optimization and observability for MCP.

Components:
- RedisCache: Distributed caching layer for sessions and context
- Compressor: ZSTD compression for large payloads
- MetricsCollector: Prometheus-compatible metrics

Features:
- Session and context item caching with TTL
- Automatic compression for payloads > 10KB
- Cache invalidation strategies
- Prometheus metrics export
- Health checks
- Cache hit/miss tracking

Author: Job Automation Team
Version: 2.0 Enterprise
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# Redis
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Compression
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    zstd = None


logger = logging.getLogger("mcp.cache")


# ═══════════════════════════════════════════════════════════════════════════════
# REDIS CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class RedisCache:
    """
    Distributed caching layer using Redis.

    Features:
    - Session caching with TTL
    - Context item list caching
    - Automatic serialization/deserialization
    - Namespace isolation
    - Bulk operations
    - Cache invalidation
    """

    def __init__(
        self,
        redis_url: str = None,
        default_ttl: int = 300,  # 5 minutes
        namespace: str = "mcp"
    ):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            namespace: Cache key namespace/prefix
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.redis_client = None

        if not REDIS_AVAILABLE:
            logger.warning("⚠️  Redis not available - caching disabled")
            return

        # Initialize Redis client
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info("✅ RedisCache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None

    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.namespace}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value (deserialized) or None
        """
        if not self.redis_client:
            return None

        try:
            full_key = self._make_key(key)
            value = await self.redis_client.get(full_key)

            if value is None:
                return None

            # Deserialize JSON
            return json.loads(value)

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: TTL in seconds (uses default if None)

        Returns:
            True if successful
        """
        if not self.redis_client:
            return False

        try:
            full_key = self._make_key(key)
            serialized = json.dumps(value)
            ttl_seconds = ttl if ttl is not None else self.default_ttl

            await self.redis_client.setex(full_key, ttl_seconds, serialized)
            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        if not self.redis_client:
            return False

        try:
            full_key = self._make_key(key)
            result = await self.redis_client.delete(full_key)
            return bool(result)

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "session:*")

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            full_pattern = self._make_key(pattern)
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=full_pattern,
                    count=100
                )

                if keys:
                    deleted += await self.redis_client.delete(*keys)

                if cursor == 0:
                    break

            return deleted

        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Domain-specific cache methods
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data."""
        return await self.get(f"session:{session_id}")

    async def set_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: int = 600  # 10 minutes
    ) -> bool:
        """Cache session data."""
        return await self.set(f"session:{session_id}", session_data, ttl)

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session cache."""
        return await self.delete(f"session:{session_id}")

    async def invalidate_context(self, session_id: str) -> int:
        """Invalidate all context cache for session."""
        return await self.delete_pattern(f"ctx:{session_id}:*")

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self.redis_client:
            return False

        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return {"available": False}

        try:
            info = await self.redis_client.info("stats")
            return {
                "available": True,
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"available": False, "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100.0

    async def cleanup(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class Compressor:
    """
    ZSTD compression for large payloads.

    Automatically compresses data > threshold size.
    Transparent compression/decompression.
    """

    def __init__(
        self,
        compression_level: int = 3,
        threshold_bytes: int = 10240  # 10KB
    ):
        """
        Initialize compressor.

        Args:
            compression_level: ZSTD compression level (1-22)
            threshold_bytes: Min size to trigger compression
        """
        self.compression_level = compression_level
        self.threshold_bytes = threshold_bytes
        self.enabled = ZSTD_AVAILABLE

        if not self.enabled:
            logger.warning("⚠️  zstandard not available - compression disabled")
            return

        # Initialize compressor/decompressor
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.decompressor = zstd.ZstdDecompressor()

        logger.info(f"✅ Compressor initialized (level={compression_level}, threshold={threshold_bytes}B)")

    def compress(self, data: str) -> str:
        """
        Compress string data.

        Args:
            data: String to compress

        Returns:
            Base64-encoded compressed data or original if compression disabled/failed
        """
        if not self.enabled:
            return data

        if len(data) < self.threshold_bytes:
            return data

        try:
            # Compress
            compressed = self.compressor.compress(data.encode('utf-8'))

            # Base64 encode for safe storage
            import base64
            encoded = base64.b64encode(compressed).decode('ascii')

            compression_ratio = len(data) / len(compressed)
            logger.debug(f"Compressed {len(data)} → {len(compressed)} bytes (ratio: {compression_ratio:.2f}x)")

            return encoded

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data

    def decompress(self, data: str) -> str:
        """
        Decompress string data.

        Args:
            data: Base64-encoded compressed data

        Returns:
            Decompressed string or original if not compressed
        """
        if not self.enabled:
            return data

        try:
            # Base64 decode
            import base64
            compressed = base64.b64decode(data.encode('ascii'))

            # Decompress
            decompressed = self.decompressor.decompress(compressed)

            return decompressed.decode('utf-8')

        except Exception as e:
            # If decompression fails, assume data wasn't compressed
            logger.debug(f"Decompression skipped (data not compressed): {e}")
            return data

    def should_compress(self, data: str) -> bool:
        """Check if data should be compressed based on size."""
        return self.enabled and len(data) >= self.threshold_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """
    Prometheus-compatible metrics collector.

    Collects:
    - Request counts by endpoint and status
    - Request duration histograms
    - Cache hit/miss rates
    - Database query counts
    - Error rates
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            # Counters
            "requests_total": {},
            "cache_hits": 0,
            "cache_misses": 0,
            "db_queries": 0,
            "errors_total": 0,

            # Histograms (simplified - store samples)
            "request_duration_samples": [],
            "db_query_duration_samples": [],

            # Gauges
            "active_sessions": 0,
            "total_context_items": 0,
        }

        self.start_time = time.time()

        logger.info("✅ MetricsCollector initialized")

    def increment_counter(self, metric: str, labels: Optional[Dict[str, str]] = None):
        """
        Increment counter metric.

        Args:
            metric: Metric name
            labels: Label dict (e.g., {"method": "GET", "endpoint": "/health"})
        """
        if metric not in self.metrics:
            self.metrics[metric] = {}

        if labels:
            label_key = self._make_label_key(labels)
            if label_key not in self.metrics[metric]:
                self.metrics[metric][label_key] = 0
            self.metrics[metric][label_key] += 1
        else:
            if not isinstance(self.metrics[metric], int):
                self.metrics[metric] = 0
            self.metrics[metric] += 1

    def record_duration(
        self,
        metric: str,
        duration_seconds: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Record duration sample for histogram.

        Args:
            metric: Metric name (e.g., "request_duration_samples")
            duration_seconds: Duration in seconds
            labels: Label dict
        """
        sample_key = f"{metric}_samples"
        if sample_key not in self.metrics:
            self.metrics[sample_key] = []

        # Keep last 1000 samples
        samples = self.metrics[sample_key]
        samples.append({
            "value": duration_seconds,
            "labels": labels or {},
            "timestamp": time.time()
        })

        if len(samples) > 1000:
            self.metrics[sample_key] = samples[-1000:]

    def set_gauge(self, metric: str, value: float):
        """
        Set gauge metric value.

        Args:
            metric: Metric name
            value: Current value
        """
        self.metrics[metric] = value

    def _make_label_key(self, labels: Dict[str, str]) -> str:
        """Create string key from labels dict."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Add metadata
        lines.append(f"# HELP mcp_uptime_seconds Service uptime in seconds")
        lines.append(f"# TYPE mcp_uptime_seconds gauge")
        lines.append(f"mcp_uptime_seconds {time.time() - self.start_time}")
        lines.append("")

        # Export counters
        for metric_name, value in self.metrics.items():
            if metric_name.endswith("_samples"):
                continue  # Skip histogram samples

            if isinstance(value, dict):
                # Counter with labels
                lines.append(f"# HELP mcp_{metric_name} Total count")
                lines.append(f"# TYPE mcp_{metric_name} counter")
                for label_key, count in value.items():
                    lines.append(f'mcp_{metric_name}{{{label_key}}} {count}')
                lines.append("")

            elif isinstance(value, (int, float)):
                # Simple counter or gauge
                metric_type = "gauge" if "active" in metric_name or "total_" in metric_name else "counter"
                lines.append(f"# HELP mcp_{metric_name} {metric_name.replace('_', ' ').title()}")
                lines.append(f"# TYPE mcp_{metric_name} {metric_type}")
                lines.append(f"mcp_{metric_name} {value}")
                lines.append("")

        # Export histograms (simplified - show percentiles)
        for metric_name in ["request_duration_samples", "db_query_duration_samples"]:
            if metric_name in self.metrics and self.metrics[metric_name]:
                samples = [s["value"] for s in self.metrics[metric_name]]
                samples.sort()

                base_name = metric_name.replace("_samples", "")
                lines.append(f"# HELP mcp_{base_name}_seconds Duration in seconds")
                lines.append(f"# TYPE mcp_{base_name}_seconds summary")

                # Calculate percentiles
                if samples:
                    p50 = samples[int(len(samples) * 0.5)]
                    p95 = samples[int(len(samples) * 0.95)]
                    p99 = samples[int(len(samples) * 0.99)]

                    lines.append(f'mcp_{base_name}_seconds{{quantile="0.5"}} {p50}')
                    lines.append(f'mcp_{base_name}_seconds{{quantile="0.95"}} {p95}')
                    lines.append(f'mcp_{base_name}_seconds{{quantile="0.99"}} {p99}')
                    lines.append(f'mcp_{base_name}_seconds_sum {sum(samples)}')
                    lines.append(f'mcp_{base_name}_seconds_count {len(samples)}')
                lines.append("")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """
        Export metrics as JSON.

        Returns:
            Metrics dict
        """
        return {
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS (for server.py integration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_cache() -> RedisCache:
    """Factory function to get RedisCache instance."""
    return RedisCache(
        redis_url=os.getenv("REDIS_URL"),
        default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "300")),
        namespace=os.getenv("CACHE_NAMESPACE", "mcp")
    )


def get_compressor() -> Compressor:
    """Factory function to get Compressor instance."""
    return Compressor(
        compression_level=int(os.getenv("COMPRESSION_LEVEL", "3")),
        threshold_bytes=int(os.getenv("COMPRESSION_THRESHOLD", "10240"))
    )


def get_metrics_collector() -> MetricsCollector:
    """Factory function to get MetricsCollector instance."""
    return MetricsCollector()


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

"""
Integration with server.py:

1. In server.py startup_event():

   from cache import get_cache, get_compressor, get_metrics_collector

   mcp_service.cache = get_cache()
   mcp_service.compressor = get_compressor()
   mcp_service.metrics_collector = get_metrics_collector()

2. Use cache in service methods:

   # Try cache first
   cached = await mcp_service.cache.get_session(session_id)
   if cached:
       return cached

   # Fetch from DB and cache
   session = await fetch_from_db(session_id)
   await mcp_service.cache.set_session(session_id, session)

3. Use compression for large content:

   if mcp_service.compressor.should_compress(content):
       content = mcp_service.compressor.compress(content)
       compressed = True

4. Track metrics:

   start = time.time()
   # ... process request ...
   duration = time.time() - start

   mcp_service.metrics_collector.record_duration(
       "request_duration",
       duration,
       {"endpoint": "/sessions", "method": "POST"}
   )

5. Export metrics endpoint:

   @app.get("/metrics/prometheus")
   async def metrics_prometheus():
       return Response(
           content=mcp_service.metrics_collector.export_prometheus(),
           media_type="text/plain"
       )
"""