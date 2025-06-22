# Smart Caching System for AI Memory Infrastructure

A high-performance, intelligent caching layer for vector search queries with LFU/LRU eviction strategy and semantic similarity matching.

## 🚀 Features

- **Two-Level Caching**:
  - L1: Exact query match (hash-based)
  - L2: Semantic similarity match (vector-based)
- **Hybrid Eviction Strategy**: Combines LFU (Least Frequently Used) and LRU (Least Recently Used)
- **Adaptive TTL**: Dynamic expiration based on query popularity and recency
- **Cache Invalidation**: Automatic invalidation when vectors are updated
- **Performance Monitoring**: Real-time metrics and statistics
- **RESTful API**: Easy integration with existing systems

## 📋 Architecture

```
Client Request → Vector Search API → L1 Cache Check → L2 Semantic Check → Qdrant Query
                                           ↓                ↓                    ↓
                                     Redis Cache      Qdrant Meta         Main Collection
```

## 🔧 Components

1. **Cache Manager** (`cache_manager.py`): Core caching logic
2. **FastAPI Application** (`main.py`): REST API endpoints
3. **Monitoring** (`monitoring.py`): Metrics collection and tracking
4. **Configuration** (`config.py`): Centralized configuration

## 📊 Performance Targets

- Cache Hit Rate: >95% for popular queries
- Search Latency: <50ms for cached queries
- Memory Efficiency: Adaptive caching with configurable limits
- Scalability: Support for 10,000+ cached queries

## 🚀 Deployment

### Prerequisites
- Python 3.10+
- Redis 7.0+
- Qdrant 1.7+
- Access to embedding service

### Quick Start

1. **Clone and Navigate**:
   ```bash
   cd /opt/ai-memory/smart_caching
   ```

2. **Install Dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Edit `config.py` or set environment variables:
   ```bash
   export REDIS_HOST=10.10.180.170
   export QDRANT_HOST=10.10.180.172
   export API_PORT=8006
   ```

4. **Run the Service**:
   ```bash
   python main.py
   ```

### Production Deployment

Use the provided deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

This will:
- Copy files to the server
- Install dependencies
- Create systemd service
- Set up log rotation
- Start the service

## 📡 API Endpoints

### Core Endpoints

- `POST /search` - Perform cached vector search
- `POST /invalidate` - Invalidate cached queries for updated vectors
- `GET /stats` - Get cache statistics
- `GET /health` - Health check for all services

### Management Endpoints

- `POST /cache/clear` - Clear all cached entries
- `POST /cache/warmup` - Pre-populate cache with queries
- `GET /cache/top-queries` - Get most frequently cached queries

### API Documentation

Access interactive API docs at: `http://10.10.180.173:8006/docs`

## 📈 Monitoring

### Built-in Monitoring

The system includes comprehensive monitoring:
- Real-time metrics collection
- Performance tracking
- Alert generation
- Time-series data storage

### Monitoring Commands

```bash
# Check service status
ssh user@10.10.180.173 '/opt/ai-memory/smart_caching/monitor_smart_cache.sh'

# View logs
journalctl -u ai-memory-smart-cache -f

# Get cache statistics
curl http://10.10.180.173:8006/stats | jq
```

## 🔧 Configuration

Key configuration parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CACHE_SIZE` | 10000 | Maximum number of cached queries |
| `SEMANTIC_SIMILARITY_THRESHOLD` | 0.98 | Threshold for L2 cache matches |
| `BASE_TTL` | 3600 | Base TTL in seconds (1 hour) |
| `LFU_WEIGHT` | 0.7 | Weight for frequency in eviction |
| `LRU_WEIGHT` | 0.3 | Weight for recency in eviction |

## 🧪 Testing

Run the test suite:
```bash
python test_client.py
```

This includes:
- Health checks
- Cache hit/miss scenarios
- Invalidation testing
- Performance benchmarks
- Stress testing

## 📊 Performance Tuning

### Redis Optimization
- Use Redis persistence for cache durability
- Configure appropriate memory limits
- Enable Redis clustering for high scale

### Qdrant Optimization
- Index the meta collection for fast semantic search
- Tune HNSW parameters for the meta collection
- Use appropriate vector dimensions

### API Optimization
- Adjust worker count in uvicorn
- Enable response compression
- Use connection pooling

## 🐛 Troubleshooting

### Common Issues

1. **Low Hit Rate**:
   - Check semantic similarity threshold
   - Verify embedding consistency
   - Review eviction settings

2. **High Latency**:
   - Check Redis connection
   - Monitor Qdrant performance
   - Review cache size limits

3. **Memory Issues**:
   - Adjust MAX_CACHE_SIZE
   - Check Redis memory usage
   - Review eviction batch size

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 🔒 Security

- API key authentication (configure in production)
- Input validation on all endpoints
- Rate limiting support
- Secure Redis connection with password

## 📝 Integration Example

```python
import httpx

# Search with caching
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://10.10.180.173:8006/search",
        json={
            "query_text": "What is machine learning?",
            "top_k": 5
        }
    )
    result = response.json()
    print(f"Source: {result['source']}")  # L1_cache, L2_cache, or qdrant_db
    print(f"Results: {result['results']}")
```

## 🚀 Advanced Features

### Cache Warmup
Pre-populate cache with popular queries:
```bash
curl -X POST http://10.10.180.173:8006/cache/warmup \
  -H "Content-Type: application/json" \
  -d '{"queries": ["machine learning", "AI", "vector database"]}'
```

### Batch Invalidation
Invalidate multiple vectors at once:
```bash
curl -X POST http://10.10.180.173:8006/invalidate \
  -H "Content-Type: application/json" \
  -d '{"vector_ids": ["vec1", "vec2", "vec3"]}'
```

## 📈 Metrics and KPIs

The system tracks:
- Cache hit rate (L1 and L2 separately)
- Average response time
- Cache utilization
- Query frequency distribution
- Eviction statistics

## 🔄 Future Enhancements

- [ ] Distributed caching with Redis Cluster
- [ ] Machine learning-based TTL prediction
- [ ] Query result compression
- [ ] Multi-tenant support
- [ ] GraphQL API support

## 📄 License

Part of the AI Memory Infrastructure project.

---

**Version**: 1.0.0  
**Last Updated**: June 2025  
**Maintainer**: AI Memory Infrastructure Team
