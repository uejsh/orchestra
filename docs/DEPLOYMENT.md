# Production Deployment Guide

This guide covers deploying Orchestra in production environments.

## Prerequisites

- Docker or Kubernetes cluster
- Redis Stack (for distributed caching)
- PostgreSQL 13+ (optional, for distributed trace storage)

## Docker Deployment

### Build the Image

```bash
cd docker
docker build -t orchestra:latest -f Dockerfile ..
```

### Run with Docker Compose

```bash
docker-compose up -d
```

This starts:
- Orchestra service
- Redis Stack (with RediSearch)
- PostgreSQL

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis Stack connection string | `redis://localhost:6379` |
| `POSTGRES_DSN` | PostgreSQL DSN for traces | None (uses SQLite) |
| `CACHE_TTL` | Default TTL for cache entries (seconds) | `3600` |
| `SIMILARITY_THRESHOLD` | Semantic matching threshold | `0.92` |

## Kubernetes Deployment

### 1. Create Secrets

```bash
kubectl create secret generic orchestra-secrets \
  --from-literal=redis-url='redis://redis-service:6379' \
  --from-literal=postgres-dsn='postgresql://user:pass@postgres:5432/orchestra'
```

### 2. Deploy Application

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 3. Verify Health

```bash
kubectl get pods -l app=orchestra
kubectl port-forward svc/orchestra 8000:80
curl http://localhost:8000/health
```

## Monitoring

### Prometheus Metrics

Orchestra exposes Prometheus-compatible metrics at `/metrics`:

```yaml
# Add to your Prometheus scrape config
scrape_configs:
  - job_name: 'orchestra'
    static_configs:
      - targets: ['orchestra-metrics:9090']
```

**Key Metrics:**
- `orchestra_cache_hits_total` - Cache hit count
- `orchestra_cache_misses_total` - Cache miss count
- `orchestra_latency_seconds` - Request latency histogram
- `orchestra_errors_total` - Error count by type

### Health Checks

Health endpoint: `GET /health`

Returns:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "checks": {
    "redis": {"status": "ok"},
    "postgres": {"status": "ok"}
  }
}
```

## Scaling

### Horizontal Scaling

Orchestra is stateless and can be scaled horizontally:

```bash
kubectl scale deployment orchestra --replicas=10
```

### Resource Recommendations

| Workload | CPU | Memory | Replicas |
|----------|-----|--------|----------|
| Light (<100 req/s) | 500m | 512Mi | 2-3 |
| Medium (100-1000 req/s) | 1000m | 1Gi | 5-10 |
| Heavy (>1000 req/s) | 2000m | 2Gi | 10+ |

## Redis Configuration

For production, use **Redis Stack** (not standard Redis):

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  redis/redis-stack:latest
```

**Why Redis Stack?**
- Native vector search (RediSearch)
- Better performance for semantic caching
- JSONB support for complex state

## Database Migration

### SQLite to PostgreSQL

```python
from orchestra.recorder.storage import SQLiteStorage, PostgresStorage

# Export from SQLite
sqlite = SQLiteStorage("./orchestra/traces.db")
traces = sqlite.list_traces(limit=10000)

# Import to Postgres
postgres = PostgresStorage("postgresql://user:pass@localhost/orchestra")
for trace in traces:
    # ... migration logic
```

## Troubleshooting

### High Memory Usage
- Enable compression: `OrchestraConfig(enable_compression=True)`
- Reduce `max_cache_size`
- Check for memory leaks in custom nodes

### Redis Connection Errors
- Verify Redis Stack is running (not standard Redis)
- Check network connectivity
- Review circuit breaker logs

### Slow Performance
- Check `similarity_threshold` (lower = more cache hits, less accuracy)
- Enable hierarchical embeddings for complex queries
- Review Prometheus latency metrics

## Security

### Enable TLS

```yaml
# Redis
redis_url: rediss://redis:6380

# PostgreSQL
postgres_dsn: postgresql://user:pass@host:5432/db?sslmode=require
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: orchestra-policy
spec:
  podSelector:
    matchLabels:
      app: orchestra
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
```

## Next Steps

- Set up log aggregation (ELK, Loki)
- Configure distributed tracing (Jaeger, Zipkin)
- Implement auto-scaling with HPA
