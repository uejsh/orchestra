# Load Testing README

## Setup

Install Locust:
```bash
pip install locust
```

## Basic Load Test

Run with 1000 concurrent users:
```bash
locust -f load_tests/locustfile.py --users 1000 --spawn-rate 100 --host http://localhost:8000
```

Open web UI at: http://localhost:8089

## Scenarios

### 1. Normal Load (OrchestraLoadTest)
Simulates realistic production traffic:
- 70% cache lookups (semantic duplicates)
- 30% new data storage
- Health checks and metrics scraping

### 2. Stress Test (StressTest)
Aggressive rapid-fire requests to find breaking points.

## Metrics to Watch

- **Response Time**: Should be <100ms for cache hits
- **Error Rate**: Should be <1%
- **RPS**: Target at least 500 req/s per instance

## CI/CD Integration

```yaml
# .github/workflows/load-test.yml
- name: Run Load Test
  run: |
    docker-compose up -d
    sleep 10
    locust -f load_tests/locustfile.py \
      --headless \
      --users 100 \
      --spawn-rate 10 \
      --run-time 2m \
      --host http://localhost:8000
```
