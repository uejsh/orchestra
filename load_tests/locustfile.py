# locustfile.py - Load Testing for Orchestra
# ============================================================================
# Run with: locust -f load_tests/locustfile.py --users 1000 --spawn-rate 100
# ============================================================================

from locust import HttpUser, task, between
import random

class OrchestraLoadTest(HttpUser):
    """
    Load test for Orchestra caching performance.
    Simulates realistic AI agent workloads.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    # Sample queries for semantic cache testing
    QUERIES = [
        "Analyze the quarterly sales report",
        "Show me Q4 revenue trends",
        "What are the sales figures for last quarter",
        "Generate a summary of customer feedback",
        "Summarize user reviews from the past month",
        "Create a report on recent customer sentiment",
        "Predict next month's inventory needs",
        "What inventory should we order for March",
        "Forecast product demand for coming month",
    ]
    
    @task(weight=7)
    def cache_lookup(self):
        """
        Simulate cache lookups.
        70% of requests should be semantically similar (cache hits).
        """
        query = random.choice(self.QUERIES)
        self.client.post("/cache/get", json={"query": query})
    
    @task(weight=3)
    def cache_store(self):
        """
        Simulate storing new results (30% of traffic).
        """
        query = f"Unique query {random.randint(1, 10000)}"
        self.client.post("/cache/put", json={
            "query": query,
            "value": {"result": "cached data"}
        })
    
    @task(weight=1)
    def health_check(self):
        """Health check requests."""
        self.client.get("/health")
    
    @task(weight=1)
    def metrics(self):
        """Metrics scraping."""
        self.client.get("/metrics")

class StressTest(HttpUser):
    """
    Stress test - rapid fire requests to test limits.
    """
    
    wait_time = between(0.1, 0.5)  # More aggressive
    
    @task
    def rapid_cache_requests(self):
        query = random.choice(OrchestraLoadTest.QUERIES)
        self.client.post("/cache/get", json={"query": query})
