from data_generator import generate_skewed_data
from online_cache import OnlineMLCache

if __name__ == "__main__":
    NUM_REQUESTS = 10000
    UNIQUE_ITEMS = 1000
    CACHE_CAPACITY = 100
    SKEW = 1.2
    stream = generate_skewed_data(NUM_REQUESTS, UNIQUE_ITEMS, SKEW)
    cache = OnlineMLCache(CACHE_CAPACITY)

    for key in stream:
        cache.access(key)

    print(f"Streaming ML-Cache Hit Rate: {cache.hit_rate():.4f}")
    print(f"Prediction Accuracy:       {cache.accuracy():.4f}")
