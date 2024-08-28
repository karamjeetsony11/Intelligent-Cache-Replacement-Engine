import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
from data_generator import generate_skewed_data


class MLCache:
    def __init__(self, capacity, model):
        self.capacity = capacity
        self.cache = set()
        self.freq = defaultdict(int)
        self.model = model
        self.hits = 0
        self.requests = 0

    def access(self, key):
        self.requests += 1
        self.freq[key] += 1
        features = np.array([[key, self.freq[key]]])

        if key in self.cache:
            self.hits += 1
        else:
            if len(self.cache) >= self.capacity:
                probs = {
                    k: self.model.predict_proba(np.array([[k, self.freq[k]]]))[0][1]
                    for k in self.cache
                }
                evict = min(probs, key=probs.get)
                self.cache.remove(evict)
            self.cache.add(key)

    def get_hit_rate(self):
        return self.hits / self.requests if self.requests > 0 else 0.0


if __name__ == "__main__":
    NUM_REQUESTS = 10000
    UNIQUE_ITEMS = 100
    CACHE_CAPACITY = 10
    SKEW = 1.3

    print("Generating test request stream...")
    test_data = generate_skewed_data(NUM_REQUESTS, UNIQUE_ITEMS, SKEW)

    print("Loading trained best model...")
    model = joblib.load("models/best_model.pkl")

    print("Running smart cache simulation...")
    cache = MLCache(CACHE_CAPACITY, model)
    for item in test_data:
        cache.access(item)

    hit_rate = cache.get_hit_rate()
    print(f"Smart cache hit rate: {hit_rate:.4f}")
