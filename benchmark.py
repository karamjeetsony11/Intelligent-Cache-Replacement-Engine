import numpy as np
import joblib
from data_generator import generate_skewed_data
from collections import defaultdict, OrderedDict, Counter


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.requests = 0

    def access(self, key):
        self.requests += 1
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = None

    def hit_rate(self):
        return self.hits / self.requests if self.requests > 0 else 0


class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = set()
        self.freq = Counter()
        self.hits = 0
        self.requests = 0

    def access(self, key):
        self.requests += 1
        if key in self.cache:
            self.hits += 1
        else:
            if len(self.cache) >= self.capacity:
                evict = min(self.cache, key=lambda x: self.freq[x])
                self.cache.remove(evict)
            self.cache.add(key)
        self.freq[key] += 1

    def hit_rate(self):
        return self.hits / self.requests if self.requests > 0 else 0


class MFUCache(LFUCache):
    def access(self, key):
        self.requests += 1
        if key in self.cache:
            self.hits += 1
        else:
            if len(self.cache) >= self.capacity:
                evict = max(self.cache, key=lambda x: self.freq[x])
                self.cache.remove(evict)
            self.cache.add(key)
        self.freq[key] += 1


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

    def hit_rate(self):
        return self.hits / self.requests if self.requests > 0 else 0


if __name__ == "__main__":
    NUM_REQUESTS = 10000
    UNIQUE_ITEMS = 100
    CACHE_CAPACITY = 10
    SKEW = 1.2

    print("Generating test request stream...")
    requests = generate_skewed_data(NUM_REQUESTS, UNIQUE_ITEMS, SKEW)

    print("Loading best model...")
    model = joblib.load("models/best_model.pkl")

    print("Simulating caches...")
    lru = LRUCache(CACHE_CAPACITY)
    lfu = LFUCache(CACHE_CAPACITY)
    mfu = MFUCache(CACHE_CAPACITY)
    ml = MLCache(CACHE_CAPACITY, model)

    for r in requests:
        lru.access(r)
        lfu.access(r)
        mfu.access(r)
        ml.access(r)

    print("\n=== Cache Hit Rates ===")
    print(f"LRU : {lru.hit_rate():.4f}")
    print(f"LFU : {lfu.hit_rate():.4f}")
    print(f"MFU : {mfu.hit_rate():.4f}")
    print(f"ML  : {ml.hit_rate():.4f}")
