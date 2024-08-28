import numpy as np
from collections import defaultdict, OrderedDict
from river import linear_model, preprocessing, metrics

class OnlineMLCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.freq = defaultdict(int)
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        self.metric = metrics.Accuracy()
        self.hits = 0
        self.requests = 0
        self.prev_key = None
        self.prev_freq = None

    def feature_dict(self, key):
        return {"key": key, "freq": self.freq[key]}

    def access(self, key):
        self.requests += 1
        self.freq[key] += 1
        feats = self.feature_dict(key)

        try:
            prob = self.model.predict_proba_one(feats)[1]
        except AttributeError:
            prob = self.model.predict_one(feats)

        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                scores = {}
                for k in self.cache:
                    f = {"key": k, "freq": self.freq[k]}
                    try:
                        scores[k] = self.model.predict_proba_one(f)[1]
                    except AttributeError:
                        scores[k] = self.model.predict_one(f)
                evict = min(scores, key=scores.get)
                self.cache.pop(evict)
            self.cache[key] = None

        if self.prev_key is not None:
            label = int(key == self.prev_key)
            prev_feats = {"key": self.prev_key, "freq": self.prev_freq}
            y_pred = self.model.predict_one(prev_feats)
            self.metric.update(label, y_pred)
            self.model.learn_one(prev_feats, label)

        self.prev_key = key
        self.prev_freq = self.freq[key]

        return prob

    def hit_rate(self):
        return self.hits / self.requests if self.requests else 0

    def accuracy(self):
        return self.metric.get()