import numpy as np
import argparse
import os
import csv
from collections import defaultdict


def generate_skewed_data(num_requests: int, unique_items: int, skew: float = 1.2):
    raw = np.random.zipf(a=skew, size=num_requests)
    items = (raw % unique_items)
    return items.tolist()


def belady_optimal_labels(requests, capacity):
    n = len(requests)
    future_indices = defaultdict(list)

    for i in reversed(range(n)):
        future_indices[requests[i]].append(i)

    labels = []
    cache = set()
    future_positions = {k: list(reversed(v)) for k, v in future_indices.items()}

    for i in range(n):
        current = requests[i]

        if future_positions[current]:
            future_positions[current].pop(0)

        if current in cache:
            labels.append(1)
        else:
            labels.append(0)
            if len(cache) >= capacity:
                farthest, evict_candidate = -1, None
                for item in cache:
                    if not future_positions[item]:
                        evict_candidate = item
                        break
                    elif future_positions[item][0] > farthest:
                        farthest = future_positions[item][0]
                        evict_candidate = item
                cache.remove(evict_candidate)
            cache.add(current)

    return labels


def save_data_csv(requests, labels, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["number", "is_cached"])
        for r, l in zip(requests, labels):
            writer.writerow([r, l])
    print(f"Saved {len(requests)} labeled requests to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate skewed data with Belady-optimal labels")
    parser.add_argument("--num_requests", type=int, default=100000)
    parser.add_argument("--unique_items", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=10)
    parser.add_argument("--skew", type=float, default=1.2)
    parser.add_argument("--output", type=str, default="data/labeled_requests.csv")
    args = parser.parse_args()

    data = generate_skewed_data(args.num_requests, args.unique_items, args.skew)
    labels = belady_optimal_labels(data, args.capacity)
    save_data_csv(data, labels, args.output)
