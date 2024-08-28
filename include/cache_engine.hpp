#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cache_engine
{

    using Key = std::int64_t;

    struct Stats
    {
        std::size_t hits = 0;
        std::size_t requests = 0;
        double hit_rate() const { return requests == 0 ? 0.0 : static_cast<double>(hits) / requests; }
    };

    inline void require_capacity(std::size_t capacity)
    {
        if (capacity == 0)
            throw std::invalid_argument("cache capacity must be greater than zero");
    }

    // O(1) lookup, promotion, and eviction.
    class LRUCache
    {
    public:
        explicit LRUCache(std::size_t capacity) : capacity_(capacity) { require_capacity(capacity); }
        bool access(Key key)
        {
            ++stats_.requests;
            auto pos = positions_.find(key);
            if (pos != positions_.end())
            {
                ++stats_.hits;
                order_.splice(order_.end(), order_, pos->second);
                return true;
            }
            if (order_.size() == capacity_)
            {
                positions_.erase(order_.front());
                order_.pop_front();
            }
            order_.push_back(key);
            positions_[key] = std::prev(order_.end());
            return false;
        }
        const Stats &stats() const { return stats_; }

    private:
        std::size_t capacity_;
        std::list<Key> order_;
        std::unordered_map<Key, std::list<Key>::iterator> positions_;
        Stats stats_;
    };

    struct FrequencyEntry
    {
        std::size_t frequency;
        std::size_t touched;
        Key key;
        bool operator<(const FrequencyEntry &rhs) const
        {
            if (frequency != rhs.frequency)
                return frequency < rhs.frequency;
            if (touched != rhs.touched)
                return touched < rhs.touched;
            return key < rhs.key;
        }
    };

    // LFU and MFU both update/evict in O(log capacity). Timestamp gives deterministic LRU tie-breaking.
    class FrequencyCache
    {
    public:
        explicit FrequencyCache(std::size_t capacity, bool evict_most_frequent)
            : capacity_(capacity), evict_most_frequent_(evict_most_frequent) { require_capacity(capacity); }
        bool access(Key key)
        {
            ++stats_.requests;
            ++clock_;
            auto current = entries_.find(key);
            if (current != entries_.end())
            {
                ++stats_.hits;
                ranking_.erase(current->second);
                current->second = {current->second.frequency + 1, clock_, key};
                ranking_.insert(current->second);
                return true;
            }
            if (entries_.size() == capacity_)
            {
                auto victim = evict_most_frequent_ ? std::prev(ranking_.end()) : ranking_.begin();
                entries_.erase(victim->key);
                ranking_.erase(victim);
            }
            FrequencyEntry entry{1, clock_, key};
            entries_.emplace(key, entry);
            ranking_.insert(entry);
            return false;
        }
        const Stats &stats() const { return stats_; }

    private:
        std::size_t capacity_;
        bool evict_most_frequent_;
        std::size_t clock_ = 0;
        std::unordered_map<Key, FrequencyEntry> entries_;
        std::set<FrequencyEntry> ranking_;
        Stats stats_;
    };

    // Native online logistic-regression cache.  It learns whether a request is reused
    // within a bounded future horizon, rather than learning the self-fulfilling fact that
    // an item happened to be in the cache.  Feedback is delayed until a re-access or expiry.
    class OnlineMLCache
    {
    public:
        explicit OnlineMLCache(std::size_t capacity, double learning_rate = 0.035, std::size_t reuse_horizon = 0)
            : capacity_(capacity), reuse_horizon_(reuse_horizon == 0 ? std::max<std::size_t>(32, capacity * 4) : reuse_horizon), learning_rate_(learning_rate)
        {
            require_capacity(capacity);
            if (learning_rate_ <= 0.0 || reuse_horizon_ == 0)
                throw std::invalid_argument("learning rate and reuse horizon must be positive");
        }
        bool access(Key key)
        {
            ++stats_.requests;
            ++clock_;
            expire_observations();
            ++frequencies_[key];
            const bool was_cached = cache_.count(key) != 0;
            const auto features = make_features(key);
            const double prediction = probability(features);

            // A re-access confirms that the most recent request for this key was useful.
            auto pending = latest_observation_.find(key);
            if (pending != latest_observation_.end()) {
                learn(pending->second->features, 1.0);
                pending->second->active = false;
                latest_observation_.erase(pending);
            }
            last_seen_[key] = clock_;

            if (was_cached)
            {
                ++stats_.hits;
                touch(key);
            }
            else
            {
                if (cache_.size() == capacity_)
                    evict_lowest_score();
                CacheEntry entry{frequencies_[key], clock_, 0.0};
                entry.score = prediction;
                cache_.emplace(key, entry);
            }
            observations_.push_back({key, features, clock_ + reuse_horizon_, true});
            latest_observation_[key] = std::prev(observations_.end());
            return was_cached;
        }
        const Stats &stats() const { return stats_; }
        double accuracy() const
        {
            return training_examples_ == 0 ? 0.0 : static_cast<double>(correct_predictions_) / training_examples_;
        }

    private:
        struct CacheEntry
        {
            std::size_t frequency;
            std::size_t touched;
            double score;
        };
        struct Observation
        {
            Key key;
            std::array<double, 3> features;
            std::size_t deadline;
            bool active;
        };
        std::array<double, 3> make_features(Key key) const
        {
            const auto frequency_it = frequencies_.find(key);
            const double frequency = frequency_it == frequencies_.end() ? 0.0 : static_cast<double>(frequency_it->second);
            const auto seen = last_seen_.find(key);
            const double age = seen == last_seen_.end() ? static_cast<double>(reuse_horizon_) : static_cast<double>(clock_ - seen->second);
            return {1.0, std::log1p(frequency), 1.0 / std::sqrt(1.0 + age)};
        }
        double probability(const std::array<double, 3> &x) const
        {
            double z = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i)
                z += weights_[i] * x[i];
            z = std::clamp(z, -35.0, 35.0);
            return 1.0 / (1.0 + std::exp(-z));
        }
        void learn(const std::array<double, 3> &x, double label)
        {
            const double prediction = probability(x);
            const double error = label - prediction;
            for (std::size_t i = 0; i < x.size(); ++i)
                weights_[i] = std::clamp(weights_[i] + learning_rate_ * (error * x[i] - 0.0005 * weights_[i]), -8.0, 8.0);
            ++training_examples_;
            correct_predictions_ += ((prediction >= 0.5) == (label >= 0.5));
        }
        void touch(Key key)
        {
            auto &entry = cache_.at(key);
            entry.frequency = frequencies_[key];
            entry.touched = clock_;
            entry.score = probability(make_features(key));
        }
        void expire_observations()
        {
            while (!observations_.empty() && observations_.front().deadline < clock_) {
                auto observation = observations_.begin();
                if (observation->active) {
                    learn(observation->features, 0.0);
                    auto latest = latest_observation_.find(observation->key);
                    if (latest != latest_observation_.end() && latest->second == observation)
                        latest_observation_.erase(latest);
                }
                observations_.pop_front();
            }
        }
        void evict_lowest_score()
        {
            for (auto &[key, entry] : cache_)
                entry.score = probability(make_features(key));
            auto victim = std::min_element(cache_.begin(), cache_.end(), [](const auto &a, const auto &b)
                                           {
            if (a.second.score != b.second.score) return a.second.score < b.second.score;
            return a.second.touched < b.second.touched; });
            cache_.erase(victim);
        }
        std::size_t capacity_, clock_ = 0, training_examples_ = 0, correct_predictions_ = 0, reuse_horizon_;
        double learning_rate_;
        std::array<double, 3> weights_{};
        std::unordered_map<Key, std::size_t> frequencies_, last_seen_;
        std::unordered_map<Key, CacheEntry> cache_;
        std::list<Observation> observations_;
        std::unordered_map<Key, std::list<Observation>::iterator> latest_observation_;
        Stats stats_;
    };

    inline std::vector<Key> generate_zipf_requests(std::size_t count, std::size_t unique_items, double skew, std::uint64_t seed)
    {
        if (unique_items == 0 || skew <= 0.0)
            throw std::invalid_argument("unique_items and skew must be positive");
        std::vector<double> weights(unique_items);
        for (std::size_t i = 0; i < unique_items; ++i)
            weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), skew);
        std::mt19937_64 rng(seed);
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        std::vector<Key> requests;
        requests.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            requests.push_back(static_cast<Key>(distribution(rng)));
        return requests;
    }

    inline std::vector<int> belady_labels(const std::vector<Key> &requests, std::size_t capacity)
    {
        require_capacity(capacity);
        std::unordered_map<Key, std::deque<std::size_t>> future;
        for (std::size_t i = 0; i < requests.size(); ++i)
            future[requests[i]].push_back(i);
        std::unordered_set<Key> cache;
        std::vector<int> labels;
        labels.reserve(requests.size());
        for (std::size_t i = 0; i < requests.size(); ++i)
        {
            const Key key = requests[i];
            future[key].pop_front();
            if (cache.count(key))
            {
                labels.push_back(1);
                continue;
            }
            labels.push_back(0);
            if (cache.size() == capacity)
            {
                Key victim{};
                std::size_t farthest = 0;
                bool chose_never = false;
                for (Key candidate : cache)
                {
                    const auto &positions = future[candidate];
                    if (positions.empty())
                    {
                        victim = candidate;
                        chose_never = true;
                        break;
                    }
                    if (positions.front() >= farthest)
                    {
                        farthest = positions.front();
                        victim = candidate;
                    }
                }
                (void)chose_never;
                cache.erase(victim);
            }
            cache.insert(key);
        }
        return labels;
    }

} // namespace cache_engine
