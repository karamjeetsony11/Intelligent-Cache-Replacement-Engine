#include "cache_engine.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace cache_engine;

namespace
{
    struct Options
    {
        std::size_t requests = 100000, items = 1000, capacity = 100;
        double skew = 1.2, learning_rate = 0.035;
        std::uint64_t seed = 42;
        std::size_t reuse_horizon = 0;
        std::string csv_output;
    };

    std::size_t parse_size(const char *value, const char *name)
    {
        try
        {
            return static_cast<std::size_t>(std::stoull(value));
        }
        catch (...)
        {
            throw std::invalid_argument(std::string("invalid ") + name);
        }
    }
    double parse_double(const char *value, const char *name)
    {
        try
        {
            return std::stod(value);
        }
        catch (...)
        {
            throw std::invalid_argument(std::string("invalid ") + name);
        }
    }
    void usage(const char *program)
    {
        std::cout << "Usage: " << program << " <benchmark|generate|self-test> [options]\n"
                  << "  --requests N --items N --capacity N --skew X --seed N\n"
                  << "  benchmark options: --learning-rate X --reuse-horizon N\n"
                  << "  generate additionally requires --output FILE\n";
    }
    Options options_from(int argc, char **argv, int start)
    {
        Options options;
        for (int i = start; i < argc; i += 2)
        {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + std::string(argv[i]));
            std::string flag = argv[i], value = argv[i + 1];
            if (flag == "--requests")
                options.requests = parse_size(value.c_str(), "request count");
            else if (flag == "--items")
                options.items = parse_size(value.c_str(), "item count");
            else if (flag == "--capacity")
                options.capacity = parse_size(value.c_str(), "capacity");
            else if (flag == "--seed")
                options.seed = parse_size(value.c_str(), "seed");
            else if (flag == "--skew")
                options.skew = parse_double(value.c_str(), "skew");
            else if (flag == "--learning-rate")
                options.learning_rate = parse_double(value.c_str(), "learning rate");
            else if (flag == "--reuse-horizon")
                options.reuse_horizon = parse_size(value.c_str(), "reuse horizon");
            else if (flag == "--output")
                options.csv_output = value;
            else
                throw std::invalid_argument("unknown option: " + flag);
        }
        require_capacity(options.capacity);
        if (options.items == 0 || options.skew <= 0.0 || options.learning_rate <= 0.0)
            throw std::invalid_argument("items, skew, and learning rate must be positive");
        return options;
    }
    void run_benchmark(const Options &options)
    {
        const auto requests = generate_zipf_requests(options.requests, options.items, options.skew, options.seed);
        LRUCache lru(options.capacity);
        FrequencyCache lfu(options.capacity, false), mfu(options.capacity, true);
        OnlineMLCache ml(options.capacity, options.learning_rate, options.reuse_horizon);
        for (Key key : requests)
        {
            lru.access(key);
            lfu.access(key);
            mfu.access(key);
            ml.access(key);
        }
        std::cout << std::fixed << std::setprecision(4)
                  << "=== Native C++ Cache Benchmark ===\n"
              << "Requests: " << options.requests << ", items: " << options.items << ", capacity: " << options.capacity << "\n"
              << "Learning rate: " << options.learning_rate << ", reuse horizon: "
              << (options.reuse_horizon == 0 ? std::max<std::size_t>(32, options.capacity * 4) : options.reuse_horizon) << "\n"
                  << "LRU       : " << lru.stats().hit_rate() << "\n"
                  << "LFU       : " << lfu.stats().hit_rate() << "\n"
                  << "MFU       : " << mfu.stats().hit_rate() << "\n"
                  << "Online ML : " << ml.stats().hit_rate() << "\n"
              << "ML reuse-label accuracy: " << ml.accuracy() << "\n";
    }
    void generate_csv(const Options &options)
    {
        if (options.csv_output.empty())
            throw std::invalid_argument("generate requires --output FILE");
        const auto requests = generate_zipf_requests(options.requests, options.items, options.skew, options.seed);
        const auto labels = belady_labels(requests, options.capacity);
        std::ofstream file(options.csv_output);
        if (!file)
            throw std::runtime_error("cannot open output file: " + options.csv_output);
        file << "number,is_cached\n";
        for (std::size_t i = 0; i < requests.size(); ++i)
            file << requests[i] << ',' << labels[i] << '\n';
        std::cout << "Saved " << requests.size() << " Belady-labeled requests to " << options.csv_output << '\n';
    }
    void self_test()
    {
        LRUCache lru(2);
        for (Key key : {1, 2, 1, 3, 1})
            lru.access(key);
        if (lru.stats().hits != 2)
            throw std::runtime_error("LRU self-test failed");
        const std::vector<Key> trace{1, 2, 3, 1, 2, 3};
        const auto labels = belady_labels(trace, 2);
        if (labels.size() != trace.size() || labels[3] != 1)
            throw std::runtime_error("Belady self-test failed");
        OnlineMLCache ml(2);
        for (Key key : trace)
            ml.access(key);
        std::cout << "Self-test passed.\n";
    }
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc < 2)
        {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        const std::string command = argv[1];
        if (command == "self-test")
        {
            if (argc != 2)
                throw std::invalid_argument("self-test takes no options");
            self_test();
        }
        else
        {
            const Options options = options_from(argc, argv, 2);
            if (command == "benchmark")
                run_benchmark(options);
            else if (command == "generate")
                generate_csv(options);
            else
            {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
