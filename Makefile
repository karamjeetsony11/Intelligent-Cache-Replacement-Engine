CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude

.PHONY: all clean test

all: cache_engine

cache_engine: src/main.cpp include/cache_engine.hpp
	$(CXX) $(CXXFLAGS) src/main.cpp -o $@

test: cache_engine
	./cache_engine self-test

clean:
	rm -f cache_engine
