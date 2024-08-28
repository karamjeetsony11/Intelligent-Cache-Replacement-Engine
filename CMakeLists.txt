cmake_minimum_required(VERSION 3.16)
project(intelligent_cache_replacement_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cache_engine src/main.cpp)
target_include_directories(cache_engine PRIVATE include)
target_compile_options(cache_engine PRIVATE -Wall -Wextra -Wpedantic -O3)
