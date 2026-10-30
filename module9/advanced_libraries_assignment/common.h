// Shared settings, types and error-checking helpers for the savings goal
// simulator (CUDA Advanced Libraries assignment).
#pragma once

#include <cstdio>
#include <cstdlib>

// ---- Error checking -------------------------------------------------------

// Stops the program with the file and line of any failed CUDA or cuRAND
// call. (Thrust reports its errors by throwing an exception.)
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CURAND_CHECK(call)                                                \
    do {                                                                  \
        curandStatus_t status__ = (call);                                 \
        if (status__ != CURAND_STATUS_SUCCESS) {                          \
            fprintf(stderr, "cuRAND error %s:%d: status %d\n", __FILE__,  \
                    __LINE__, static_cast<int>(status__));                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// ---- Input data -----------------------------------------------------------

// Yearly S&P 500 and 10-year US Treasury bond returns, 1928-2025.
constexpr const char* RETURNS_FILE = "data/annual_returns.csv";
constexpr int MIN_HISTORY_YEARS = 10;

// ---- Default savings plan (each can be changed on the command line) -------

constexpr double DEFAULT_MONTHLY = 500.0;      // saved every month
constexpr double DEFAULT_START = 0.0;          // balance on day one
constexpr double DEFAULT_GOAL = 1000000.0;
constexpr int DEFAULT_YEARS = 30;
constexpr int DEFAULT_FUTURES = 1000000;       // simulated futures
constexpr unsigned long long DEFAULT_SEED = 2026;
constexpr int MONTHS_PER_YEAR = 12;
constexpr int MIN_FUTURES = 1000;

// Limits that keep the random returns within GPU memory: two floats per
// future per year, so MAX_DRAWS uses about 1 GB.
constexpr int MAX_YEARS = 70;
constexpr long long MAX_DRAWS = 120000000;

// ---- Simulation -----------------------------------------------------------

// Stock/bond mixes compared in every run: 100% stocks down to 0% stocks
// (the rest in bonds), rebalanced back to the mix every year.
constexpr int NUM_MIXES = 6;
constexpr float STOCK_SHARES[NUM_MIXES] = {1.0f, 0.8f, 0.6f,
                                           0.4f, 0.2f, 0.0f};

// A bell curve can, very rarely, draw a yearly loss of more than 100%.
// No real fund can lose more than everything, so such a year is treated
// as losing 99%.
constexpr float WORST_YEAR_RETURN = -0.99f;

// "Bad case" and "good case" futures: 1 in 10 ends below BAD and 1 in 10
// ends above GOOD.
constexpr double BAD_PERCENTILE = 0.10;
constexpr double MEDIAN_PERCENTILE = 0.50;
constexpr double GOOD_PERCENTILE = 0.90;

// ---- Checks ---------------------------------------------------------------

// A Monte Carlo average lands within 4 standard errors of the true value
// about 99.99% of the time.
constexpr double MAX_STANDARD_ERRORS = 4.0;
// GPU and CPU do the same float arithmetic, so their answers should agree
// to within float rounding.
constexpr double GPU_CPU_TOLERANCE = 1e-4;

// The GPU mix step is short enough that one measurement is easily thrown
// off by other work on the machine, so its time is the average of this
// many repeats.
constexpr int TIMED_REPEATS = 5;

// ---- Types ----------------------------------------------------------------

// Where each simulated year's returns come from.
enum class Model {
    Normal,   // bell curve with the historical average and swing
    History   // a randomly chosen real year from 1928-2025
};

struct Plan {
    double monthly;
    double start;
    double goal;
    int years;
    int futures;
    Model model;
    unsigned long long seed;
};

// Summary of the final balances of one stock/bond mix.
struct MixResult {
    float stockShare;   // 0.6 = 60% stocks, 40% bonds
    double chance;      // fraction of futures that reach the goal
    double mean;        // average final balance
    double stddev;      // spread of the final balances
    double bad;         // BAD_PERCENTILE of the final balances
    double median;
    double good;        // GOOD_PERCENTILE
};

// Index of percentile p (0..1) in a sorted list of n values.
inline int percentileIndex(double p, int n)
{
    return static_cast<int>(p * (n - 1));
}
