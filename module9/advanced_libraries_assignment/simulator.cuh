// GPU side of the savings simulator: cuRAND draws every future's yearly
// returns, then Thrust grows the savings and summarizes the results.
#pragma once

#include <thrust/device_vector.h>
#include "common.h"
#include "history_data.h"

// Yearly returns for every simulated future, stored year by year: the
// return of future f in year y is at [y * futures + f]. Neighbouring GPU
// threads work on neighbouring futures, so they read neighbouring memory.
struct Returns {
    thrust::device_vector<float> stocks;
    thrust::device_vector<float> bonds;
    int futures = 0;
    int years = 0;
};

// cuRAND step: fills `out` with plan.futures x plan.years stock and bond
// returns, drawn from a bell curve or from real years (plan.model).
void generateReturns(const Plan& plan, const ReturnHistory& history,
                     Returns* out);

// Average of the first `count` values (used to check the random returns).
double averageOf(const thrust::device_vector<float>& values, size_t count);

// Thrust step: grows every future's savings with the given stock share,
// leaves the final balances in `balances` (sorted) and summarizes them.
MixResult simulateMix(const Plan& plan, const Returns& returns,
                      float stockShare,
                      thrust::device_vector<float>& balances);
