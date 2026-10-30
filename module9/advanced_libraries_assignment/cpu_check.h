// CPU versions of the simulation, used to check the GPU and to time it.
#pragma once

#include <vector>
#include "common.h"

// Same steps as simulateMix (grow, then summarize) on the CPU, using the
// GPU's random returns copied to the host (same year-by-year layout).
MixResult cpuSimulateMix(const Plan& plan, const std::vector<float>& stocks,
                         const std::vector<float>& bonds, float stockShare);

// Exact average final balance when every year's return has the given
// average: each year, average balance = (balance + saving) * (1 + avg).
double expectedBalance(const Plan& plan, double averageReturn);
