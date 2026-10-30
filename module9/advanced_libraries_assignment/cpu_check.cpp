// Plain C++ versions of the GPU steps, written independently of the GPU
// code so the two can be compared.

#include "cpu_check.h"

#include <algorithm>
#include <cmath>

// Final balance of one future (same arithmetic as GrowSavings).
static float growOne(const Plan& plan, const std::vector<float>& stocks,
                     const std::vector<float>& bonds, float stockShare,
                     int future)
{
    const float yearly = static_cast<float>(plan.monthly * MONTHS_PER_YEAR);
    float balance = static_cast<float>(plan.start);
    for (int y = 0; y < plan.years; ++y) {
        size_t i = static_cast<size_t>(y) * plan.futures + future;
        float r = stockShare * stocks[i] + (1.0f - stockShare) * bonds[i];
        r = std::max(r, WORST_YEAR_RETURN);
        balance = (balance + yearly) * (1.0f + r);
    }
    return balance;
}

static MixResult summarize(std::vector<float>& balances, float stockShare,
                           double goal)
{
    const int n = static_cast<int>(balances.size());
    MixResult m{};
    m.stockShare = stockShare;
    double sum = 0.0;
    int reached = 0;
    for (float b : balances) {
        sum += b;
        if (b >= static_cast<float>(goal)) ++reached;
    }
    m.chance = static_cast<double>(reached) / n;
    m.mean = sum / n;
    double squares = 0.0;
    for (float b : balances) squares += (b - m.mean) * (b - m.mean);
    m.stddev = std::sqrt(squares / (n - 1));

    std::sort(balances.begin(), balances.end());
    m.bad = balances[percentileIndex(BAD_PERCENTILE, n)];
    m.median = balances[percentileIndex(MEDIAN_PERCENTILE, n)];
    m.good = balances[percentileIndex(GOOD_PERCENTILE, n)];
    return m;
}

MixResult cpuSimulateMix(const Plan& plan, const std::vector<float>& stocks,
                         const std::vector<float>& bonds, float stockShare)
{
    std::vector<float> balances(plan.futures);
    for (int f = 0; f < plan.futures; ++f) {
        balances[f] = growOne(plan, stocks, bonds, stockShare, f);
    }
    return summarize(balances, stockShare, plan.goal);
}

double expectedBalance(const Plan& plan, double averageReturn)
{
    const double yearly = plan.monthly * MONTHS_PER_YEAR;
    double balance = plan.start;
    for (int y = 0; y < plan.years; ++y) {
        balance = (balance + yearly) * (1.0 + averageReturn);
    }
    return balance;
}
