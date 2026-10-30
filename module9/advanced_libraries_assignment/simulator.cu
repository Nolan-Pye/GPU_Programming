// cuRAND + Thrust: random yearly returns in, savings outcomes out.
//
// cuRAND writes the random returns straight into Thrust device_vectors,
// and every Thrust call reads them in place: nothing is copied back to
// the CPU except the handful of summary numbers.

#include "simulator.cuh"

#include <cmath>
#include <curand.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace {

// Raw device pointers, for cuRAND and for the functors below.
float* ptr(thrust::device_vector<float>& v)
{
    return thrust::raw_pointer_cast(v.data());
}

const float* ptr(const thrust::device_vector<float>& v)
{
    return thrust::raw_pointer_cast(v.data());
}

// Turns a uniform random number in (0, 1] into the return of one
// randomly chosen year of the history table.
struct PickYear {
    const float* table;
    int count;
    __device__ float operator()(float u) const
    {
        int k = min(static_cast<int>(u * count), count - 1);
        return table[k];
    }
};

// Grows one future's savings: each year the year's savings are added,
// then the whole balance earns that year's mix of stock and bond returns.
struct GrowSavings {
    const float* stocks;
    const float* bonds;
    int futures;
    int years;
    float stockShare;
    float yearlySaving;
    float start;
    __device__ float operator()(int future) const
    {
        float balance = start;
        for (int y = 0; y < years; ++y) {
            size_t i = static_cast<size_t>(y) * futures + future;
            float r = stockShare * stocks[i] + (1.0f - stockShare) * bonds[i];
            r = fmaxf(r, WORST_YEAR_RETURN);
            balance = (balance + yearlySaving) * (1.0f + r);
        }
        return balance;
    }
};

struct ReachesGoal {
    float goal;
    __device__ bool operator()(float balance) const
    {
        return balance >= goal;
    }
};

struct SquaredGap {
    double mean;
    __device__ double operator()(float x) const
    {
        return (x - mean) * (x - mean);
    }
};

// cuRAND's pseudo-random normal generator works in pairs, so buffers are
// rounded up to an even length.
size_t evenCount(size_t n)
{
    return n + n % 2;
}

}  // namespace

// Bell-curve model: stock and bond returns are drawn separately, each
// with its historical average and swing.
static void drawNormal(curandGenerator_t gen, const ReturnHistory& h,
                       Returns* out)
{
    const AssetStats& s = h.stockStats;
    const AssetStats& b = h.bondStats;
    CURAND_CHECK(curandGenerateNormal(gen, ptr(out->stocks),
                                      out->stocks.size(), (float)s.mean,
                                      (float)s.stddev));
    CURAND_CHECK(curandGenerateNormal(gen, ptr(out->bonds),
                                      out->bonds.size(), (float)b.mean,
                                      (float)b.stddev));
}

// History model: each simulated year copies one real year. A single
// random pick sets both the stock and the bond return, so years such as
// 2008 (stocks -37%, bonds +20%) stay together.
static void drawFromHistory(curandGenerator_t gen, const ReturnHistory& h,
                            Returns* out)
{
    thrust::device_vector<float> stockTable(h.stocks.begin(),
                                            h.stocks.end());
    thrust::device_vector<float> bondTable(h.bonds.begin(), h.bonds.end());
    const int count = static_cast<int>(h.stocks.size());

    // The picks are stored in `bonds` first. Stocks are looked up from
    // them, then the bond lookup overwrites the picks in place.
    CURAND_CHECK(curandGenerateUniform(gen, ptr(out->bonds),
                                       out->bonds.size()));
    thrust::transform(out->bonds.begin(), out->bonds.end(),
                      out->stocks.begin(), PickYear{ptr(stockTable), count});
    thrust::transform(out->bonds.begin(), out->bonds.end(),
                      out->bonds.begin(), PickYear{ptr(bondTable), count});
}

void generateReturns(const Plan& plan, const ReturnHistory& history,
                     Returns* out)
{
    const size_t draws = static_cast<size_t>(plan.futures) * plan.years;
    out->futures = plan.futures;
    out->years = plan.years;
    out->stocks.resize(evenCount(draws));
    out->bonds.resize(evenCount(draws));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, plan.seed));
    if (plan.model == Model::Normal) drawNormal(gen, history, out);
    else drawFromHistory(gen, history, out);
    CURAND_CHECK(curandDestroyGenerator(gen));
}

double averageOf(const thrust::device_vector<float>& values, size_t count)
{
    return thrust::reduce(values.begin(), values.begin() + count, 0.0) /
           count;
}

// Chance of reaching the goal, average, spread and percentiles of the
// final balances. Sorting puts the percentiles at fixed positions.
static MixResult summarize(thrust::device_vector<float>& balances,
                           float stockShare, double goal)
{
    const int n = static_cast<int>(balances.size());
    MixResult m{};
    m.stockShare = stockShare;
    m.chance = static_cast<double>(thrust::count_if(
                   balances.begin(), balances.end(),
                   ReachesGoal{static_cast<float>(goal)})) / n;
    m.mean = averageOf(balances, n);
    double squares = thrust::transform_reduce(
        balances.begin(), balances.end(), SquaredGap{m.mean}, 0.0,
        thrust::plus<double>());
    m.stddev = std::sqrt(squares / (n - 1));

    thrust::sort(balances.begin(), balances.end());
    m.bad = balances[percentileIndex(BAD_PERCENTILE, n)];
    m.median = balances[percentileIndex(MEDIAN_PERCENTILE, n)];
    m.good = balances[percentileIndex(GOOD_PERCENTILE, n)];
    return m;
}

MixResult simulateMix(const Plan& plan, const Returns& returns,
                      float stockShare,
                      thrust::device_vector<float>& balances)
{
    GrowSavings grow{ptr(returns.stocks),
                     ptr(returns.bonds),
                     returns.futures,
                     returns.years,
                     stockShare,
                     static_cast<float>(plan.monthly * MONTHS_PER_YEAR),
                     static_cast<float>(plan.start)};
    balances.resize(returns.futures);
    thrust::counting_iterator<int> first(0);
    thrust::transform(first, first + returns.futures, balances.begin(),
                      grow);
    return summarize(balances, stockShare, plan.goal);
}
