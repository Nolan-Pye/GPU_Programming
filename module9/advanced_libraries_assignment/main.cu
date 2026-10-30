// CUDA Advanced Libraries assignment — Module 9
//
// Savings goal simulator: "If I save $500 a month for 30 years, what is
// my chance of reaching $1,000,000, and how does my stock/bond mix change
// that?" Real yearly returns from 1928-2025 drive a million possible
// futures.
//
//   cuRAND   draws one stock and one bond return per year for every
//            future: from a bell curve fitted to history (model=normal),
//            or by picking random real years (model=history)
//   Thrust   grows every future's savings for 6 stock/bond mixes, then
//            counts, sorts and averages the final balances
//
// cuRAND writes into Thrust's device vectors, so the returns never leave
// the GPU. The CPU repeats the simulation on the same returns to check
// the GPU and to time it, and the averages are checked against an exact
// formula. Prints PASS/FAIL for every check.
//
// Usage: savings_sim [key=value ...]
//   monthly=500 start=0 goal=1000000 years=30 futures=1000000
//   model=normal|history seed=2026

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <string>
#include <vector>
#include "cpu_check.h"
#include "simulator.cuh"

// Results of the GPU run, kept for the checks and the timing report.
struct GpuRun {
    std::vector<MixResult> mixes;
    double stockAverage;   // average of all random stock returns
    double bondAverage;
    float generateMs;      // cuRAND step
    float simulateMs;      // Thrust step, all mixes (average of repeats)
};

// ---- Command line ---------------------------------------------------------

static bool applyOption(const std::string& key, const std::string& value,
                        Plan& p)
{
    const char* v = value.c_str();
    if (key == "monthly") p.monthly = atof(v);
    else if (key == "start") p.start = atof(v);
    else if (key == "goal") p.goal = atof(v);
    else if (key == "years") p.years = atoi(v);
    else if (key == "futures") p.futures = atoi(v);
    else if (key == "seed") p.seed = strtoull(v, nullptr, 10);
    else if (key == "model" && value == "normal") p.model = Model::Normal;
    else if (key == "model" && value == "history") p.model = Model::History;
    else return false;
    return true;
}

static bool validPlan(const Plan& p)
{
    long long draws = static_cast<long long>(p.futures) * p.years;
    return p.monthly >= 0.0 && p.start >= 0.0 &&
           (p.monthly > 0.0 || p.start > 0.0) && p.goal > 0.0 &&
           p.years >= 1 && p.years <= MAX_YEARS &&
           p.futures >= MIN_FUTURES && draws <= MAX_DRAWS;
}

static Plan parsePlan(int argc, char** argv)
{
    Plan p{DEFAULT_MONTHLY, DEFAULT_START, DEFAULT_GOAL, DEFAULT_YEARS,
           DEFAULT_FUTURES, Model::Normal, DEFAULT_SEED};
    bool ok = true;
    for (int i = 1; i < argc && ok; ++i) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        ok = eq != std::string::npos &&
             applyOption(arg.substr(0, eq), arg.substr(eq + 1), p);
    }
    if (!ok || !validPlan(p)) {
        fprintf(stderr,
                "Usage: %s [key=value ...]\n"
                "  monthly=500 start=0 goal=1000000 years=1..%d\n"
                "  futures=%d.. (futures x years <= %lld)\n"
                "  model=normal|history seed=N\n",
                argv[0], MAX_YEARS, MIN_FUTURES, MAX_DRAWS);
        exit(EXIT_FAILURE);
    }
    return p;
}

// ---- Formatting -----------------------------------------------------------

// Whole dollars with thousands separators, e.g. $12,345.
static std::string money(double x)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%.0f", std::fabs(x));
    std::string digits(buf);
    std::string out = (x < 0) ? "-$" : "$";
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}

// Whole numbers with thousands separators, e.g. 1,000,000.
static std::string count(long long n)
{
    return money(static_cast<double>(n)).substr(1);
}

static void printAsset(const char* name, const AssetStats& s)
{
    printf("  %-25s average %5.2f%% a year, typical swing +/-%4.1f%%,\n"
           "  %-25s worst year %5.1f%% (%d)\n",
           name, 100.0 * s.mean, 100.0 * s.stddev, "", 100.0 * s.worst,
           s.worstYear);
}

static void printHistory(const ReturnHistory& h)
{
    printf("Return history: %d years, %d-%d (%s)\n",
           static_cast<int>(h.years.size()), h.years.front(),
           h.years.back(), RETURNS_FILE);
    printAsset("Stocks (S&P 500)", h.stockStats);
    printAsset("Bonds (10-year Treasury)", h.bondStats);
}

static void printPlan(const Plan& p)
{
    double yearly = p.monthly * MONTHS_PER_YEAR;
    printf("\nPlan:  save %s a month (%s a year) for %d years, "
           "starting from %s\n",
           money(p.monthly).c_str(), money(yearly).c_str(), p.years,
           money(p.start).c_str());
    printf("       you put in %s in total; goal %s\n",
           money(p.start + yearly * p.years).c_str(),
           money(p.goal).c_str());
    printf("Model: %s\n",
           p.model == Model::Normal
               ? "normal  (each year drawn from a bell curve fitted to "
                 "history)"
               : "history (each year is a randomly chosen real year)");
    printf("       %s simulated futures, seed %llu\n",
           count(p.futures).c_str(), p.seed);
}

static void printTable(const std::vector<MixResult>& mixes)
{
    printf("\n  Stocks/Bonds  Chance of goal   Bad case      Median"
           "   Good case     Average\n");
    for (const MixResult& m : mixes) {
        int stocks = static_cast<int>(std::lround(100.0 * m.stockShare));
        printf("  %6d/%-5d  %12.1f%%  %10s  %10s  %10s  %10s\n", stocks,
               100 - stocks, 100.0 * m.chance, money(m.bad).c_str(),
               money(m.median).c_str(), money(m.good).c_str(),
               money(m.mean).c_str());
    }
    printf("  Bad case: 1 future in 10 ends lower.  "
           "Good case: 1 in 10 ends higher.\n");
}

// ---- GPU run --------------------------------------------------------------

static float elapsedMs(cudaEvent_t from, cudaEvent_t to)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, from, to));
    return ms;
}

// The first call to each GPU function loads its code onto the GPU. A
// small untimed run (both models) does that, so the timings measure only
// the work itself.
static void warmUp(Plan plan, const ReturnHistory& history)
{
    plan.futures = MIN_FUTURES;
    Returns returns;
    thrust::device_vector<float> balances;
    for (Model model : {Model::Normal, Model::History}) {
        plan.model = model;
        generateReturns(plan, history, &returns);
        simulateMix(plan, returns, STOCK_SHARES[0], balances);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Every mix uses the same random returns, so differences between mixes
// come from the mix itself, not from luck.
static GpuRun runGpu(const Plan& plan, const ReturnHistory& history,
                     Returns& returns)
{
    GpuRun run{};
    thrust::device_vector<float> balances;
    cudaEvent_t start, mid, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&mid));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    generateReturns(plan, history, &returns);
    CUDA_CHECK(cudaEventRecord(mid));
    for (int repeat = 0; repeat < TIMED_REPEATS; ++repeat) {
        run.mixes.clear();  // every repeat gives the same results
        for (float share : STOCK_SHARES) {
            run.mixes.push_back(simulateMix(plan, returns, share, balances));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    run.generateMs = elapsedMs(start, mid);
    run.simulateMs = elapsedMs(mid, stop) / TIMED_REPEATS;

    size_t draws = static_cast<size_t>(plan.futures) * plan.years;
    run.stockAverage = averageOf(returns.stocks, draws);
    run.bondAverage = averageOf(returns.bonds, draws);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(mid));
    CUDA_CHECK(cudaEventDestroy(stop));
    return run;
}

// ---- Checks ---------------------------------------------------------------

static void check(int& failures, bool ok, const std::string& text)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", text.c_str());
    if (!ok) ++failures;
}

// cuRAND check: the random returns must average out to the historical
// average, within MAX_STANDARD_ERRORS.
static void checkAverage(int& failures, const char* name, double average,
                         const AssetStats& s, size_t draws)
{
    double standardError = s.stddev / std::sqrt(static_cast<double>(draws));
    double gap = std::fabs(average - s.mean) / standardError;
    char text[160];
    snprintf(text, sizeof(text),
             "random %s returns average %.3f%% (history %.3f%%, "
             "%.1f standard errors apart)",
             name, 100.0 * average, 100.0 * s.mean, gap);
    check(failures, gap <= MAX_STANDARD_ERRORS, text);
}

static double relDiff(double a, double b)
{
    return std::fabs(a - b) / std::max(std::fabs(b), 1.0);
}

// Largest difference between a GPU and a CPU result for one mix.
static double largestDiff(const MixResult& gpu, const MixResult& cpu)
{
    double d = std::fabs(gpu.chance - cpu.chance);
    d = std::max(d, relDiff(gpu.mean, cpu.mean));
    d = std::max(d, relDiff(gpu.bad, cpu.bad));
    d = std::max(d, relDiff(gpu.median, cpu.median));
    return std::max(d, relDiff(gpu.good, cpu.good));
}

// Copies the random returns to the CPU, repeats every mix there and
// compares. Returns the CPU time for all mixes, in milliseconds.
static double checkAgainstCpu(int& failures, const Plan& plan,
                              const Returns& returns,
                              const std::vector<MixResult>& gpuMixes)
{
    std::vector<float> stocks(returns.stocks.size());
    std::vector<float> bonds(returns.bonds.size());
    thrust::copy(returns.stocks.begin(), returns.stocks.end(),
                 stocks.begin());
    thrust::copy(returns.bonds.begin(), returns.bonds.end(), bonds.begin());

    auto start = std::chrono::steady_clock::now();
    double worst = 0.0;
    for (const MixResult& gpu : gpuMixes) {
        MixResult cpu = cpuSimulateMix(plan, stocks, bonds, gpu.stockShare);
        worst = std::max(worst, largestDiff(gpu, cpu));
    }
    std::chrono::duration<double, std::milli> ms =
        std::chrono::steady_clock::now() - start;

    char text[160];
    snprintf(text, sizeof(text),
             "GPU matches CPU on the same returns for all %d mixes "
             "(largest difference %.5f%%)",
             NUM_MIXES, 100.0 * worst);
    check(failures, worst <= GPU_CPU_TOLERANCE, text);
    return ms.count();
}

// Thrust check: each mix's average final balance must match the exact
// formula for its average yearly return, within MAX_STANDARD_ERRORS.
static void checkFormula(int& failures, const Plan& plan,
                         const ReturnHistory& h,
                         const std::vector<MixResult>& mixes)
{
    double worstGap = 0.0;
    for (const MixResult& m : mixes) {
        double average = m.stockShare * h.stockStats.mean +
                         (1.0 - m.stockShare) * h.bondStats.mean;
        double exact = expectedBalance(plan, average);
        double standardError = m.stddev / std::sqrt(plan.futures);
        worstGap = std::max(worstGap,
                            std::fabs(m.mean - exact) / standardError);
    }
    double exactAll = expectedBalance(plan, h.stockStats.mean);
    char text[200];
    snprintf(text, sizeof(text),
             "average balance matches the exact formula for all %d mixes "
             "(largest gap %.1f standard errors;\n"
             "         100%% stocks: simulated %s, formula %s)",
             NUM_MIXES, worstGap, money(mixes[0].mean).c_str(),
             money(exactAll).c_str());
    check(failures, worstGap <= MAX_STANDARD_ERRORS, text);
}

// ---- Report ---------------------------------------------------------------

static void printTiming(const Plan& plan, const GpuRun& run, double cpuMs)
{
    printf("\nTiming (%s futures x %d years, %d mixes)\n",
           count(plan.futures).c_str(), plan.years, NUM_MIXES);
    printf("  GPU  random returns (cuRAND)  %9.1f ms\n", run.generateMs);
    printf("  GPU  all mixes (Thrust)       %9.1f ms\n", run.simulateMs);
    printf("  CPU  all mixes                %9.1f ms  "
           "(GPU is %.0fx faster)\n",
           cpuMs, cpuMs / run.simulateMs);
}

static int run(int argc, char** argv)
{
    Plan plan = parsePlan(argc, argv);
    ReturnHistory history = loadHistory(RETURNS_FILE);
    printHistory(history);
    printPlan(plan);

    warmUp(plan, history);
    Returns returns;
    GpuRun run = runGpu(plan, history, returns);
    printTable(run.mixes);

    int failures = 0;
    size_t draws = static_cast<size_t>(plan.futures) * plan.years;
    printf("\nChecks\n");
    checkAverage(failures, "stock", run.stockAverage, history.stockStats,
                 draws);
    checkAverage(failures, "bond", run.bondAverage, history.bondStats,
                 draws);
    double cpuMs = checkAgainstCpu(failures, plan, returns, run.mixes);
    checkFormula(failures, plan, history, run.mixes);
    printTiming(plan, run, cpuMs);

    if (failures == 0) printf("\nAll checks PASSED\n");
    else printf("\n%d check(s) FAILED\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Thrust reports errors (such as running out of GPU memory) by throwing.
int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        fflush(stdout);
        fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
