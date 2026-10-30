// CSV loading and summary statistics for the yearly return history.

#include "history_data.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include "common.h"

static void fail(const char* path, const std::string& why)
{
    fprintf(stderr, "Error reading %s: %s\n", path, why.c_str());
    exit(EXIT_FAILURE);
}

// Reads one "year,stock,bond" line; returns false if it is malformed.
static bool parseRow(const std::string& line, ReturnHistory& h)
{
    int year = 0;
    float stock = 0.0f;
    float bond = 0.0f;
    if (sscanf(line.c_str(), "%d,%f,%f", &year, &stock, &bond) != 3) {
        return false;
    }
    h.years.push_back(year);
    h.stocks.push_back(stock);
    h.bonds.push_back(bond);
    return true;
}

static AssetStats summarize(const std::vector<float>& returns,
                            const std::vector<int>& years)
{
    const int n = static_cast<int>(returns.size());
    AssetStats s{0.0, 0.0, returns[0], years[0]};
    for (int i = 0; i < n; ++i) {
        s.mean += returns[i] / n;
        if (returns[i] < s.worst) {
            s.worst = returns[i];
            s.worstYear = years[i];
        }
    }
    double squares = 0.0;
    for (float r : returns) squares += (r - s.mean) * (r - s.mean);
    s.stddev = std::sqrt(squares / (n - 1));
    return s;
}

ReturnHistory loadHistory(const char* path)
{
    std::ifstream in(path);
    if (!in) fail(path, "cannot open file (run from the assignment folder)");

    ReturnHistory h;
    std::string line;
    std::getline(in, line);  // header: Year,Stocks,Bonds
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (!parseRow(line, h)) fail(path, "bad line: " + line);
    }
    if (static_cast<int>(h.years.size()) < MIN_HISTORY_YEARS) {
        fail(path, "too few years of returns");
    }
    h.stockStats = summarize(h.stocks, h.years);
    h.bondStats = summarize(h.bonds, h.years);
    return h;
}
