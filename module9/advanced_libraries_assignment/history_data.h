// Loads the yearly stock and bond returns (data/annual_returns.csv).
#pragma once

#include <vector>

// Average, spread and worst year of one asset's yearly returns.
struct AssetStats {
    double mean;       // e.g. 0.1185 = 11.85% a year
    double stddev;     // typical swing around the average
    double worst;
    int worstYear;
};

struct ReturnHistory {
    std::vector<int> years;
    std::vector<float> stocks;   // S&P 500, dividends included
    std::vector<float> bonds;    // 10-year US Treasury bond
    AssetStats stockStats;
    AssetStats bondStats;
};

// Exits with a message if the file is missing or malformed.
ReturnHistory loadHistory(const char* path);
