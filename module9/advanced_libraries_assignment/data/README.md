# Return data

`annual_returns.csv` holds the yearly return of US stocks and US bonds for
every year from 1928 to 2025 (98 years).

| Column | Meaning |
|--------|---------|
| `Year` | Calendar year |
| `Stocks` | S&P 500 total return, dividends included (0.25 = +25%) |
| `Bonds` | 10-year US Treasury bond total return |

Returns are nominal, not adjusted for inflation.

| | Average | Typical swing (std dev) | Worst year |
|--|--:|--:|--:|
| Stocks | 11.85% | 19.4% | -43.8% (1931) |
| Bonds | 4.82% | 7.9% | -17.8% (2022) |

Source: Aswath Damodaran (NYU Stern), *Historical Returns on Stocks,
Bonds, Bills and Real Estate*, sheet "Returns by year", columns
"S&P 500 (includes dividends)" and "US T. Bond (10-year)":
<https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls>,
downloaded 2026-10-29. Values are rounded to six decimal places. The file
is committed so the build and run need no network access.
