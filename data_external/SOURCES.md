# External data (downloaded 2026-10-22, approved by the author)

| File | Source | Content | Coverage | SHA-256 (first 16) |
|---|---|---|---|---|
| `CMO-Historical-Data-Monthly.xlsx` | World Bank Commodity Price Data (Pink Sheet), "Monthly prices" link on https://www.worldbank.org/en/research/commodity-markets → https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx | sheet `Monthly Prices`, nominal USD; used: Iron ore, cfr spot ($/dmtu), Coal, Australian ($/mt), Crude oil, Brent ($/bbl) | 1960M01–2026M09, "Updated on October 02, 2026" | ea1c350827878ea3 |
| `FRED_PIORECRUSDM.csv` | FRED, https://fred.stlouisfed.org/series/PIORECRUSDM (IMF) | Global price of Iron Ore, USD/metric ton, NSA | 1992-01 – 2026-07 (complete file, 415 rows) | 94f2e617c942cd5e |
| `FRED_WPU1017.csv` | FRED, https://fred.stlouisfed.org/series/WPU1017 (BLS) | PPI by Commodity: Metals and Metal Products: Steel Mill Products, 1982=100, NSA | 2000-01 – 2026-09 (rows from 2000-01 kept; FRED file starts 1939-01) | a54486673a6999eb |
| `FRED_CPIAUCSL.csv` | FRED, https://fred.stlouisfed.org/series/CPIAUCSL (BLS); downloaded 2026-10-22, approved by the author | CPI for All Urban Consumers, All Items, U.S. City Average, 1982–84=100, seasonally adjusted | 2000-01 – 2026-09 (rows from 2000-01 kept; FRED file starts 1947-01); 2025-10 is blank in the source (not published by BLS) | 24c03d8062d14e25 |
| `FRED_WPU1012.csv` | FRED, https://fred.stlouisfed.org/series/WPU1012 (BLS) | PPI by Commodity: Metals and Metal Products: Iron and Steel Scrap, 1982=100, NSA | 2000-01 – 2026-09 (rows from 2000-01 kept; FRED file starts 1947-01) | afb6f1c74d1fe6e2 |

Notes
- The Pink Sheet URL changes with every monthly release; the older address
  (`.../5d903e848db1d1b83e0ec8f744e55570-0350012021/...`) still serves the January 2025 file.
- FRED CSVs were fetched from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>` through
  the browser (direct requests from this machine were reset by the server). Row counts and value
  sums were checked against the downloaded text (PIORECRUSDM 415 rows, sum 26631.816; WPU1017
  321 rows, sum 66269.064; WPU1012 321 rows, sum 138843.9).
- CPIAUCSL: 321 rows, sum of values 75503.943 (one blank, 2025-10), checked against the downloaded text.
  `validate_long_horizon.py --deflator CPIAUCSL` fills 2025-10 by log-linear interpolation between
  Sep and Nov 2025. The seasonally adjusted CPI is used to deflate the not seasonally adjusted PPI;
  seasonality in US CPI is small relative to steel price changes.
- PIORECRUSDM is an annual contract price repeated monthly before Dec 2008; only 2017+ is used.
- BLS PPIs are revised for four months after first release; the vintage above is final for
  months up to about May 2026.
