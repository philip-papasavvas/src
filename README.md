# AIAlpha - Financial Time-Series Analysis Framework

[![Made with Python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/philip-papasavvas/src/blob/master/LICENCE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A Python framework for analysing time-series financial security data, with modules for cloud database connectivity, financial data retrieval, and quantitative analysis.

## Project Structure

```
src/
├── api/                        # External API integrations
│   └── coinbase_api.py         # Coinbase cryptocurrency data & charting
│
├── dataload/                   # Database and data loading
│   ├── database.py             # MongoDB Atlas / Arctic connection & CRUD operations
│   ├── parser.py               # DataFrame parsing utilities
│   └── config/                 # Configuration templates
│
├── securityAnalysis/           # Financial security analysis
│   ├── utils_finance.py        # Returns, Sharpe/Sortino/Information ratios, volatility
│   └── stationarity.py         # ADF stationarity tests, descriptive statistics
│
├── utils/                      # General-purpose utilities
│   ├── decorators.py           # @timer, @deprecated decorators
│   ├── get_paths.py            # Path resolution helpers
│   ├── utils_dataframe.py      # DataFrame comparison & reconciliation
│   ├── utils_date.py           # Date format conversions
│   ├── utils_generic.py        # Array ops, dict helpers, file search, bucketing
│   └── utils_lists.py          # List flattening, chunking, deduplication
│
├── tests/                      # Unit tests for utils, parser, and integration
├── jupyter-notebooks/          # Educational notebooks (AR, Efficient Frontier, etc.)
├── simpy/                      # Discrete event simulation examples
└── miscellaneous/              # MATLAB/Octave examples
```

## Key Features

### Financial Analysis
- **Returns calculation**: relative, log, and absolute returns from price DataFrames
- **Risk metrics**: annualised Sharpe ratio, Sortino ratio, Information ratio
- **Volatility**: annualised volatility from daily price data
- **Stationarity testing**: Augmented Dickey-Fuller test with configurable significance levels
- **Descriptive statistics**: skewness, kurtosis, t-statistics, and hypothesis testing

### Data Infrastructure
- **MongoDB Atlas**: connect, read, write, and append to Arctic time-series libraries
- **Coinbase API**: fetch product stats, historical OHLCV data, and candlestick charts
- **Yahoo Finance**: download stock price data via `yfinance`

### Utilities
- DataFrame comparison and numeric reconciliation with configurable tolerance
- Linear bucketing, array conversion, and R-style `match()` function
- Recursive dict flattening, list chunking, and file search with regex patterns
- Date format conversions between NumPy, Excel serial, and string formats

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (for database features)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/philip-papasavvas/src.git
   cd src
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

For MongoDB features, create a config file at `src/dataload/config/mongo_private.json`:

```json
{
    "mongo_user": "YOUR_USERNAME",
    "mongo_pwd": "YOUR_PASSWORD",
    "url_cluster": "YOUR_CLUSTER_URL"
}
```

## Running Tests

From the `src/` directory:

```bash
# Run all utility tests
python -m pytest src/tests/ -v

# Run finance tests
python -m pytest src/securityAnalysis/tests/ -v

# Run a specific test file
python -m unittest src.tests.test_utils_generic
```

### Test Coverage

| Module | Test File | Functions Covered |
|--------|-----------|-------------------|
| `utils_date` | `test_utils_date.py` | `np_dt_to_str`, `excel_date_to_np`, `datetime_to_str`, `time_delta_to_days` |
| `utils_generic` | `test_utils_generic.py` | `average`, `difference`, `flatten_dict`, `dict ops`, `chunk_list`, `to_array`, `match`, `linear_bucketing`, `change_dict_keys` |
| `utils_lists` | `test_utils_lists.py` | `flatten`, `flatten_list`, `has_duplicates`, `all_unique`, `chunk`, `count_occurrences`, `list_as_comma_sep` |
| `utils_dataframe` | `test_utils_dataframe.py` | `replace_underscores_df`, `drop_null_columns_df`, `compare_dataframe_col`, `reconcile_dataframes_numeric`, `return_reconciliation_summary_table`, `get_selected_column_names`, `concat_columns` |
| `utils_finance` | `test_utils_finance.py` | `calculate_relative_return_from_array`, `calculate_security_returns` (all return types), `calculate_annual_return`, `calculate_annual_volatility`, `return_sharpe_ratio`, `return_sortino_ratio`, `return_information_ratio` |
| `parser` | `test_parser.py` | `get_columns`, `rename_columns`, `name_columns`, `sort_columns` |

## Jupyter Notebooks

Educational notebooks covering key financial and data science concepts:

- **Auto-Regression**: AR time-series modelling techniques
- **Efficient Frontier**: Modern Portfolio Theory and portfolio optimisation
- **Stationarity**: Time-series stationarity analysis and testing
- **Normally Distributed Returns**: Return distribution analysis
- **Python Tips & Tricks**: Python programming best practices
- **SQL Interview Questions**: SQL and data interview preparation

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `scipy` | Scientific computing and statistics |
| `statsmodels` | Statistical modelling (ADF test) |
| `matplotlib` / `mplfinance` | Data visualisation and financial charting |
| `pymongo` / `arctic` | MongoDB and time-series data storage |
| `yfinance` | Yahoo Finance API |
| `requests` | HTTP requests for API calls |

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
