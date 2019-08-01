# Projects

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
</p>

## Introduction

*This is a selection of **small projects** that I'll be working on in 2019.*

## Projects
1. New_Fund_Analysis.py: 
 - Class to analyse security price data to output fundamentals: (annualised) return, (annualised) volatility, Sharpe & Information ratio.
 - Summary tables produced: annualised performance, user-specified custom lookback performance, normalised returns, correlation
 - Methods for plotting: normalised returns & rolling volatility, (6m) rolling Sharpe Ratio, bollinger bands
 - *Development*
     - [ ] *Melt/vstack the securities data so that stats can be computed for lookbacks of each security regardless of when price history started. Summary table will then have columns: (start_date, end_date, no_observations, fund, year x return, year x+1 return, .....year x+n return), so if one fund started in 2012 and another in 2014, both funds in summary table but NaNs populate for smaller dataset. Flag to user which securities are not populated for lookback*
    - [ ] *Add key,value mapping for each fund and it's benchmark, then plotting each security with associated benchmark (if any)*
    - [ ] *Integrate efficient frontier script as class method for specified subset of securities*
    - [ ] *Change the input parameters to be in a config.json to read as inputs*
 
2. Efficient Frontier.py
- Markowitz portfolio optimisation using input security price data for a given portfolio. 
- Performance measures (annualised return, volatility, Sharpe Ratio) calculated, random portfolio allocation simulated and optimal portfolios identified for maximising Sharpe ratio, and minimising volatility.
- *Development*
    - [ ] *Change the input parameters to be in a config.json to read as inputs*

3. Stationarity.py
- Investigate time-series of securites, compute daily returns and assess if returns are stationary
- Augmented Dickey Fuller (ADF) test for unit roots, with null hypothesis, H_0, of non-stationary. Compute p-values for given threshold, default alpha = 5%. Within this method the skewness and kurtosis is calculated, to be compared with the assumption of normal returns.
- *Development*
    - [ ] *Translate this into a Jupyter Notebook to display the theory behind the ADF test, and list the hypotheses*

## Future project ideas
10. finance-database.py
- Download security data from Yahoo Finance using yfinance and store the data in MongoDB using Arctic
- *Development*
    - [ ] *Auto-run this for a given day and data stored down to database. Diagnostic tool to deal with bad data*


## Miscellaneous useful functions
101. organise_files.py: Organise files by extension (supports xlsx and jpg) and move to specified folder
102. password_generator.py: Generate custom-length alphanumeric password (and if requested special characters)


## Requirements
Currently works with and tested using:
* Python 3.6
* Pandas
* xlrd
* Arctic
