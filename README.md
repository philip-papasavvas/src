# Projects

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
</p>

## Contents
- [Introduction](#introduction)
- [Prerequisities](#prerequisites)
- [Fund Analysis](#fund-analysis)
- [Efficient Frontier](#efficient-frontier)
- [Stationarity](#stationarity)
- [Roadmap](#roadmap)
- [Other Modules](#other-modules)

## Introduction

This is a selection of **small projects** that I'll be working on in 2019. They mainly focus around analysis of securities data,
I've predominantly focussed on mutual funds listed on the London Stock Exchange.


# Prerequisites
* [Arctic](https://github.com/manahl/arctic) >= 1.79.0
* [yfinance](https://github.com/ranaroussi/yfinance) >= 0.1.43
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [pymongo](https://github.com/mher/pymongo) >= 3.8.0


## Fund Analysis
- New_Fund_Analysis.py
- Module designed for analysing price data of a security/securities to calculate fundamentals: (annualised) return, (annualised) volatility, Sharpe & Information ratio.
 - Summary tables produced: annualised performance, user-specified custom lookback performance, normalised returns, correlation
 - Methods for plotting: normalised returns & rolling volatility, (6m) rolling Sharpe Ratio, bollinger bands
 - *Development*
     - [ ] *Melt/vstack the securities data so that stats can be computed for lookbacks of each security regardless of when price history started. Summary table will then have columns: (start_date, end_date, no_observations, fund, year x return, year x+1 return, .....year x+n return), so if one fund started in 2012 and another in 2014, both funds in summary table but NaNs populate for smaller dataset. Flag to user which securities are not populated for lookback*
    - [ ] *Add key,value mapping for each fund and it's benchmark, then plotting each security with associated benchmark (if any)*
    - [ ] *Integrate efficient frontier script as class method for specified subset of securities*
    - [ ] *Change the input parameters to be in a config.json to read as inputs*
 
## Efficient Frontier
- Efficient Frontier.py
- Markowitz portfolio optimisation for an input portfolio.
- Performance measures (annualised return, volatility, Sharpe Ratio) calculated, random portfolio allocation simulated and optimal portfolios identified for maximising Sharpe ratio, and minimising volatility.
- *Development*
    - [ ] *Change the input parameters to be in a config.json to read as inputs*

## Stationarity
Stationarity.py
- Investigate time-series of securites, compute daily returns and assess if returns are stationary
- Augmented Dickey Fuller (ADF) test for unit roots, with null hypothesis, H_0, of non-stationary. Compute p-values for given threshold, default alpha = 5%. Within this method the skewness and kurtosis is calculated, to be compared with the assumption of normal returns.
- *Development*
    - [ ] *Translate this into a Jupyter Notebook to display the theory behind the ADF test, and list the hypotheses*

## Roadmap
I have the following modules planned out and hope to implement soon
- finance-database.py
    - Download security data from Yahoo Finance using yfinance and store the data in MongoDB using Arctic
    - *Development*
        - [ ] *Auto-run this for a given day and data stored down to database. Diagnostic tool to deal with bad data*

## Other Modules
- organise_files.py: *organise files by extension (supports xlsx and jpg) and move to specified folder*
- password_generator.py: *custom-length alphanumeric password (and if requested special characters)*
