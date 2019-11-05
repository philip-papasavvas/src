# Projects

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
</p>

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/VivekPa/AIAlpha/blob/master/LICENSE) 
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This repository has a few different small projects, including:
- Automated [daily email](#cf-wod) sending the [CrossFit](http://www.crossfit.com) Workout of the Day. This uses web scraping and pulling a random quote from a MongoDB Atlas database
- [Technical analysis](#securities-analysis) (risk-adjusted) of security price data 
- [Portfolio optimisation](#efficient-frontier)/Markowitz Efficient frontier 
- [Stationarity](#stationarity) (a stochastic process where the unconditional joint probability 
 distribution does not change when shifted in time) 

## Contents
- [Prerequisities](#prerequisites)
- [CrossFit WOD email](#cf-wod)
- [Securities Analysis](#securities-analysis)
- [Efficient Frontier](#efficient-frontier)
- [Stationarity](#stationarity)
- [Roadmap](#roadmap)
- [Other Modules](#other-modules)


## CF WOD
module: *cf_wod_email.py*  
- Script to scrape the [CrossFit](http://www.crossfit.com) website to get the daily WOD and add an 
inspirational quote (retrieved from MongoDB Atlas database) at the end, then send daily email to distribution list (specified by json config).
- Supports updating of quote database using an input json
- *Development*
    - [X] *Wrap up into a class (to factorise the functions)*
    - [X] *Set up task scheduler (on Raspberry Pi) to run daily*
    - [ ] *Store WODs to database (date as key)*
    - [ ] *Enrich email with embedded photo*

- Example
> The WOD for 20191013:  
> 4 rounds of Tabata row, bike, ski erg, jump rope, or other monostructural exercise.   
> **Hard work pays off** - *Josh Bridges*

## Securities Analysis 
module: *securities_analysis.py*

Technical analysis of security price data to calculate: (annualised & normalised) 
return, volatility, Sharpe & Information ratios.

Supports:
- [X] price data in long format, csv (date as index, columns as security prices)
- [X] mapping table as inputs between security identifier and name
- [ ] json input config 

Methods:
 - [X] Summary tables: performance (annualised, normalised and custom time-period) and correlation
 - [X] Risk adjusted measures: individual (static) methods for calculating [Information Ratio](https://www.investopedia.com/terms/i/informationratio.asp),
  [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp) and [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp)
 - [X] Plots: normalised returns & rolling volatility, bollinger bands, rolling Sharpe Ratio (noisy!)

Usage
```python
from securities_analysis import Analysis

# df is some financial data, wkdir defined here
rn = Analysis(data=df, wkdir= wkdir)
clean_df = rn.clean_slice_data(df=df)
results = rn.performance_summary(data=clean_df, save_results=True)
rn.plot_bollinger_bands(data=df, window=60) #see example of plot below
```

Development
- [ ] *Melt/vstack the securities data - summary table will then populate with NaNs if different lookbacks for 
each security, with a column to display number of observations. Print statement for
securities without data for entire lookback*
- [ ] *Add mapping for each fund to benchmark - as {key:value} - plotting of individual fund with benchmark (if any)*
- [ ] *Integrate **EfficientFrontier.py** script into (static) class method*

Plots
- Example: Bollinger Band & Rolling Volatility Plot - MONKS INVESTMENT TRUST
<img src="https://pythonpapshome.files.wordpress.com/2019/10/monks-investment-trust-plc-price-vol-history.png">

## Efficient Frontier
module: *efficient_frontier.py*
- Markowitz portfolio optimisation for an input portfolio- includes calculation of key
technical measures (return, volatility, Sharpe Ratio)
- Supports plotting of raw data using matplotlib
- Optimisation is run for both maximising Sharpe Ratio, and minimising volatility
```python
# df is some financial security price data
daily_return = df.pct_change()
mean_return = daily_return.mean()
covariance_return = daily_return.cov()
risk_free = 0.015

# return the performance of a four-securuity portfolio with equal weighting
sample_returns, sample_cov = port_perform_annual(mean_returns=mean_return, cov=covariance_return, weights=np.repeat(0.25,4))

# return optimal portfolio weighting by running analysis with 1000 different portfolio weight combinations
results, optimal_weights = random_portfolios(n_pts=1000, mean_returns=mean_return, cov=covariance_return, risk_free=0.015)

# Plot efficient frontier for portfolios (annualised return versus volatility) - bullet shape,
# highlighting most efficient portfolio (risk/reward and lowest volatility)
res = display_simulated_frontier_random(mean_returns= mean_return, cov= covariance_return, n_pts=500, \
                                        risk_free=0.015, wk_dir=wkdir, save_results=True, save_plots=False)
```
Console output
```
------------------------------------------
Maximum Sharpe Ratio Portfolio Allocation: 
 
 	 Sharpe Ratio: 	1.436 
 	 Annualised Return: 	0.1128  
 	 Annualised Volatility: 	0.0681 
 
             Dow Jones  MSCI World  Fundsmith Equity  LindsellTrain Global Eq
allocation      866.0      7701.0            1304.0                    130.0

------------------------------------------
Minimum volatility Portfolio Allocation: 
 	 Sharpe Ratio: 	1.436 
 	 Annualised Return: 	0.1128  
 	 Annualised Volatility: 	0.0681 
 
             Dow Jones  MSCI World  Fundsmith Equity  LindsellTrain Global Eq
allocation      866.0      7701.0            1304.0                    130.0
```

## Stationarity
module: *stationarity.py*
- Module with theory and explanation investigating time-series of securites, by computing daily returns assess if returns are stationary
- Augmented Dickey Fuller (ADF) test for unit roots, with null hypothesis,
  h<sub>0</sub> : &alpha; = 0.05, of non-stationarity. Compute p-values for given threshold, default 
  &alpha; = 0.05. 
  Within this method skewness and kurtosis is calculated, to be compared with the assumption of normal returns.
- *Development*
    - [ ] *Translate this into a Jupyter Notebook to display the theory behind the ADF test, and list the hypotheses*

## Quick start
Clone the repo: `git clone https://github.com/philip-papasavvas/projects.git`

## Roadmap
I have the following module(s) planned out and hope to implement soon
                                                                                                                             
- finance-database.py
    - Download security data from Yahoo Finance using yfinance and store the data in MongoDB using Arctic
    - *Development*
        - [ ] *Schedule for daily run to download data and write to database (MongoDB)*
        - [ ] *Develop tool to cleanse data* 


## Prerequisites
The following modules are needed for the library
* [Arctic](https://github.com/manahl/arctic) >= 1.79.0
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [pymongo](https://github.com/mher/pymongo)
* [pyperclip](https://github.com/asweigart/pyperclip)
* [requests](https://pypi.org/project/requests/2.7.0/)
* [smtplib](https://docs.python.org/3/library/smtplib.html)
* [yfinance](https://github.com/ranaroussi/yfinance) >= 0.1.43

## Contributing
Pull requests are welcome.

## Other Modules
- organise_files_extension.py: *organise files by extension (supports xlsx and jpg) and move to specified folder*
- password_generator.py: *custom-length alphanumeric password (and if requested special characters)*
