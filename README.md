# Projects

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
</p>

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/VivekPa/AIAlpha/blob/master/LICENSE) 
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A mixutre of different (small projects), focused on:
- daily email (the [CrossFit](http://www.crossfit.com) Workout of the Day (WOD)) using web scraping and pulling a random quote from a MongoDB Atlas database
- technical analysis of security price data 
- portfolio optimisation 
- stationarity (a stochastic process where the unconditional joint probability 
 distribution does not change when shifted in time) 

## Contents
- [Prerequisities](#prerequisites)
- [Fund Analysis](#fund-analysis)
- [Efficient Frontier](#efficient-frontier)
- [Stationarity](#stationarity)
- [Roadmap](#roadmap)
- [Other Modules](#other-modules)


## Prerequisites
The following modules are needed for the library
* [Arctic](https://github.com/manahl/arctic) >= 1.79.0
* [yfinance](https://github.com/ranaroussi/yfinance) >= 0.1.43
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [pymongo](https://github.com/mher/pymongo)
* [pyperclip](https://github.com/asweigart/pyperclip)
* [smtplib](https://docs.python.org/3/library/smtplib.html)

## CrossFit Daily WOD email & inspirational quote
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

## Fund Analysis 
module: *New_Fund_Analysis.py*

Technical analysis of security price data to calculate: (annualised & normalised) 
return, volatility, Sharpe & Information ratios.

Supports:
- [X] price data in long format, csv (date as index, columns as security prices)
- [X] mapping table as inputs between security identifier and name
- [ ] json input config 

Methods:
 - [X] Summary tables: performance (annualised, normalised and custom time-period) and correlation
 - [X] Plots: normalised returns & rolling volatility, bollinger bands, rolling Sharpe Ratio (noisy!)

Usage
```python
from New_Fund_Analysis import Analysis

rn = Analysis(data=dataframe, wkdir= wkdir)
rn.csv_summary(outputDir=os.path.join(wkdir, "output"))  
rn.plot_bollinger_bands(data=dataframe, window=60) # see below for example of returned plot
```

Development
- [ ] *Melt/vstack the securities data - summary table will then populate with NaNs if different lookbacks for 
each security, with a column to display number of observations. Print statement for
securities without data for entire lookback*
- [ ] *Add mapping for each fund to benchmark - as {key:value} - plotting of individual fund with benchmark (if any)*
- [ ] *Integrate **EfficientFrontier.py** script into (static) class method*

<!---
Plots
- Bollinger band plot - Monks Investment Trust.

![alt text][image] https://github.com/philip-papasavvas/projects/blob/master/images/MONKS%20INVESTMENT%20TRUST%20PLC%20Price%20%26%20Vol%20History.png "Example Bollinger Band & Rolling Volatility Plot"
-->

## Efficient Frontier
module: *Efficient Frontier.py*
- Markowitz portfolio optimisation for an input portfolio- includes calculation of key
technical measures (return, volatility, Sharpe Ratio).
- Optimisation is run for:
    1. Maximising Sharpe Ratio
    2. Minimising volatility
- *Development*
    - [ ] *Support json config as input*

## Stationarity
module: *Stationarity.py*
- Investigate time-series of securites, by computing daily returns assess if returns are stationary
- Augmented Dickey Fuller (ADF) test for unit roots, with null hypothesis,
  h<sub>0</sub> : &alpha; = 0.05, of non-stationary. Compute p-values for given threshold, default 
  &alpha; = 0.05. 
  Within this method skewness and kurtosis is calculated, to be compared with the assumption of normal returns.
- *Development*
    - [ ] *Translate this into a Jupyter Notebook to display the theory behind the ADF test, and list the hypotheses*

## Roadmap
I have the following module(s) planned out and hope to implement soon
                                                                                                                             
- finance-database.py
    - Download security data from Yahoo Finance using yfinance and store the data in MongoDB using Arctic
    - *Development*
        - [ ] *Schedule for daily run to download data and write to database (MongoDB)*
        - [ ] *Develop tool to cleanse data* 

## Contributing
Pull requests are welcome.

## Other Modules
- organise_files.py: *organise files by extension (supports xlsx and jpg) and move to specified folder*
- password_generator.py: *custom-length alphanumeric password (and if requested special characters)*

