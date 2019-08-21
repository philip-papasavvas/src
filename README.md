# Projects

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
</p>

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/VivekPa/AIAlpha/blob/master/LICENSE) 
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


Small projects, focused on:
- Technical analysis of security price data
- Portfolio optimisation 
- Stationarity (a stochastic process where the unconditional joint probability 
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


## Fund Analysis 
module: *New_Fund_Analysis.py*

*Technical analysis of security price data to calculate: (annualised & normalised) 
return, volatility, Sharpe & Information ratios.*

**Supports**:
- [X] Price data in long format, csv (date as index, columns as security prices)
- [X] Mapping table as inputs between security identifier and name
- [ ] Configs in json format

**Methods**:
 - [X] Summary tables: performance (annualised, normalised and custom time-period) and correlation
 - [X] Plots: normalised returns & rolling volatility, bollinger bands, rolling Sharpe Ratio (noisy!)

**Usage**
```python
from New_Fund_Analysis import Analysis

rn = Analysis(data=dataframe, wkdir= wkdir)
rn.csv_summary(outputDir=os.path.join(wkdir, "output"))  
rn.plot_bollinger_bands(data=dataframe, window=60) # see below for example of returned plot
```

**Development**
- [ ] *Melt/vstack the securities data - summary table will then populate with NaNs if different lookbacks for 
each security, with a column to display number of observations. Print statement for
securities without data for entire lookback*
- [ ] *Add mapping for each fund to benchmark - as {key:value} - plotting of individual fund with benchmark (if any)*
- [ ] *Integrate **EfficientFrontier.py** script into (static) class method*

![][image] 

[image]: https://github.com/philip-papasavvas/projects/blob/master/MONKS%20INVESTMENT%20TRUST%20PLC%20Price%20%26%20Vol%20History.png "Example Bollinger Band & Rolling Volatility Plot"


 
## Efficient Frontier
module: *Efficient Frontier.py*

- *Markowitz portfolio optimisation for an input portfolio- includes calculation of key technical measures (return, volatility, Sharpe Ratio)*.
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
  Skewness and kurtosis is calculated, to be compared with the assumption of normal returns.
- *Development*
    - [ ] *Translate this into a Jupyter Notebook to display the theory behind the ADF test, and list the hypotheses*

## Roadmap
Modules due for implementation
- finance-database.py
    - Download security data from Yahoo Finance using yfinance and store the data in MongoDB using Arctic
    - *Development*
        - [ ] *Auto-run this for a given day and data stored down to database. Diagnostic tool to deal with bad data*


## Contributing
Pull requests are welcome.

## Other Modules

- organise_files.py: *Organise files by extension (supports xlsx and jpg) and move to specified folder*

- password_generator.py: *Custom-length alphanumeric password (and if requested special characters)*
    ```python
    import string
    import random
    random.seed(7) # to give the same result 

    def passwordGen(length, special=False):
        """Generates a password of a user given length, and can specify if want special characters"""
        lowers = list(string.ascii_lowercase)
        uppers = list(string.ascii_uppercase)
        nums = [str(i) for i in range(0,10)]
        specialChar = list("!\"£$%^&*()#@?<>")

        if special:
            charList = lowers + uppers + nums + specialChar
        else:
            charList = lowers + uppers + nums

        result = "".join([random.choice(charList) for i in range(0, length)])
        return result
        
    passwordGen(length = 8, special=True) # gives DVWqyfkr
    ````


