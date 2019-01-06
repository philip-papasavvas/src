"""
Author: Philip.Papasavvas
Date created: 25/11/18

Class to create functions to analyse fund data
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
os.chdir('/Users/ppapasav/Desktop/Finance Analysis/')

class basicStockAnalysis():
    """
    Display returns and volatilities over the whole time period specified

    Params:
        file: dataframe
    Returns:
        summaryTable: for each stock the annualised return and annualised
        volatility is displayed, as well as the Sharpe Ratio over the total
        lookback period
    """
    def __init__(self, file):
        self.file = pd.DataFrame(pd.read_csv(file, parse_dates=True, index_col = 'Date'))
        self.returns = None
        self.summary = None

    def summaryTable(self, toClipboard = None, toCsv = None):
        """
        Summarises return and volatility for input data

        Params:
        toClipboard:    default None. If True, stored on clipboard
        toCsv:          default None. If True, written into cwd, with run date and
                        date range specified
        """
        df = self.file
        returns = df.pct_change(1).iloc[1:, ]
        annualReturn = np.mean(returns) * 252
        annualVolatility = np.std(returns) * np.sqrt(252)
        sharpeRatio = annualReturn / annualVolatility

        self.summary = pd.concat([annualReturn, annualVolatility, sharpeRatio], axis=1)
        self.summary.columns = ['Annualised Return', 'Annual Volatility', 'Sharpe Ratio']
        self.summary.sort_values(by = ['Sharpe Ratio'], ascending=False, inplace=True)
        self.summary.dropna(inplace=True)
        self.summary.index.name = "Fund/Stock"

        now = dt.datetime.now().strftime("%Y%m%d")
        beginDate = self.file.index.min().strftime("%Y%m%d")
        endDate = self.file.index.max().strftime("%Y%m%d")

        if toClipboard:
            self.summary.to_clipboard()

        if toCsv:
            self.summary.to_csv(now + " SummaryTable" + beginDate + "-" + endDate + ".csv")

eg = basicStockAnalysis('fulldata.csv')
eg.summaryTable(toClipboard=True)

df = pd.DataFrame(pd.read_csv('data.csv', index_col= 'Date', parse_dates=True))


class FundAnalysis():
    """
    Date in format 'YYYY-MM-DD'
    """
    def __init__(self, file, date1, date2):
        self.file = pd.DataFrame(pd.read_csv(file, index_col='Date', parse_dates=True))
        self.date1 = date1
        self.date2 = date2
        self.runDate = dt.datetime.today()

    def summaryTable(self):
        df = self.file.loc[self.date1:self.date2,:]
        dailyReturn = df.pct_change()
        dailyReturn = returns.iloc[1:,]
        annualReturn = np.mean(dailyReturn) * 252
        stdDev = np.std(dailyReturn) * np.sqrt(252)
        compoundAnnualGrowth = df.apply(lambda x: (x[-1]/ x[0])**(252/len(df)) -1)
        infoRatio = annualReturn/stdDev

        summary = pd.concat([annualReturn, stdDev, infoRatio], axis=1)
        summary.columns = ['Annualised Return', 'Annualised Volatility', 'Sharpe Ratio']
        summary.index.name = "Fund Stats for " + str(self.date1) + " to " + str(self.date2)

        summary.to_csv('BasicSummary' + str(self.runDate.date()) +'.csv')
        summary.to_clipboard(sep="\t")

run = FundAnalysis('fulldata.csv', date1 = '2018-01-01', date2 = '2018-07-01')
run.summaryTable()

###Development stuff
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

data = pdr.get_data_yahoo("SPY", start="2018-01-02", end="2018-01-3")
