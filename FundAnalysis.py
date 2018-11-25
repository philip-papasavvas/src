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

class DataClean():
    """
    Clean data and produce returns and volatilities
    """
    def __init__(self, file):
        self.file = pd.DataFrame(pd.read_csv(file, parse_dates=True))
        self.returns = None
        self.summary = None

    def clean_data(self):
        """
        Cleans data
        """
        df = self.file
        returns = df.pct_change(1).iloc[1:, ]
        ret_an = np.mean(returns) * 252
        std_an = np.std(returns) * np.sqrt(252)
        sharpe = ret_an / std_an

        self.summary = pd.concat([ret_an, std_an, sharpe], axis=1)
        self.summary.to_clipboard()

eg = DataClean('fulldata.csv')
eg.clean_data()

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