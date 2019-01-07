"""
Author: Philip.Papasavvas
Date created: 25/11/18

Class to create functions to analyse fund data
Development:
 - If date provided not in index, default to nearest date to the error
 - Mapping of Bloomberg Ticker to Fund Name
 - Specification of return with lookback list = ["1M", "3M", "6M"]
 - Print out monthly returns, yearly returns
 - Plotting capabilities for stock charts
 - Automatic scraping of prices

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
        dates: in format "YYYY-MM-DD" or "YYYYMMDD"
            startDate: start to analyse the financial data, default None
            endDate: end to analyse financial data, default None
    Returns:
        summaryTable: for each stock the annualised return and annualised
        volatility is displayed, as well as the Sharpe Ratio over the total
        lookback period
    """
    def __init__(self, file, startDate=None, endDate=None):
        self.data = pd.DataFrame(pd.read_csv(file, parse_dates=True, index_col = 'Date'))
        self.returns = None
        self.summary = None
        self.startDate = np.datetime64(startDate)
        self.endDate = np.datetime64(endDate)
        self.runDate = np.datetime64(dt.datetime.now().strftime("%Y-%m-%d"))

    def summaryTable(self, toClipboard = None, toCsv = None):
        """
        Summarises return and volatility for input data

        Params:
        toClipboard:    default None. If True, stored on clipboard
        toCsv:          default None. If True, written into cwd, with run date and
                        date range specified
        """
        if isinstance(self.startDate, np.datetime64) & isinstance(self.endDate, np.datetime64):
            df = self.data.loc[self.startDate: self.endDate, : ]
        else:
            df = self.data
        dailyReturn = df.pct_change(1).iloc[1:, ]
        annualReturn = np.mean(dailyReturn) * 252
        annualVolatility = np.std(dailyReturn) * np.sqrt(252)
        infoRatio = annualReturn / annualVolatility

        summary = pd.concat([annualReturn, annualVolatility, infoRatio], axis=1)

        summary.columns = ['Annualised Return', 'Annual Volatility', 'Information Ratio']
        summary.sort_values(by = ['Information Ratio'], ascending=False, inplace=True)
        summary.dropna(inplace=True)
        summary.index.name = "Fund/Stock"

        log = "Fund Stats for " + str(df.index.min().strftime("%Y-%m-%d")) + \
              " to " + str(df.index.max().strftime("%Y-%m-%d"))
        errors = df.columns.difference(summary.index).values.tolist()

        # beginDate = self.file.index.min().strftime("%Y%m%d")
        # endDate = self.file.index.max().strftime("%Y%m%d")

        print(log)
        print("The following funds were not analysed due to errors in the dataset: ")
        print(errors)

        if toClipboard:
            summary.to_clipboard()
            print("Summary table has been copied to the clipboard")

        if toCsv:
            summary.to_csv(str(self.runDate) + " SummaryTable" +
                           str(self.startDate) + "-" + str(self.endDate) + ".csv")
            print("Summary table has been written to csv file in current working directory: " + os.getcwd())

file = "fulldata.csv"
startDate="2012-01-02"
endDate="2014-01-01"
self = eg

eg = basicStockAnalysis(file = 'fulldata.csv', startDate="2012-01-02", endDate="2014-01-01")
eg.summaryTable(toClipboard = False, toCsv = True)




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
