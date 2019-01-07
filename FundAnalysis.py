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
from re import sub, search

os.chdir('/Users/ppapasav/Desktop/Finance Analysis/')

def timeDelta(date, period):
    """
    Function to return the date in the past specifying the current date and lookback period.

    Parameters
    ----------
        date: np.datetime64
        period: str, expressed as number letter pair. Allowed periods are weeks (w/W), months (m/M),
                    and years (y/Y)
                examples: "1W", "3m", "5Y"
    Returns
    ----------
        newDate: np.datetime64
    Example:
        # >>> timeDelta(date = np.datetime64("2018-12-01"), period = "3M")
    """
    date = date.astype(np.datetime64)
    if bool(search("w|W", period)):
        w = int(sub("\\D", "", period))
        newDate = date - np.timedelta64(w * 7, "D")
    if bool(search("m|M", period)):
        m = int(sub("\\D", "", period))
        monthDate = date.astype("datetime64[M]")
        day = date.astype(int) - monthDate.astype('datetime64[D]').astype(int) + 1
        newDate = (monthDate - m).astype('datetime64[D]') + day - 1
    if bool(search("y|Y", period)):
        y = int(sub("\\D", "", period))
        yearDate = date.astype("datetime64[Y]")
        monthDate = date.astype("datetime64[M]")
        day = date.astype(int) - monthDate.astype('datetime64[D]').astype(int) + 1
        monthDate = monthDate.astype(int) - yearDate.astype('datetime64[M]').astype(int) + 1
        newMonthDate = (yearDate - y).astype('datetime64[M]') + monthDate - 1
        newDate = newMonthDate.astype("datetime64[D]") + day - 1
    return newDate


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
        summary = round(summary, 3)

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


    def annualReturns(self, toCsv):
        """Basic table of annual returns"""
        if isinstance(self.startDate, np.datetime64) & isinstance(self.endDate, np.datetime64):
            df = self.data.loc[self.startDate: self.endDate, :]
        else:
            df = self.data
        df = df.copy(True)
        df.dropna(axis=1, inplace = True)
        annualReturn = (df.groupby(df.index.year).last()/ \
                        df.groupby(df.index.year).first() -1 ).T
        annualReturn = round((annualReturn*100),3)
        annualReturn.index.name = "Annual Return % / Date"

        if toCsv:
            # printing to csv currently not working
            #annualReturn.to_csv("Stock/Fund Annual Return.csv")
        print("Summary table has been written to csv file in current working directory: " + os.getcwd())


    def monthlyReturnTable(self):
        """Table for monthly returns"""
        if isinstance(self.startDate, np.datetime64) & isinstance(self.endDate, np.datetime64):
            df = self.data.loc[self.startDate: self.endDate, :]
        else:
            df = self.data
        df = df.copy(True)
        df.dropna(axis=1, inplace=True)
        df.index = df.index.strftime("%Y-%m-%d")
        # df.index['year'] = df.index.year
        # df.index['month'] = df.index.month



eg = basicStockAnalysis(file = 'fulldata.csv', startDate="2012-01-02", endDate="2014-01-01")
eg.summaryTable(toClipboard = False, toCsv = True)


###Development stuff
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

data = pdr.get_data_yahoo("SPY", start="2018-01-02", end="2018-01-3")
