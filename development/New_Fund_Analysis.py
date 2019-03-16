"""
Author: Philip.Papasavvas
Date created (n-th version): 16/03/19

Class to analyse financial data.
Input data includes:
    - Financial data with dates, stocks and prices
    - Optional, mapping table between Bloomberg ticker and Fund name

Development:
 - If date provided not in index, default to nearest date to the error
 - Specification of return with lookback list = ["1M", "3M", "6M"]
 - Print out monthly returns, yearly returns
 - Plotting capabilities for stock charts
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
from re import sub, search
import matplotlib.pyplot as plt
dateToStr = lambda d: d.astype(str).replace('-', '')

# os.chdir("C://Users//Philip.P_adm//Documents//Fund Analysis")
os.chdir("C://Users//ppapasav//Documents//python//input")


# Load the data, clean the data for the N/A rows
df = pd.read_csv("cleaned_subset_funds_20190315.csv", index_col="Date", parse_dates=True)
df.dropna(axis=0, inplace=True)

### Functions  ###
# Define lookback periods from particular days
def datePlusTenorNew(date, pillar, reverse = False, expressInDays=False):
    '''
    Function to return the day count fraction(per year) between two dates based on convention

    Parameters
    ----------
        date : np.datetime64, np.array(np.datetime64[D])
        pillar : str, adds tenor to date, expressed as number letter pair
                examples of number letter pairs: 1d, 4D, 3w, 12W, 2M, 5m, 1y, 5Y
                other values allowed are 'spot', 'ON', 'O/N', 'SN', 'S/N'
        reverse: bool, default False. If False then the function will run to a time in the future,
                    if True then it can successfully subtract the pillar from the date
        expressInDay: bool, default False. If False a date is returned, if True an float is
                returned representing the number of days between date and pillar date / 365
    Returns
    -------
        newDate: np.datetime64, np.array(np.datetime64[D])
                 or if expressInDays = True: float, np.array()
    '''
    date = date.astype('datetime64[D]')
    if pillar == 'spot':
        newDate = date
    if bool(search('y|Y', pillar)):
        y = int(sub('\\D', '', pillar))
        yearly = date.astype('datetime64[Y]')
        monthly = date.astype('datetime64[M]')
        myDay = date.astype(int) - monthly.astype('datetime64[D]').astype(int) + 1
        myMonth = monthly.astype(int) - yearly.astype('datetime64[M]').astype(int) + 1
        if reverse == False:
            newDate = yearly + y
        else:
            newDate = yearly - y
        newDate = newDate.astype('datetime64[M]') + myMonth - 1
        # Leap year case, make sure not rolling into next month
        outmonth = newDate.copy()
        newDate = newDate.astype('datetime64[D]') + myDay - 1
        if outmonth != newDate.astype('datetime64[M]'):
            newDate = newDate.astype('datetime64[M]').astype('datetime64[D]') - 1
    if bool(search('m|M', pillar)):
        m = int(sub('\\D', '', pillar))
        monthly = date.astype('datetime64[M]')
        myDay = date.astype(int) - monthly.astype('datetime64[D]').astype(int) + 1
        if reverse == False:
            newDate = monthly + m
        else:
            newDate = monthly - m
        outmonth = newDate.copy()
        # add the days
        newDate = newDate.astype('datetime64[D]') + myDay - 1
        if outmonth != newDate.astype('datetime64[M]'):
            newDate = newDate.astype('datetime64[M]').astype('datetime64[D]') - 1
    if bool(search('w|W', pillar)):
        w = int(sub('\\D', '', pillar))
        if reverse == False:
            newDate = date + np.timedelta64(w * 7, 'D')
        else:
            newDate = date - np.timedelta64(w * 7, 'D')
    if bool(search('d|D', pillar)):
        d = int(sub('\\D', '', pillar))
        if reverse == False:
            newDate = date + np.timedelta64(d, 'D')
        else:
            newDate = date - np.timedelta64(d, 'D')

    if expressInDays:
            newDate = (newDate - date) / np.timedelta64(1, 'D')

        return newDate

def previousDate(dataframe, date, timeDifference):
    """
    Function to return a date in the past according to the input date provided
    for the dataframe being analysed

    Params:
        dataframe:
            Contains datetime information as well as financial data
        date: np.datetime64
        timeDifference:
            Expressed in terms of "M", "Y", "D", eg. "3M"
    """
    df = dataframe
    dateDelta = datePlusTenorNew(date, timeDifference, reverse=True)
    if dateDelta not in df.index:
        if dateDelta - 1 not in df.index:
            dateDelta = dateDelta - 2
        else:
            dateDelta = dateDelta - 1
    else:
        dateDelta = dateDelta
    return dateDelta

# Examples of above functions
# currentDate = np.datetime64("2018-11-23")
# dateDelta = datePlusTenorNew(currentDate, "6M", reverse = True)
# previousDate(dataframe=df, date= np.datetime64("2019-01-01"), timeDifference="3M")



### Class for Analysis ###

class Analysis():
    """
    Class to store clean historical prices for financial/fund data- will automatically remove
    NA values in the data.
    Methods for analysing returns and volatilities etc.

    Params:
        file: dataframe

        dates: in format "YYYY-MM-DD" or "YYYYMMDD"
            startDate & endDate: lookback periods for analysing financial data, default None

        tickerMapping: dataframe
            dataframe containing mapping between Bloomberg ticker and name of the fund/security

    # Returns:
    #     summaryTable: for each stock the annualised return and annualised
    #     volatility is displayed, as well as the Sharpe Ratio over the total
    #     lookback period
    """
    def __init__(self, file, startDate=None, endDate=None, tickerMapping=None):

        # Load & clean data for efficient analysis for all products

        dataframe = pd.DataFrame(pd.read_csv(file, parse_dates=True, index_col = 'Date'))
        dataframe.dropna(axis=0, inplace=True)
        if isinstance(startDate, np.datetime64) & isinstance(endDate, np.datetime64):
            df = dataframe.loc[startDate: endDate, : ]
        else:
            df = dataframe

        startDateStr = str(dt.datetime.strftime(df.index.min(), "%Y-%m-%d"))
        endDateStr = str(dt.datetime.strftime(df.index.max(), "%Y-%m-%d"))
        print("Time series for data runs " + startDateStr + " to " + endDateStr)

        if tickerMapping is not None:
            fundList = pd.read_csv(tickerMapping)
            self.fundDict = dict(zip(fundList['Ticker'], fundList['Security Name']))

        self.startDate = np.datetime64(startDateStr)
        self.endDate = np.datetime64(endDateStr)
        self.runDate = np.datetime64(dt.datetime.now().strftime("%Y-%m-%d"))
        self.data = df
        # self.returns = None
        # self.summary = None

        # make directory for outputs
        self.set_output_folder()

    def annualReturns(self, toCsv=True):
        """
        Basic summary table of annual returns of stocks
        """

        if isinstance(self.startDate, np.datetime64) & isinstance(self.endDate, np.datetime64):
            df = self.data.loc[self.startDate: self.endDate, :]
        else:
            df = self.data
        df = df.copy(True)
        df.dropna(axis=1, inplace = True) #drop NA columns
        annualReturn = (df.groupby(df.index.year).last()/ \
                        df.groupby(df.index.year).first() -1).T
        # annualReturn = round((annualReturn*100),3)
        annualReturn.index.name = "Fund / Annual Return"

        if self.fundDict is not None:
            annualReturn.rename(index=self.fundDict, inplace=True)

        if toCsv:
            annualReturn.to_csv(self.output_dir + "/" + dateToStr(self.runDate) + " Stock & Fund Annual Return Table.csv")
            print("Summary table has been written to csv file in current working directory: " + os.getcwd())

    def summaryTable(self, toCsv = False):
        """
        Summarises return and volatility for input data over whole period

        Params:
            toClipboard:    bool, default False. If True, stored on clipboard
            toCsv:          bool, default False. If True, written into cwd, with run date and
                        date range specified
        """

        # if isinstance(self.startDate, np.datetime64) & isinstance(self.endDate, np.datetime64):
        #     df = self.data.loc[self.startDate: self.endDate, : ]
        # else:
        #     df = self.data
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

        log = "Fund Stats for " + str(self.startDate) + " to " + str(self.endDate)
        errors = df.columns.difference(summary.index).values.tolist()

        print(log)
        if len(errors) >0:
            print("The following funds were not analysed due to errors in the dataset: ")
            print(errors)

        if toCsv:
            summary.to_csv(self.output_dir + "/" + "SummaryTable " + \
                           dateToStr(self.startDate) + "-" + dateToStr(self.endDate) + ".csv")
            print("Summary table has been written to csv file in current working directory: " + os.getcwd())

    def lookbackPerformance(self, endDate=self.endDate, lookbackList = ["0D", "6M", "1Y", "2Y", "3Y"],
                            results = False, returnPlot = False):
        """
        Analyse performance of certain funds over a custom lookback period

        Params:
            endDate:        type np.datetime64
                Defaults to last valid date in dataset
            lookbackList:   type list.
                default ["0D", "6M", "1Y", "2Y", "3Y"]
        """
        df = self.data

        if lookbackList is None:
            lookbackList = ["0D", "3M", "6M", "9M", "12M", "18M", "24M"]

        #TODO: if a date in the lookback is not in the range of the dataset then we drop this date
        target_dates = [previousDate(df, endDate, i) for i in lookbackList]
        target_prices = [df.loc[i,:].values for i in target_dates]

        # iloc[::-1] is to reverse the dataframe by the date index --> earliest to latest
        lookbackTable = pd.DataFrame.from_records(target_prices, index=target_dates,
                                                  columns=df.columns.map(self.fundDict)).iloc[::-1]

        # Period return
        cumulativeReturn = lookbackTable.apply(lambda x: x/x[0])
        cumulativeReturn['Return Period'] = lookbackList[::-1] # labelling
        cumulativeReturn = cumulativeReturn[cumulativeReturn.columns.tolist()[-1:] +
                                            cumulativeReturn.columns.tolist()[:-1]]

        if results:
            writer = pd.ExcelWriter(self.output_dir + "/Custom Lookback Performance.xlsx")
            lookbackTable.to_excel(writer, "Prices")
            cumulativeReturn.to_excel(writer, "Return")
            writer.save()

        # Plotting the results
        if returnPlot:
            #ax = cumulativeReturn.drop(['Return Period'], axis =1).plot()
            #vals = ax.get_yticks()
            #ax.set_yticklabels(([format(x, ',') for x in vals]))
            plt.ylabel("Normalised Return")
            plt.grid()
            plt.tight_layout()
            plt.title("Normalised Return for funds")
            plt.legend(loc="upper left", fontsize='xx-small', ncol=2)
            plt.savefig(self.output_dir + "/CumulativeReturn Plot.png")
            plt.close()

    def set_output_folder(self):
        output_path = wkdir + "/output/" + dateToStr(self.runDate)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            self.output_dir = output_path
        self.output_dir = output_path



    # def monthlyReturnTable(self):
    #     """Table for monthly returns"""
    #     if isinstance(self.startDate, np.datetime64) & isinstance(self.endDate, np.datetime64):
    #         df = self.data.loc[self.startDate: self.endDate, :]
    #     else:
    #         df = self.data
    #     df = df.copy(True)
    #     df.dropna(axis=1, inplace=True)
    #     df.index = df.index.strftime("%Y-%m-%d")
    #     # df.index['year'] = df.index.year
    #     # df.index['month'] = df.index.month


wkdir = "C://Users//ppapasav//Documents//python"
eg = Analysis(file = wkdir + "/input/" + 'cleaned_subset_funds_20190315.csv',
              startDate = np.datetime64("2016-01-01"), endDate = np.datetime64("2019-01-01"),
              tickerMapping = wkdir + "/input/" + "tickerNameMapping.csv")



def stockPerformance(dataframe, endDate, lookbackList = ["0D", "6M", "1Y", "2Y", "3Y"],
                     print_pricesSummary = True, print_returnSummary = True,
                     showPlots = False, fundMapping= None):
    """"
    Function --> to be made into a class, to take in dataframe of security prices, analyse
    them and produce comparison tables.

    Parameters:
        dataframe:      consisting of date index, and historical prices for security
        date:           date present in the dataframe, np.datetime64
        lookbackList:   specify in "M" and/or "Y" the price comparison to be made,
                        default "6M", "1Y", "2Y", "3Y"
        fundMapping     mapping of the Bloomberg tickers to the Fund names

    """
    categoriesFile = pd.ExcelFile(fundMapping)
    fundList = categoriesFile.parse("Sheet1").iloc[:,:2]
    fundDict = dict(zip(fundList['Ticker'], fundList['Security Name']))

    df = pd.read_csv("data_2019.csv", index_col="Date", parse_dates=True)
    df.rename(columns=fundDict, inplace=True)

    if lookbackList is None:
        lookbackList = ["0D", "3M", "6M", "9M", "12M"]

    dates  = [previousDate(df, endDate, i) for i in lookbackList]
    prices = [df.loc[i,: ].values for i in dates]

    ### Summary table of prices
    # final iloc[::-1] is to reverse the dataframe by the date index to get from earliest to latest date
    summaryTable = pd.DataFrame.from_records(prices, index= dates, columns=df.columns).iloc[::-1]

    ### Period returns
    cumulativeReturn = summaryTable.apply(lambda x: x/x[0]) - 1
    returnSummary = round((cumulativeReturn * 100),2).astype(str) + "%"
    returnSummary['Return Period'] = lookbackList[::-1] #reverse the order from earliest to latest
    returnSummary = returnSummary[returnSummary.columns.tolist()[-1:] + returnSummary.columns.tolist()[:-1]]

    # Normalised returns chart to earliest date specified in lookbackList,
    # use rolling price over 5 days to remove noise from charts
    # TODO: Develop later as this is not working currently
    # subset_df = df.loc[dates[-1]: dates[0],:]
    # subset_return_toStart = subset_df.apply(lambda x: x/x[0])
    # subset_return_toStart.plot()
    # subset_df_rolled = subset_df.rolling(window=10).mean()

    if print_pricesSummary:
      print("Prices summary: \n")
      print(summaryTable)

    if print_returnSummary:
        print("Return summary: \n")
        print(returnSummary)

    if showPlots:
        return_lookbackList = summaryTable.apply(lambda x: x/x[0])
        return_lookbackList.plot()

dataframe= "data_2019.csv"
endDate = np.datetime64('2018-12-31')
lookbackList = ['0D', '6M', '1Y', '2Y', '3Y', '4Y', '5Y']
fundMapping = "fundList.xlsx"
print_pricesSummary = True
print_returnSummary = True
showPlots = True
stockPerformance(dataframe= "data_2019.csv", endDate= np.datetime64('2018-12-31'),
                lookbackList = ['0D', '6M', '1Y', '2Y', '3Y', '4Y', '5Y'], fundMapping = "fundList.xlsx",
                print_pricesSummary = True, print_returnSummary = True, showPlots = True)
