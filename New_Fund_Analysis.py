"""
Author: Philip.Papasavvas
Date created (n-th version): 16/03/19
Date updated: 26/05/19

Class to analyse financial data.
Input data includes:
    - Financial data with dates, stocks and prices
    - Optional, mapping table between Bloomberg ticker and Fund name

To do:
 - Charts for lookback performance, limiting the number if more than 5 funds
 - Charts (2 subplots) for volatility of each stock and name properly, or do charting overlay
 - Produce yearly (& maybe monthly) returns
 - Plotting capabilities for stock charts
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
import utils
from re import sub, search
from utils import previousDate

import matplotlib.pyplot as plt
plt.style.use('seaborn')

dateToStr = lambda d: d.astype(str).replace('-', '')

# os.chdir("C://Users//Philip.P_adm//Documents//Fund Analysis")
wkdir = "C://Users//Philip//Documents//python//"
inputFolder = wkdir + "input/"
outputFolder = wkdir + "output/"

class Analysis():
    """
    Class to store clean historical prices for financial/fund data- will automatically remove
    NA values in the data.
    Methods for analysing returns and volatilities etc.

    Params:
        data: dataframe
            Financial data with the (date) index labelled 'Date'

        startDate & endDate: str "YYYY-MM-DD" or "YYYYMMDD", default None
            Lookback periods for analysing financial data

        tickerMapping: dataframe
            Mapping between Bloomberg ticker and name of the fund/security

        drop: bool, default False
            Drops rows that contain any NaNs, so all of the columns of df can
            be compared against each other

    # TODO: Plots to go to separate ones (when lots), and for the legends to be included

    # Returns:
    #     summaryTable: for each stock the annualised return and annualised
    #     volatility is displayed, as well as the Sharpe Ratio over the total
    #     lookback period
    """
    def __init__(self, data, startDate=None, endDate=None, tickerMapping=None, drop=False):

        # Load & clean data for efficient analysis for all products

        dataframe = utils.char_to_date(data)

        # check for NaN values and drop, alerting user for what has been dropped
        na_securities = dataframe.columns[dataframe.isnull().any()].values
        if len(na_securities) > 0:
            print("The following securities have NaNs in the dataset and are therefore excluded "
                  "from the analysis: \n {}".format(na_securities))

        beginDateStr = str(dt.datetime.strftime(dataframe.index.min(), "%Y-%m-%d"))
        finalDateStr = str(dt.datetime.strftime(dataframe.index.max(), "%Y-%m-%d"))
        print("Time series for data runs " + beginDateStr + " to " + finalDateStr + "\n")

        self.startDate = np.datetime64(startDate)
        self.endDate = np.datetime64(endDate)
        self.runDate = np.datetime64(dt.datetime.now().strftime("%Y-%m-%d"))

        if drop:
            dataframe.dropna(axis=0, inplace=True)

        try:
            df = dataframe.loc[startDate: endDate, : ]
        except KeyError:
            df = dataframe

        print("Data analysed for period " + startDate + " to " + endDate)

        if tickerMapping is not None:
            self.fundDict = dict(zip(tickerMapping['Ticker'], tickerMapping['Security Name']))
            if any(np.isnan(df.columns.map(self.fundDict).values)):
                pass
            else:
                df.columns = df.columns.map(self.fundDict)

        self.data = df
        # self.returns = None
        # self.summary = None

        self.set_output_folder()

    def set_output_folder(self):
        output_path = wkdir + "output/" + dateToStr(self.runDate) + "/"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            self.output_dir = output_path
        self.output_dir = output_path

    @staticmethod
    def daily_returns(data):
        """Give daily returns"""
        dailyReturn = data.pct_change(1).iloc[1:, ]
        return dailyReturn

    @staticmethod
    def log_daily_returns(data):
        """Give log daily returns"""
        log_dailyReturn = data.apply(lambda x: np.log(x) - np.log(x.shift(1)))[1:]
        return log_dailyReturn

    def annualReturns(self, toCsv=True):
        """
        Basic summary table of annual returns of stocks
        """

        df = self.data

        df = df.copy(True)
        df.dropna(axis=1, inplace = True) #drop NA columns
        annualReturn = (df.groupby(df.index.year).last()/ \
                        df.groupby(df.index.year).first() -1).T
        # annualReturn = round((annualReturn*100),3)
        annualReturn.index.name = "Fund / Annual Return"

        # if self.fundDict is not None:
        #     annualReturn.rename(index=self.fundDict, inplace=True)

        if toCsv:
            annualReturn.to_csv(self.output_dir + dateToStr(self.runDate) + " Securities Annual Return.csv")
            print("Summary table has been written to csv file in directory: " + self.output_dir)

    def summaryTable(self, toCsv = False, r = None):
        """
        Summarises return and volatility for input data over whole period

        Params:
            toCsv:  bool, default False. If True, written into output_dir
            r: float, default None
                Risk free rate of return,

        Returns:
            summary: table of returns and volatility of securities entered
        """

        df = self.data
        dailyReturn = df.pct_change(1).iloc[1:, ]
        annualReturn = np.mean(dailyReturn) * 252
        annualVolatility = np.std(dailyReturn) * np.sqrt(252)
        infoRatio = annualReturn / annualVolatility

        if r is None:
            summary = pd.concat([annualReturn, annualVolatility, infoRatio], axis=1)
            summary.columns = ['Annualised Return', 'Annual Volatility', 'Information Ratio']
        else:
            sharpe = (annualReturn - r) / annualVolatility
            summary = pd.concat([annualReturn, annualVolatility, infoRatio, sharpe], axis=1)
            summary.columns = ['Annualised Return', 'Annual Volatility', 'Information Ratio', 'Sharpe Ratio']

        summary.dropna(inplace=True)
        summary.index.name = "Fund/Stock"
        # summary = round(summary, 3)

        log = "Fund Stats for " + str(self.startDate) + " to " + str(self.endDate)
        errors = df.columns.difference(summary.index).values.tolist()

        print(log)
        if len(errors) >0:
            print("The following funds were not analysed due to errors in the dataset: ")
            print(errors)

        if toCsv:
            summary.to_csv(self.output_dir + "SummaryTable " + \
                           dateToStr(self.startDate) + "-" + dateToStr(self.endDate) + ".csv")
            print("Summary table has been written to csv file in directory: " + self.output_dir)

        return summary

    def lookbackPerformance(self, endDate = None, lookbackList = ["0D", "6M", "1Y", "2Y", "3Y"],
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

        if endDate is None:
            endDate = self.endDate

        if lookbackList is None:
            lookbackList = ["0D", "3M", "6M", "9M", "12M", "18M", "24M"]

        #TODO: if a date in the lookback is not in the range of the dataset then we drop this date
        target_dates = [previousDate(df, endDate, i) for i in lookbackList]
        target_prices = [df.loc[i,:].values for i in target_dates]

        # iloc[::-1] is to reverse the dataframe by the date index --> earliest to latest
        lookbackTable = pd.DataFrame.from_records(target_prices, index=target_dates, columns=df.columns)
        lookbackTable.sort_index(ascending = True, inplace=True)

        # Period return
        cumulativeReturn = lookbackTable.apply(lambda x: x/x[0])
        cumulativeReturn['Return Period'] = lookbackList
        cumulativeReturn = cumulativeReturn[cumulativeReturn.columns.tolist()[-1:] +
                                            cumulativeReturn.columns.tolist()[:-1]]

        if results:
            fileName = dateToStr(self.startDate) + "_" + dateToStr(self.endDate) + "_"
            writer = pd.ExcelWriter(self.output_dir + fileName + "Security Performance.xlsx")

            lookbackTable.index = lookbackTable.index.values.astype("datetime64[D]")
            lookbackTable_print = lookbackTable.T
            lookbackTable_print.to_excel(writer, "Prices")

            cumulativeReturn.index = cumulativeReturn.index.values.astype("datetime64[D]")
            cumulativeReturn.T.to_excel(writer, "Return")

            writer.save()
            print("Lookback performance table has been written to directory: {dry}".format(dry = self.output_dir))

        # Plotting the results
        # if returnPlot:
        #     data_to_plot = cumulativeReturn.drop(['Return Period'], axis =1)
        #     # plt.figure()
        #     if data_to_plot.shape[1] > 5:
        #         nSubplots = round(data_to_plot.shape[1]/5)
        #         for i in range(nSubplots):
        #             plt.figure()
        #             subset_data = data_to_plot.iloc[:,(5*i):(5*i)+5]
        #             plt.suptitle("Normalised Return Securities" + str(i))
        #             plt.plot(subset_data)
        #             plt.xlabel('Time Period')
        #
        #
        #     ax = cumulativeReturn.drop(['Return Period'], axis =1).plot()
        #
        #     vals = ax.get_yticks()
        #     ax.set_yticklabels(([format(x, ',') for x in vals]))
        #     plt.ylabel("Normalised Return")
        #     plt.grid()
        #     plt.tight_layout()
        #     plt.title("Normalised Return for funds")
        #     plt.legend(loc="upper left", fontsize='xx-small', ncol=2)
        #     plt.savefig(self.output_dir + "/CumulativeReturn Plot.png")
        #     plt.close()

    @staticmethod
    def bollinger_band(data, window, no_std):
        """Function to return bollinger bands for securities

        Inputs:
            data: df
                Dataframe of stock prices with index as np.datetime64
            window: int
                Rolling window for mean price and standard deviation
            no_std: int
                Number of standard deviations

        Returns:
            roll_mean, roll_std, boll_high, boll_low

        """
        roll_mean = data.rolling(window).mean()
        roll_std = data.rolling(window).std()

        boll_high = roll_mean + (roll_std * no_std)
        boll_low = roll_mean - (roll_std * no_std)

        return roll_mean, roll_std, boll_high, boll_low

    def plot_bollinger_bands(self, data, window=20, no_std=2):
        """Function to do bollinger band plots for each of the stocks in the dataframe"""

        for col in data.columns:
            slice = data.loc[:, col]
            normed_px = slice / slice[0]

            # Info for bollinger plots, also useful elsewhere
            roll_mn, roll_std, boll_high, boll_low = Analysis.bollinger_band(data=slice, window=window, no_std=no_std)

            # Plot the charts
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")
            ax1.plot(roll_mn, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(boll_high, linestyle="dashed", color="k", linewidth=0.5)
            ax1.plot(boll_low, linestyle="dashed", color="k", linewidth=0.5)

            norm_std_rolling = normed_px.rolling(window=window).std()
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Rolling Volatility', color=color)
            ax2.plot(norm_std_rolling, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 0.25)

            plt.suptitle(col + "\t (rolling {n}-day window)".format(n=window))
            # fig.tight_layout()
            plt.show()
            plt.savefig(self.output_dir + "{stock} Price & Vol History.png".format(stock=col))
            plt.close()

    @staticmethod
    def plot_total_return(data, output_dir, isLog=False):
        """
        Plot the normalised return over time, anchored back to start of lookback
        period

        Params:
            data:       Cleaned dataframe of stock prices
            output_dir: Where to print output
            isLog:      Log returns if True, simple if not
        """

        for col in data.columns:
            slice = data.loc[:, col]

            if isLog:
                normed_px = np.log(slice / slice[0])
            else:
                normed_px = slice / slice[0]

            # Plot the charts
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("Time")
            ax1.plot(normed_px, color=color)

            if isLog:
                ax1.set_ylabel("Log Total Return")
                plt.suptitle(col + " - Log Total Return")
            else:
                ax1.set_ylabel("Total Return")
                plt.suptitle(col + " - Total Return")

            plt.show()
            if isLog:
                plt.savefig(output_dir + "{stock} Log Total Return Chart.png".format(stock=col))
            else:
                plt.savefig(output_dir + "{stock} Total Return Chart.png".format(stock=col))
            plt.close()

    def rolling_sharpe_plots(self, data):
        #TODO: smooth out the curve as very noisy

        oned_returns = self.daily_returns(data)
        std_data = np.std(oned_returns)

        for col in data.columns:
            slice = data.loc[:,col]
            roll_mean = slice.rolling(window=5).mean()[5:]
            rolling_daily_rtns = roll_mean.pct_change(1).iloc[1:]

            annualised_return = np.mean(rolling_daily_rtns) * 252
            annual_volatility = np.std(rolling_daily_rtns)* np.sqrt(252)
            sharpe = annualised_return / annual_volatility

            # 21 business days in a month
            def rolling_sharpe(data):
                return (np.sqrt(21)*6) * (data.mean() / data.std())

            roll_sharpe = rolling_daily_rtns.rolling('180d').apply(rolling_sharpe)[1:]

            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Rolling Sharpe Ratio (3 month)")
            ax1.plot(roll_sharpe, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            #ax1.plot(sharpe, linestyle="dashed", color="k", linewidth=0.5)
            ax1.axhline(y=sharpe)

            plt.suptitle(col + "\t - rolling Sharpe ratio (3 month window)")
            # fig.tight_layout()
            plt.show()
            plt.savefig(self.output_dir + "{col} Rolling Sharpe Ratio.png".format(stock=col))
            plt.close()



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

if __name__ == "main":

    from New_Fund_Analysis import Analysis

    wkdir = "C://Users//Philip//Documents//python//"
    inputDir = wkdir + "input/"

    # "example_data.csv", "example_data_na.csv" has NA rows
    # df = pd.read_csv(inputDir + 'example_data.csv') #, parse_dates=True)
    df = pd.read_csv(inputDir + "funds_stocks_2019.csv")
    df = utils.char_to_date(df) #convert all dates to np datetime64
    df.set_index('Date', inplace=True)

    # tick_mapping = pd.read_csv(inputDir + 'tickerNameMapping.csv') #also:"tickerNameMapping.csv", 'securityMapping_subset.csv'

    # For manual debugging
    # input_data = df
    # startDate = "2016-01-01"
    # endDate = "2019-01-01"
    # tickerMapping= tick_mapping

    rn = Analysis(data = df, startDate = "2014-01-01", endDate = "2019-06-01", tickerMapping = None)
    rn.summaryTable(toCsv=True, r = 0.015)
    # rn.annualReturns(toCsv=True)
    # rn.lookbackPerformance(lookbackList = ["0D", "6M", "1Y", "2Y", "3Y"], results=True, returnPlot=False)
    # rn.plot_bollinger_bands(data = df[df.index > "2014-01-01"])

    data = rn.data
    Analysis.plot_total_return(data = rn.data, output_dir = outputFolder, isLog=False)



    # Look at the volatility on a rolling level



    # @staticmethod
    # def plot_bollinger_bands(self, data, window=20, no_std=2):
    #     """Function to do bollinger band plots for each of the stocks in the dataframe"""
    #
    #     for col in data.columns:
    #         slice = data.loc[:, col]
    #         normed_px = slice/slice[0]
    #
    #         # Info for bollinger plots, also useful elsewhere
    #         roll_mn, roll_std, boll_high, boll_low = Analysis.bollinger_band(data=slice, window=window, no_std= no_std)
    #
    #         # Plot the charts
    #         fig, ax1 = plt.subplots()
    #         color = 'tab:red'
    #         ax1.set_xlabel("Time")
    #         ax1.set_ylabel("Price")
    #         ax1.plot(roll_mn, color=color)
    #         ax1.tick_params(axis='y', labelcolor=color)
    #         ax1.plot(boll_high, linestyle="dashed", color="k", linewidth=0.5)
    #         ax1.plot(boll_low, linestyle="dashed", color="k", linewidth=0.5)
    #
    #         norm_std_rolling = normed_px.rolling(window=window).std()
    #         ax2 = ax1.twinx()
    #         color = 'tab:blue'
    #         ax2.set_ylabel('Rolling Volatility', color=color)
    #         ax2.plot(norm_std_rolling, color=color)
    #         ax2.tick_params(axis='y', labelcolor=color)
    #         ax2.set_ylim(0, 0.25)
    #
    #         plt.suptitle(col + "\t (rolling {n}-day window)".format(n=window))
    #         # fig.tight_layout()
    #         plt.show()
    #         plt.savefig(self.output_dir + "{stock} Price & Vol History.png".format(stock=col))
    #         plt.close()
    #
    #
    #     roll_mn, roll_std, boll_high, boll_low = Analysis.bollinger_band(data= data, window=20, no_std=2)
    #
    #
    #
    #     fig, ax1 = plt.subplots()
    #     color = 'tab:red'
    #     ax1.set_xlabel("Time")
    #     ax1.set_ylabel("Price")
    #     ax1.plot(roll_mn, color=color)
    #     ax1.tick_params(axis='y', labelcolor=color)
    #     ax1.plot(boll_high, linestyle="dashed", color="k", linewidth=0.5)
    #     ax1.plot(boll_low, linestyle="dashed", color="k", linewidth=0.5)
    #
    #     ax2 = ax1.twinx()
    #     color = 'tab:blue'
    #     ax2.set_ylabel('Rolling Volatility', color=color)
    #     ax2.plot(roll_std, color=color)
    #     ax2.tick_params(axis='y', labelcolor=color)
    #     ax2.set_ylim(0, 0.2)
    #
    #     plt.suptitle(rn.data.columns[3])
    #     # fig.tight_layout()
    #     plt.show()
    #
    #
    # @staticmethod
    # def bollinger_band(data, window, no_std):
    #     """Function to return bollinger bands for securities
    #
    #     Inputs:
    #         data: df
    #             Dataframe of stock prices with index as np.datetime64
    #         window: int
    #             Rolling window for mean price and standard deviation
    #         no_std: int
    #             Number of standard deviations
    #
    #     Returns:
    #         roll_mean, roll_std, boll_high, boll_low
    #
    #     """
    #     roll_mean = data.rolling(window).mean()
    #     roll_std = data.rolling(window).std()
    #
    #     boll_high = roll_mean + (roll_std * no_std)
    #     boll_low = roll_mean - (roll_std * no_std)
    #
    #     return roll_mean, roll_std, boll_high, boll_low
    #
    # roll_mn, roll_std, boll_high, boll_low = bollinger_band(data= res, window= 20, no_std=2)
    #
    # fig, ax1 = plt.subplots()
    # color = 'tab:red'
    # ax1.set_xlabel("Time")
    # ax1.set_ylabel("Price")
    # ax1.plot(roll_mn, color = color)
    # ax1.tick_params(axis='y', labelcolor= color)
    # ax1.plot(boll_high, linestyle = "dashed", color = "k", linewidth = 0.5)
    # ax1.plot(boll_low, linestyle="dashed", color="k", linewidth = 0.5)
    #
    # ax2 = ax1.twinx()
    # color = 'tab:blue'
    # ax2.set_ylabel('Rolling Volatility', color = color)
    # ax2.plot(roll_std, color = color)
    # ax2.tick_params(axis='y', labelcolor = color)
    # ax2.set_ylim(0,0.2)
    #
    # plt.suptitle(rn.data.columns[3])
    # #fig.tight_layout()
    # plt.show()
    #
    #
    # plt.figure()
    # plt.plot(res_roll_std)
    # plt.plot(rn.data.iloc[:,3])




    # b = pd.rolling_std(arg = a, window = 30) #rolling window is 30 days
    # col_labels = [i + "std" for i in b.columns]
    # b.columns = col_labels
    # c = pd.concat([a,b], axis = 1)
    # c[['MXWO Index', 'MXWO Index_std']].plot(subplots=True, color = "r", figsize = (10,6))

# def stockPerformance(dataframe, endDate, lookbackList = ["0D", "6M", "1Y", "2Y", "3Y"],
#                      print_pricesSummary = True, print_returnSummary = True,
#                      showPlots = False, fundMapping= None):
#     """"
#     Function --> to be made into a class, to take in dataframe of security prices, analyse
#     them and produce comparison tables.
#
#     Parameters:
#         dataframe:      consisting of date index, and historical prices for security
#         date:           date present in the dataframe, np.datetime64
#         lookbackList:   specify in "M" and/or "Y" the price comparison to be made,
#                         default "6M", "1Y", "2Y", "3Y"
#         fundMapping     mapping of the Bloomberg tickers to the Fund names
#
#     """
#     categoriesFile = pd.ExcelFile(fundMapping)
#     fundList = categoriesFile.parse("Sheet1").iloc[:,:2]
#     fundDict = dict(zip(fundList['Ticker'], fundList['Security Name']))
#
#     df = pd.read_csv("data_2019.csv", index_col="Date", parse_dates=True)
#     df.rename(columns=fundDict, inplace=True)
#
#     if lookbackList is None:
#         lookbackList = ["0D", "3M", "6M", "9M", "12M"]
#
#     dates  = [previousDate(df, endDate, i) for i in lookbackList]
#     prices = [df.loc[i,: ].values for i in dates]
#
#     ### Summary table of prices
#     # final iloc[::-1] is to reverse the dataframe by the date index to get from earliest to latest date
#     summaryTable = pd.DataFrame.from_records(prices, index= dates, columns=df.columns).iloc[::-1]
#
#     ### Period returns
#     cumulativeReturn = summaryTable.apply(lambda x: x/x[0]) - 1
#     returnSummary = round((cumulativeReturn * 100),2).astype(str) + "%"
#     returnSummary['Return Period'] = lookbackList[::-1] #reverse the order from earliest to latest
#     returnSummary = returnSummary[returnSummary.columns.tolist()[-1:] + returnSummary.columns.tolist()[:-1]]
#
#     # Normalised returns chart to earliest date specified in lookbackList,
#     # use rolling price over 5 days to remove noise from charts
#     # TODO: Develop later as this is not working currently
#     # subset_df = df.loc[dates[-1]: dates[0],:]
#     # subset_return_toStart = subset_df.apply(lambda x: x/x[0])
#     # subset_return_toStart.plot()
#     # subset_df_rolled = subset_df.rolling(window=10).mean()
#
#     if print_pricesSummary:
#       print("Prices summary: \n")
#       print(summaryTable)
#
#     if print_returnSummary:
#         print("Return summary: \n")
#         print(returnSummary)
#
#     if showPlots:
#         return_lookbackList = summaryTable.apply(lambda x: x/x[0])
#         return_lookbackList.plot()
#
# dataframe= "data_2019.csv"
# endDate = np.datetime64('2018-12-31')
# lookbackList = ['0D', '6M', '1Y', '2Y', '3Y', '4Y', '5Y']
# fundMapping = "fundList.xlsx"
# print_pricesSummary = True
# print_returnSummary = True
# showPlots = True
# stockPerformance(dataframe= "data_2019.csv", endDate= np.datetime64('2018-12-31'),
#                 lookbackList = ['0D', '6M', '1Y', '2Y', '3Y', '4Y', '5Y'], fundMapping = "fundList.xlsx",
#                 print_pricesSummary = True, print_returnSummary = True, showPlots = True)
