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
from re import sub, search
import utils
from utils import previousDate, prep_fund_data

pd.set_option('display.max_columns', 5)

import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
plt.style.use('ggplot')
plt.tight_layout()

dateToStr = lambda d: d.astype(str).replace('-', '')

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
    def __init__(self, data, wkdir, startDate=None, endDate=None, tickerMapping=None, drop_na=False):

        """
        Parameters
        ----------

        data: dataframe
            Security data of price over time period
        wkdir: str
            Path of working directory
        startDate: str, default None
            Analyse data from the specified start date in the format YYYY-MM-DD. Will default to start of
            time series
        endDate: str, default None
            Analyse data from the specified end date in the format YYYY-MM-DD. Will default to end of
            time series
        drop_na: bool, default False
            Drop NA columns to not pollute the results
        tickerMapping: dataframe
            Mapping between the BBG Ticker of the security, and
            the security name
            #TODO: make the above into a dict with keys and values
        """

        # Load & clean data for efficient analysis for all products

        dataframe = utils.char_to_date(data)

        # check for NaN values and drop, alerting user for what has been dropped
        na_securities = dataframe.columns[dataframe.isnull().any()].values
        if len(na_securities) > 0:
            print("The following securities have NaNs in the dataset but "
                  "will be included in the analysis: \n {}".format(na_securities))

        beginDateStr = str(dt.datetime.strftime(dataframe.index.min(), "%Y-%m-%d"))
        finalDateStr = str(dt.datetime.strftime(dataframe.index.max(), "%Y-%m-%d"))
        print("Time series for data runs " + beginDateStr + " to " + finalDateStr + "\n")

        if startDate is not None:
            self.startDate = np.datetime64(startDate)
        else:
            self.startDate = np.datetime64(beginDateStr)
        if endDate is not None:
            self.endDate = np.datetime64(endDate)
        else:
            self.endDate = np.datetime64(finalDateStr)

        self.runDate = np.datetime64(dt.datetime.now().strftime("%Y-%m-%d"))

        if drop_na:
            dataframe.dropna(axis=0, inplace=True)

        try:
            df = dataframe.loc[startDate: endDate, : ]
        except KeyError:
            df = dataframe

        print("Data analysed for period " + self.startDate.astype(str) +
              " to " + self.endDate.astype(str))

        if tickerMapping is not None:
            self.fundDict = dict(zip(tickerMapping['Ticker'], tickerMapping['Security Name']))
            if any(np.isnan(df.columns.map(self.fundDict).values)):
                pass
            else:
                df.columns = df.columns.map(self.fundDict)

        self.data = df
        self.wkdir = wkdir
        # self.returns = None
        # self.summary = None

        self.set_output_folder()
        print("Output folder set as " + self.output_dir)

    def set_output_folder(self):
        """Set output """
        output_path = self.wkdir + "output/" + dateToStr(self.runDate) + "/"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            self.output_dir = output_path
        self.output_dir = output_path

    def daily_returns(self, data):
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
        #df.dropna(axis=1, inplace = True) #drop NA columns
        annualReturn = (df.groupby(df.index.year).last()/ \
                        df.groupby(df.index.year).first() -1).T
        # annualReturn = round((annualReturn*100),3)
        annualReturn.index.name = "Fund / Annual Return"

        if toCsv:
            annualReturn.to_csv(self.output_dir + dateToStr(self.runDate) + " Securities Annual Return.csv")
            print("Summary table has been written to csv file in directory: " + self.output_dir)

        return annualReturn

    def summaryTable(self, toCsv = False, r = 1):
        """
        Summarises return and volatility for input data over whole period

        Params:
            toCsv:  bool, default False. If True, written into output_dir
            r: float, default None
                Risk free rate of return, default 1% (you can get this in banks)

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
            #ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(boll_high, linestyle="dashed", color="k", linewidth=0.5)
            ax1.plot(boll_low, linestyle="dashed", color="k", linewidth=0.5)
            ax1.yaxis.set_major_locator(mtick.LinearLocator(6)) # set there to be N=6 lines on y-axis

            norm_std_rolling = normed_px.rolling(window=window).std()
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Rolling Volatility')
            ax2.plot(norm_std_rolling, color=color)
            #ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 0.25)
            ax2.yaxis.set_major_locator(mtick.LinearLocator(6))

            plt.suptitle(col + " (rolling {n}-day window)".format(n=window))
            # fig.tight_layout()
            plt.show()
            plt.savefig(self.output_dir + "{stock} Price & Vol History.png".format(stock=col))
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

    @staticmethod
    def plot_total_return(data, output_dir, isLog=False):
        """
        Plot the normalised return over time, anchored back to start of lookback
        period
        """

        for col in data.columns:
            slice = data.loc[:, col]

            if isLog:
                normed_px = 1 + np.log(slice/slice[0])
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
                plt.savefig(os.path.join(output_dir, "{stock} - Log Total Return Chart.png".format(stock = col)))
            else:
                plt.savefig(os.path.join(output_dir,  "{stock} - Total Return Chart.png".format(stock=col)))
            plt.close()

    def csv_summary(self, outputDir):
        """Print the summary to Excel"""

        if outputDir is None:
            outputDir = self.output_dir

        writer = pd.ExcelWriter(os.path.join(outputDir, "Stock Summary Measures.xlsx"))


        # Summary table of return/volatility/info ratio/sharpe ratio
        summary_one = self.summaryTable(toCsv=False, r=0.01)

        summary_one[['Annualised Return', 'Annual Volatility']] *= 100
        summary_one[['Annualised Return', 'Annual Volatility']] = \
            summary_one[['Annualised Return', 'Annual Volatility']].applymap("{0:.2f}%".format)
        summary_one[['Information Ratio', 'Sharpe Ratio']] = summary_one[['Information Ratio', 'Sharpe Ratio']].applymap("{0:.4}".format)

        summary_one.to_excel(writer, "Summary Table")

        # Useful for printing the display output, but str not useful elsewhere
        # summ_one_str = summary_one.to_string(formatters={
        #     'Annualised Return': '{:,.2%}'.format,
        #     'Annual Volatility': '{:,.2%}'.format,
        #     'Information Ratio': '{:,.3f}'.format,
        #     'Sharpe Ratio': '{:,.3f}'.format,
        # })


        # Annual Returns
        annual_table = self.annualReturns(toCsv=False)
        annual_table.columns = annual_table.columns.astype(str)

        print_annual_table = annual_table*100
        print_annual_table = print_annual_table.applymap("{0:.2f}%".format)
        print_annual_table.to_excel(writer, "Annual Return")

        # Correlation matrix
        correlation_mat = self.data.corr()
        correlation_mat.to_excel(writer, "Correlation")

        writer.save()
        print("Summary statistics produced, and in the following directory: " + self.output_dir)


if __name__ == "main":

    from New_Fund_Analysis import Analysis

    # "example_data.csv", "example_data_na.csv" has NA rows, master= funds_stocks_2019.csv
    # os.chdir("C://Users//Philip.P_adm//Documents//Fund Analysis")

    wkdir = "C://Users//Philip//Documents//python//"
    inputFolder = os.path.join(wkdir, "input")
    outputFolder = os.path.join(wkdir, "output")
    df = prep_fund_data(df_path= os.path.join(inputFolder, "example_data_na.csv"))

    rn = Analysis(data=df, wkdir= wkdir)
    rn.csv_summary(outputDir=os.path.join(wkdir, "output"))
    rn.plot_bollinger_bands(data=df, window=60)

    # tick_mapping = pd.read_csv(inputDir + 'tickerNameMapping.csv') #also:"tickerNameMapping.csv", 'securityMapping_subset.csv'

    # rn = Analysis(data = df, startDate = "2014-01-01", endDate = "2019-06-01", tickerMapping = None)
    # rn.summaryTable(toCsv=True, r = 0.015)
    # rn.annualReturns(toCsv=True)
    # rn.lookbackPerformance(lookbackList = ["0D", "6M", "1Y", "2Y", "3Y"], results=True, returnPlot=False)
    # rn.plot_bollinger_bands(data = df[df.index > "2014-01-01"])
    #
    # data = rn.data
    # Analysis.plot_total_return(data = rn.data, output_dir = outputFolder, isLog=False)

    #inputDir = os.path.join(inputDir, "security_data")
    # outputDir = os.path.join(inputFolder, "output")
    #
    # # cut the dataframe and only look at the nulls
    # df = df.loc[:, df.isnull().sum() != 0]
    # null_lst = list(df.isnull().sum().values)  # list of first null values
    #
    # for i, e in enumerate(null_lst):
    #     slice = df.iloc[e:, i]
    #     slice.dropna(axis=0, inplace=True)
    #     startDate = slice.index[0].strftime("%Y-%m-%d")
    #     name = pd.Series(slice).name
    #
    #     try:
    #         rn = Analysis(data=slice, startDate=startDate, endDate="2019-07-05")
    #     except (AttributeError) or (IndexError):
    #         rn = Analysis(data=pd.DataFrame(slice), startDate=startDate, endDate="2019-07-05")
    #
    #     # rn = Analysis(data=slice, startDate=startDate, endDate="2019-07-05")
    #     rn.excel_summary()
    #     os.rename(os.path.join(outputDir, "Stock Summary Measures.xlsx"),
    #               os.path.join(outputDir, "Stock Summary Measures_" + name + ".xlsx"))
    #     rn.plot_total_return(data=rn.data, output_dir=outputDir, isLog=True)
    #     rn.plot_total_return(data=rn.data, output_dir=outputDir, isLog=False)