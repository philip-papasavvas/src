"""
Author: Philip.P
Date created: 16/03/19

Analyse financial security (mainly price) data.
Input securities data either sourced from Bloomberg or YahooFinance via the API.
Mapping between ticker and fund name can be specified

To do:
 - Refactor the init in Analysis class to load in the data, fewer setting of start and end dates

 - Charts for lookback performance, limiting the number if more than 5 funds
 - Produce yearly (& maybe monthly) returns
 - Plotting capabilities for stock charts
"""

# usual suspects
import os
import pandas as pd
import numpy as np
import datetime as dt
from re import sub, search
import utils

import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
plt.style.use('ggplot')
plt.tight_layout()
plt.close()

pd.set_option('display.max_columns', 5)
dateToStr = lambda d: d.astype(str).replace('-', '')
extract_str_timestamp = lambda d: dt.datetime.strftime(d, "%Y%m%d")

class Analysis():
    """
    Analysis of historical security data, providing analytics on performance- will remove NANs
    Methods for analysing returns and volatilities etc.

    Params:
        data: dataframe
            Financial data with the (date) index labelled 'Date'

        start_date & end_date: str "YYYY-MM-DD" or "YYYYMMDD", default None
            Lookback periods for analysing financial data

        ticker_mapping: dataframe, default None
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
    def __init__(self, data, wkdir, data_src, start_date=None, end_date=None, ticker_mapping=None, drop_na=False):
        """
        Params
        ------
        data: dataframe
            Security data of price over time period, with first column as date, other columns as individual stock
            close prices
        data_src: str, acceptable sources 'bbg', 'yfin'
            Source of financial data, whether from Bloomberg (bbg) or YahooFinance via the API (yfin).
        wkdir: str
        start_date, end_date: str, default None
            Specify slicing of dataframe on which to do analysis (defaults to end points of dataframe). format "YYYY-MM-DD"
        drop_na: bool, default False
            Drop securities with any NA values
        ticker_dict: dict
            Mapping between the Bloomberg ticket and (custom) name
        """
        self.run_date = extract_str_timestamp(dt.datetime.now())
        self.wkdir = wkdir
        self.set_output_folder()
        print("Output folder: {}".format(self.output_dir))

        if data_src == 'bbg':
            dataframe = utils.char_to_date(data)

            self.start_date = extract_str_timestamp(dataframe.index.min())
            self.end_date = extract_str_timestamp(dataframe.index.max())
            print("Data for period runs {start} to {end}".format(start=self.start_date, end=self.end_date))

            self.data = dataframe
        elif data_src == 'yfin':
            pass

    def clean_slice_data(self, df, drop_na=True, start=None, end=None):
        """
        Method to clean/slice (or both) data by removing NAs from the analysis, and restricting the time frame

        Params
        ------
            df: dataframe of security data in wide, short format
            drop_na: bool, default True
            start, end: str, default None
                Start and end dates for analysing the data, if not defined will default to beginning and end
                of dates in df

        """
        # check for NaN values and drop, alerting user for what has been dropped
        na_secs_names = df.columns[df.isnull().any()].values
        if len(na_secs_names) > 0:
            print("The following securities have NaNs in the dataset and will"
                  "be included in the analysis: \n {}".format(na_secs_names))

        if drop_na:
            df.dropna(axis=0, inplace=True)

        try:
            clean_df = df.loc[self.start_date: self.end_date, :]
        except KeyError:
            clean_df = df

        print("Data analysed for period {start} to {end}".format(start=self.start_date, end=self.end_date))

        return clean_df

    def set_output_folder(self):
        """Set output """
        output_path = os.path.join(self.wkdir, "output", self.run_date)
        self.output_dir = output_path
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

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

    @staticmethod
    def calculate_sortino_ratio(input_data, target, rfr, return_period=1):
        """
        Method to calculate Sortino Ratio (gives a better measure of downside
        volatility and therefore risk). Unlike the Sharpe Ratio it does not
        penalise upside volatility.

        Params
        ------
            input_data: df
                Original dataframe of input data
            target: float
                Target return (for the return period)
            rfr: float, default 1
                Risk free rate
            return_period: float, default 1
                Specify the return period (number of days) for the ratio.

        Returns
        -------
            sortino: ndarray
        """

        prd_return = input_data.pct_change(return_period).iloc[1:, ]
        downside_return = np.array(prd_return.values - target)
        # subtract_array = np.ones(downside_return.shape[1])*target

        inner_bit = np.minimum(np.zeros(shape=downside_return.shape[1]), downside_return)

        tdd_sum = np.sum(np.square(inner_bit), axis=0)
        target_downside_dev = np.sqrt(tdd_sum / len(prd_return))

        sortino = (prd_return.mean() - rfr) / target_downside_dev

        return sortino

    def annual_return_table(self, data, save_results=True):
        """
        Basic summary table of annual returns of stocks

        Params
        ------
            data: df
            save_results: bool, default True

        Returns
        -------
            annual_rtn: df
                Annual returns dataframe
        """

        df = data.copy(True)
        total_annual_rtn = (df.groupby(df.index.year).last()/ df.groupby(df.index.year).first())

        annual_rtn = (total_annual_rtn - 1).T
        annual_rtn.index.name = "Security / Annual Return"

        if save_results:
            annual_rtn.to_csv(os.path.join(self.output_dir, self.run_date + "_sec_annual_return.csv"))
            print("Annual returns table saved as csv in {dir}".format(dir=self.output_dir))

        return annual_rtn

    def performance_summary(self, data, rfr=0, target=0, risk_measures=True, save_results=False):
        """
        Summarises return and volatility for input data over whole period

        Params:
            toCsv:  bool, default False. If True, written into output_dir
            rfr: float, default 0
                Risk Free Rate of annual return, default 0%
            target: float, default 0
                Target rate of (daily) period return, default 0%
            risk_measures: bool, default True
                Whether to include Sharpe and Sortino Ratios in the summary table
            save_results: bool, default False

        Returns:
            summary: table of returns and volatility of securities entered
        """

        df = data.copy(True)

        daily_rtn = self.daily_returns(df)
        annual_rtn = np.mean(daily_rtn) * 252
        annual_vol = np.std(daily_rtn) * np.sqrt(252)
        info_ratio = np.divide(annual_rtn, annual_vol)

        cols = ['Annual Return', 'Annual Volatility', 'Info Ratio']
        summary = pd.concat([annual_rtn, annual_vol, info_ratio], axis=1)
        summary.columns = cols

        if risk_measures:
            if rfr is None:
                rfr = 0

            sharpe = np.divide(annual_rtn-rfr, annual_vol)
            sortino = self.calculate_sortino_ratio(input_data=df, target=target, rfr=rfr)

            cols += ['Sharpe Ratio', 'Sortino Ratio']
            summary = pd.concat([summary, sharpe, sortino], axis=1)
            summary.columns = cols

        summary.dropna(inplace=True)
        summary.index.name = "Fund/Stock"

        log = " ".join(["Fund Stats for", self.start_date, "to", self.end_date])
        errors = df.columns.difference(summary.index).values.tolist()

        print(log)
        if len(errors) > 0:
            print("The following funds were not analysed due to errors in the dataset: \n {}".format(errors))

        if save_results:
            summary.to_csv(os.path.join(self.output_dir, \
                                        "_".join(["securities_summary", self.start_date, self.end_date, ".csv"])))
            print("Summary table has been written to csv file in directory: {}".format(self.output_dir))

        return summary

    def lookbackPerformance(self, end_date = None, lookbackList = ["0D", "6M", "1Y", "2Y", "3Y"],
                            results = False, returnPlot = False):
        """
        Analyse performance of certain funds over a custom lookback period

        Params:
            end_date:        type np.datetime64
                Defaults to last valid date in dataset
            lookbackList:   type list.
                default ["0D", "6M", "1Y", "2Y", "3Y"]
        """
        df = self.data

        if end_date is None:
            end_date = self.end_date

        if lookbackList is None:
            lookbackList = ["0D", "3M", "6M", "9M", "12M", "18M", "24M"]

        #TODO: if a date in the lookback is not in the range of the dataset then we drop this date
        target_dates = [utils.previousDate(df, end_date, i) for i in lookbackList]
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
            fileName = dateToStr(self.start_date) + "_" + dateToStr(self.end_date) + "_"
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
    #     if isinstance(self.start_date, np.datetime64) & isinstance(self.end_date, np.datetime64):
    #         df = self.data.loc[self.start_date: self.end_date, :]
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

    from securities_analysis import Analysis

    wkdir = "/Users/philip_p/Documents/python/"
    input_folder = os.path.join(wkdir, "data/finance")
    output_folder = os.path.join(wkdir, "output")
    # df = utils.prep_fund_data(df_path= os.path.join(input_folder, "example_data_na.csv"))
    df = utils.prep_fund_data(df_path=os.path.join(input_folder, "funds_stocks_2019.csv"))

    import json
    from utils import get_config_path

    with open(get_config_path("bbg_ticker.json")) as f:
        ticker_map_dict = json.load(f)

    data = df
    wkdir = wkdir
    data_src = 'bbg'

    rn = Analysis(data=df, wkdir= wkdir, data_src='bbg')
    clean_df = rn.clean_slice_data(df=df)
    results = rn.performance_summary(data=clean_df, save_results=True)
    # rn.csv_summary(outputDir=os.path.join(wkdir, "output"))
    # rn.plot_bollinger_bands(data=df, window=60)

    # tick_mapping = pd.read_csv(inputDir + 'tickerNameMapping.csv') #also:"tickerNameMapping.csv", 'securityMapping_subset.csv'

    # rn = Analysis(data = df, start_date = "2014-01-01", end_date = "2019-06-01", ticker_mapping = None)
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
    #     start_date = slice.index[0].strftime("%Y-%m-%d")
    #     name = pd.Series(slice).name
    #
    #     try:
    #         rn = Analysis(data=slice, start_date=start_date, end_date="2019-07-05")
    #     except (AttributeError) or (IndexError):
    #         rn = Analysis(data=pd.DataFrame(slice), start_date=start_date, end_date="2019-07-05")
    #
    #     # rn = Analysis(data=slice, start_date=start_date, end_date="2019-07-05")
    #     rn.excel_summary()
    #     os.rename(os.path.join(outputDir, "Stock Summary Measures.xlsx"),
    #               os.path.join(outputDir, "Stock Summary Measures_" + name + ".xlsx"))
    #     rn.plot_total_return(data=rn.data, output_dir=outputDir, isLog=True)
    #     rn.plot_total_return(data=rn.data, output_dir=outputDir, isLog=False)