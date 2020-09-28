"""
Created by: Philip.P
Created on: 16 Mar 2019

Analyse financial security (mainly price) data (sourced from Bloomberg or YahooFinance (via API)).
Can specify ticker:fund name json mapping in config

TODO:
 - Refactor the init in Analysis class to load in the data, fewer setting of start and end dates
 - Charts for lookback performance, limiting the number if more than 5 funds
 - Plotting capabilities for stock charts
"""

import datetime as dt
import json
import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker as mtick

from securityAnalysis.utils_finance import (return_info_ratio, return_sharpe_ratio,
                                            return_sortino_ratio,
                                            calculate_annualised_return_from_df,
                                            calculate_annual_volatility)
from utils_date import char_to_date, datetime_to_str

plt.style.use('ggplot')
plt.tight_layout()
plt.close()
pd.set_option('display.max_columns', 5)


class Analysis:
    """Analysis of historical security data, providing analytics on performance- will remove nan

    Args:
        data: Security price data over time period (assumed daily)
        is_bloomberg: If not bloomber, assume it is from YahooFinance via the API (yfinance module)
        input_directory: Working directory

    Attributes:
        run_date: YYYMMDD
        input_directory: Working directory
        output_dir: Output directory
        data: Data to be analysed
    """

    def __init__(self, data: pd.DataFrame, input_directory: str, is_bloomberg: bool) -> None:
        self.run_date = datetime_to_str(dt.datetime.now())
        self.wkdir = input_directory
        self.set_output_folder()
        self.output_dir = None

        if is_bloomberg:
            input_data = char_to_date(data)

            self.start_date = datetime_to_str(input_data.index.min())
            self.end_date = datetime_to_str(input_data.index.max())
            print(f"Data for period runs {self.start_date} to {self.end_date}")

            self.data = input_data
        elif is_bloomberg:
            print("Data likely downloaded from Yahoo Finance?")
            pass

    def clean_slice_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean/slice (or both) data by removing NAs from the analysis

        Args:
            input_data: Security data in wide, short format

        Returns:
            pd.DataFrame
        """
        # check for NaN values and drop, alerting user for what has been dropped
        na_secs_names = input_data.columns[input_data.isnull().any()].values
        if len(na_secs_names) > 0:
            print("The following securities have NaNs in the dataset and will"
                  "be included in the analysis: \n {}".format(na_secs_names))

        clean_data = input_data.dropna(axis=0, inplace=True)

        print(f"Data analysed for period {self.start_date} to {self.end_date}")

        return clean_data

    def set_output_folder(self):
        """Set output folder according to working directory specified"""
        output_path = os.path.join(self.wkdir, "output", self.run_date)
        self.output_dir = output_path
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        print(f"Output folder created: {self.output_dir}")

    def annual_return_table(self, data: pd.DataFrame, save_results: bool = True) -> pd.DataFrame:
        """Basic summary table of annual returns of stocks

        Args:
            data
            save_results: Saves in the self.output_dir

        Returns:
            pd.DataFrame
        """

        input_data = data.copy(True)
        total_annual_rtn = (input_data.groupby(input_data.index.year).last() / input_data.groupby(
            input_data.index.year).first())

        annual_rtn = (total_annual_rtn - 1).T
        annual_rtn.index.name = "Security / Annual Return"

        if save_results:
            annual_rtn.to_csv(f"{self.output_dir}/{self.run_date}_sec_annual_return.csv")
            print(f"Annual returns table saved as csv in {self.output_dir}")
        return annual_rtn

    def performance_summary(self, data: pd.DataFrame, risk_free_rate: float = 0,
                            target_return_rate: float = 0,
                            to_calculate_risk_measures: bool = True,
                            save_results: bool = False) -> pd.DataFrame:
        """
        Summarises return and volatility for input data over whole period

        Args:
            data: Data to analyse performance of
            risk_free_rate: Annual risk free rate
            target_return_rate: Target rate of (daily) period return
            to_calculate_risk_measures: Include Sharpe and Sortino Ratios in the summary table
            save_results: If True by default will save in self.output_dir

        Returns:
            summary: table of returns and volatility of securities entered
        """

        data_to_clean = data.copy(True)

        annual_rtn = calculate_annualised_return_from_df(data=data_to_clean)
        annual_vol = calculate_annual_volatility(data=data_to_clean)
        info_ratio = return_info_ratio(data=data_to_clean)

        cols = ['Annual Return', 'Annual Volatility', 'Info Ratio']
        summary = pd.concat([annual_rtn, annual_vol, info_ratio], axis=1)
        summary.columns = cols

        if to_calculate_risk_measures:
            if risk_free_rate is None:
                risk_free_rate = 0

            sharpe = return_sharpe_ratio(data=data,
                                         risk_free=risk_free_rate)
            sortino = return_sortino_ratio(data=data,
                                           target_return=target_return_rate,
                                           risk_free=risk_free_rate)

            cols += ['Sharpe Ratio', 'Sortino Ratio']
            summary = pd.concat([summary, sharpe, sortino], axis=1)
            summary.columns = cols

        summary.dropna(inplace=True)
        summary.index.name = "Fund/Stock"

        log = " ".join(["Fund Stats for", self.start_date, "to", self.end_date])
        errors = data_to_clean.columns.difference(summary.index).values.tolist()

        print(log)
        if len(errors) > 0:
            print(f"There were errors in the dataset for the following funds: \n {errors}")

        if save_results:
            file_name = "_".join(["securities_summary", self.start_date, self.end_date, ".csv"])
            summary.to_csv(os.path.join(f"{self.output_dir}/{file_name}"))
            print(f"Summary table has been written to csv file in directory: {self.output_dir}")

        return summary

    # TODO: rewrite as this isn't good in terms of the timedelta
    def calculate_lookback_performance(self, end_date: np.datetime64 = None,
                                       lookback_periods: List[str] = None,
                                       to_save_results=False) -> pd.DataFrame:
        """Analyse performance of certain funds over a custom lookback period (list)

        Args:
            end_date: If not specified, defaults to last valid date in dataset
            lookback_periods: Specify a list of lookback periods to compare stock prices to
            to_save_results: If True, will save in output_directory

        """
        if lookback_periods is None:
            lookback_periods = ["0D", "6M", "1Y", "2Y", "3Y"]

        security_data = self.data

        if end_date is None:
            end_date = self.end_date

        if lookback_periods is None:
            lookback_periods = ["0D", "3M", "6M", "9M", "12M", "18M", "24M"]

        # TODO: if a date in the lookback is not in the range of the dataset then we drop this date
        # target_dates = [return_date_diff(df, end_date, i) for i in lookback_periods]
        target_dates = []
        target_prices = [security_data.loc[i, :].values for i in target_dates]

        # iloc[::-1] is to reverse the dataframe by the date index --> earliest to latest
        lookback_table = pd.DataFrame.from_records(target_prices,
                                                   index=target_dates,
                                                   columns=security_data.columns)
        lookback_table.sort_index(ascending=True, inplace=True)

        # Period return
        cumulative_return = lookback_table.apply(lambda x: x / x[0])
        cumulative_return['Return Period'] = lookback_periods
        cumulative_return = cumulative_return[cumulative_return.columns.tolist()[-1:] +
                                              cumulative_return.columns.tolist()[:-1]]

        if to_save_results:
            file_name = f"{self.start_date}_{self.end_date}_"
            writer = pd.ExcelWriter(self.output_dir + file_name + "Security Performance.xlsx")

            lookback_table.index = lookback_table.index.values.astype("datetime64[D]")
            lookback_table_print = lookback_table.T
            lookback_table_print.to_excel(writer, "Prices")

            cumulative_return.index = cumulative_return.index.values.astype("datetime64[D]")
            cumulative_return.T.to_excel(writer, "Return")

            writer.save()
            print(f"Lookback performance table has been written to directory: {self.output_dir}")

        return lookback_table

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
    def return_bollinger_band(data: pd.DataFrame,
                              window: int,
                              std_devs: int
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Function to return bollinger bands for securities

        Args:
            data: Dataframe of stock prices with index as np.datetime64
            window: Rolling window (in days) for mean price and standard deviation
            std_devs: Number of standard deviations for bollinger band, usually an integer

        Returns:
            pd.DataFrame: roll_mean
            pd.DataFrame: roll_std
            pd.DataFrame: boll_high
            pd.DataFrame: boll_low
        """

        roll_mean = data.rolling(window).mean()
        roll_std = data.rolling(window).std()

        boll_high = roll_mean + (roll_std * std_devs)
        boll_low = roll_mean - (roll_std * std_devs)

        return roll_mean, roll_std, boll_high, boll_low

    def plot_bollinger_bands(self, data: pd.DataFrame, rolling_window: int = 20,
                             num_st_devs: int = 2) -> None:
        """Bollinger band plots for each of the stocks in the dataframe"""

        for col in data.columns:
            slc = data.loc[:, col]
            normed_px = slc / slc[0]

            # Info for bollinger plots, also useful elsewhere
            roll_mn, roll_std, boll_high, boll_low = \
                Analysis.return_bollinger_band(data=slc,
                                               window=rolling_window,
                                               std_devs=num_st_devs)

            # Plot the charts
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")
            ax1.plot(roll_mn, color=color)
            # ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(boll_high, linestyle="dashed", color="k", linewidth=0.5)
            ax1.plot(boll_low, linestyle="dashed", color="k", linewidth=0.5)
            # set there to be N=6 lines on y-axis
            ax1.yaxis.set_major_locator(mtick.LinearLocator(6))

            norm_std_rolling = normed_px.rolling(window=rolling_window).std()
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Rolling Volatility')
            ax2.plot(norm_std_rolling, color=color)
            # ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 0.25)
            ax2.yaxis.set_major_locator(mtick.LinearLocator(6))

            plt.suptitle(col + " (rolling {n}-day window)".format(n=rolling_window))
            # fig.tight_layout()
            plt.show()
            plt.savefig(f"{self.output_dir}/{col} Price & Vol History.png")
            plt.close()

    @staticmethod
    def plot_total_return(input_data: pd.DataFrame,
                          output_dir: str,
                          log_returns=False) -> None:
        """Plot the normalised return over time, anchored back to start of lookback period"""

        for col in input_data.columns:
            slce = input_data.loc[:, col]

            if log_returns:
                normed_px = 1 + np.log(slce / slce[0])
            else:
                normed_px = slce / slce[0]

            # Plot the charts
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("Time")
            ax1.plot(normed_px, color=color)

            if log_returns:
                ax1.set_ylabel("Log Total Return")
                plt.suptitle(col + " - Log Total Return")
            else:
                ax1.set_ylabel("Total Return")
                plt.suptitle(col + " - Total Return")

            plt.show()
            if log_returns:
                plt.savefig(f"{output_dir}/{col} - Log Total Return Chart.png")
            else:
                plt.savefig(f"{output_dir}/{col} - Total Return Chart.png")
            plt.close()

    # def csv_summary(self, output_dir: str) -> None:
    #     """Print the summary to csv file"""
    #
    #     if output_dir is None:
    #         output_dir = self.output_dir
    #
    #     writer = pd.ExcelWriter(os.path.join(output_dir, "Stock Summary Measures.xlsx"))
    #     summary_one = self.summaryTable(toCsv=False, r=0.01)
    # Summary table- return/volatility/info & sharpe ratio
    #
    #     summary_one[['Annualised Return', 'Annual Volatility']] *= 100
    #     summary_one[['Annualised Return', 'Annual Volatility']] = \
    #         summary_one[['Annualised Return', 'Annual Volatility']].applymap("{0:.2f}%".format)
    #     summary_one[['Information Ratio', 'Sharpe Ratio']] = summary_one[
    #         ['Information Ratio', 'Sharpe Ratio']].applymap("{0:.4}".format)
    #
    #     summary_one.to_excel(writer, "Summary Table")
    #
    #     annual_table = self.annualReturns(toCsv=False)
    #     annual_table.columns = annual_table.columns.astype(str)
    #
    #     print_annual_table = annual_table * 100
    #     print_annual_table = print_annual_table.applymap("{0:.2f}%".format)
    #     print_annual_table.to_excel(writer, "Annual Return")
    #
    #     correlation_mat = self.data.corr()  # correlation matrix
    #     correlation_mat.to_excel(writer, "Correlation")
    #
    #     writer.save()
    #     print("Summary statistics produced, and in the following directory: " + self.output_dir)


if __name__ == "main":
    from src.get_paths import get_config_path

    wk_dir = "/Users/philip_p/Documents/python/"
    input_folder = os.path.join(wk_dir, "data/finance")
    output_folder = os.path.join(wk_dir, "output")
    df = pd.read_csv(f"{input_folder}/funds_stocks_2019.csv")

    with open(get_config_path("bbg_ticker.json")) as f:
        ticker_map_dict = json.load(f)

    rn = Analysis(data=df, input_directory=wk_dir, is_bloomberg=True)
    clean_df = rn.clean_slice_data(input_data=df)
    results = rn.performance_summary(data=clean_df, save_results=True)
    # rn.plot_bollinger_bands(data=df, window=60)

    rn = Analysis(data=df, input_directory=input_folder, is_bloomberg=True)
    # rn.plot_bollinger_bands(data = df[df.index > "2014-01-01"])

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
