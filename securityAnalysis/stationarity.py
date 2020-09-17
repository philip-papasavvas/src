"""
Created on 30 June 2019

Look at the stationarity of stock time series, carry out analysis on the log-returns of a stock.
Provide summary statistics on the data
Introduce tests (such as Augmented Dickey Fuller) to check stationarity of time series
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller

import utils_date

pd.set_option('display.max_columns', 5)
plt.style.use('seaborn')


def log_daily_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Give log daily returns"""
    log_daily_return = data.apply(lambda x: np.log(x) - np.log(x.shift(1)))[1:]
    return log_daily_return


def drop_null_columns(data: pd.DataFrame):
    """Drop columns from the dataframe with null values"""
    original_columns = list(data.columns)
    cleaned_data = data.dropna(axis=1)
    new_columns = list(cleaned_data.columns)
    cut_columns = [x for x in original_columns if x not in new_columns]

    print(f"The following columns have been dropped from the dataframe as they contain NaNs: "
          f"\n {cut_columns}")
    return cleaned_data


def dickey_fuller_test(time_series):
    """
    Perform Dickey-Fuller test:
    If the test statistic is less than the critical value then reject H_0, and the times series
    is stationary.
    If test statistic greater than critcal value, fail to reject H_0, and time series is
    non-stationary/it has time dependence
    """
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(time_series, autolag='AIC')
    df_output = pd.Series(df_test[0:4],
                          index=['Test Statistic', 'p-value', '#Lags Used',
                                 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print(df_output)


def adf_test_result(timeseries, sig=5):
    # Perform Dickey-Fuller test and give True/False for stationarity of time series.
    # If True (test statistic less than critical value at 5%), reject H_0, and time series
    # stationarity
    # Offered at 1, 5 or 10% significance level
    dftest = adfuller(timeseries, autolag='AIC')
    assert (sig == 1) or (sig == 5) or (sig == 10), "Choose a significance of 1, 5 or 10 pc"
    key = str(sig) + "%"
    is_stationary = dftest[0] < dftest[4][key]
    return is_stationary


# create high level summary stats for the log returnss
def get_descriptive_stats(data: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Compute descriptive stats (p-values given for two tailed tests)

    Args:
        data: Clean dataframe with no NaNs
        alpha: level of significance for the two-tailed test. must lie between 0 and 1

    Returns
        pd.DataFrame
    """
    result_df = pd.DataFrame(columns=['Size', 'Mean', 'Std Dev', 'Skewness', 'Excess Kurtosis'])
    # , 'K Min', 'K Max'])
    result_df['Size'] = data.count()
    result_df['Mean'] = data.mean()
    result_df['Std Dev'] = np.std(data)
    result_df['Min'] = np.min(data)
    result_df['Max'] = np.max(data)
    result_df['Skewness'] = skew(data)
    result_df['Skewness t-statistic'] = \
        result_df['Skewness'].values / np.sqrt(6 / result_df['Size'].values)
    result_df['Skewness p-value'] = 2 * (1 - stats.t.cdf(result_df['Skewness t-statistic'], df=1))
    # so, one can reject h_0 (skewness of log returns = 0) for a p-value of less than alpha
    skew_h0_title = "Skewness reject H_0 at " + str(100 * alpha) + "% sig level"
    skew_h0_values = result_df['Skewness p-value'].values < alpha
    result_df['Skewness accept H_0'] = skew_h0_values
    result_df.rename(columns={'Skewness accept H_0': skew_h0_title}, inplace=True)

    result_df['Excess Kurtosis'] = kurtosis(data)  # if high excess kurtosis it means heavy tails
    result_df['Excess Kurtosis t-statistic'] = \
        result_df['Excess Kurtosis'].values / np.sqrt(24 / result_df['Size'].values)
    result_df['Excess Kurtosis p-value'] = \
        2 * (1 - stats.t.cdf(result_df['Excess Kurtosis t-statistic'], df=1))
    kurt_h0_title = "Kurtosis reject H_0 at " + str(100 * alpha) + "% sig level"
    kurt_h0_values = result_df['Excess Kurtosis p-value'].values < alpha
    result_df['Excess Kurtosis accept H_0'] = kurt_h0_values
    result_df.rename(columns={'Excess Kurtosis accept H_0': kurt_h0_title}, inplace=True)

    res = []
    for i in data.columns:
        res.append(adf_test_result(data.loc[:, i]))

    result_df['Aug Dickey-Fuller Test'] = res

    return result_df


if __name__ == '__main__':

    # ----------
    # parameters
    # ----------
    wk_dir = "C://Users//Philip//Documents//python//"
    input_folder, output_folder = os.path.join(wk_dir, "input"), os.path.join(wk_dir, "output")
    # "example_data.csv", "example_data_na.csv" has NA rows
    # df = pd.read_csv(INPUT_FOLDER + 'example_data.csv') #, parse_dates=True)

    # ---------
    # data read & clean
    # ---------
    df = pd.read_csv(input_folder + "funds_stocks_2019.csv")
    df = utils_date.char_to_date(df)
    df.set_index('Date', inplace=True)
    df_returns = log_daily_returns(data=df)
    clean_df_returns = drop_null_columns(df_returns)

    # deal with returns not time series of prices, as prices are non-stationary transform the time
    # series so that it becomes stationary. If the non-stationary is a random walk with or without
    # drift, it is transformed to a stationary process by differencing - it is now a stationary
    # stochastic (random probability distribution) process if time series data also exhibits a
    # deterministic trend, spurious results can be avoided by detrending if non-stationary time
    # series are both stochastic and deterministic at the same time, differencing and detrending
    # should be applied - differencing removes the trend in variance and detrending removes
    # determinstic trend

    # Example: adf_test_result(clean_df_returns.iloc[:,0])

    # test the hypothesis that the returns are normally distibuted by looking at skewness and kurtosis
    # under the normality assumption, s(x) and k(x) - 3 are distributed asymptotically as normal
    # with zero mean and variances of 6/T and 24/T respectively.
    # so in this case we have the (log) return series of the asset {r_1, ..., r_n}, and we want
    # to test the skewness
    # of the returns, so consider the null hypothesis H_0: s(r) = 0 with the
    # alternate hypthesis s(r) != 0.
    # the t ratio statistic of the sample skewness is t = (s(hat)(r) / sqrt(6/T))
    # where you would reject null hypothesis at alpha sig level if |t| > z_(alpha/2) where
    # z_(alpha/2) is the upper 100(alpha/2)th quantile of a standard normal distribution.
    # or you could compute the p value of the test statistic t and reject H_0 iff p-value < alpha

    # can also test the excess kurtosis of the return series using hypotheses H_0: k(r) - 3 = 0,
    # versus alternate test statistic is then t = k(r) - 3 / sqrt(24/T)

    # look at symmetry of return distribution

    # look at the Augmented Dickey Fuller test for unit roots. The null hypothesis, H_0, is that
    # the time series can be represented by a unit root, therefore it is non-stationary (it
    # has a time dependent structure). The alternate hypothesis is that the time series
    # is stationary
    # If you fail to reject null hypothesis, then time series has a unit root, so it is non-stationary,
    # it has some time dependency
    # If null hypothesis is rejected, time series doesn't have a unit root, time series is stationary

    # So a p-value below below a certain threshold (alpha, usually 1 or 5%)
    # suggests we reject null hypothesis (so time series is stationary),
    # whilst a p-value above the threshold suggests we fail to reject null hypothesis H_0

    # ADF pulled from https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/


    # Use a simple Augmented Dickey Fuller test to test stationarity of time series:
    get_descriptive_stats(data=clean_df_returns, alpha=0.05)

    # Look at the price time series first:
    df = pd.read_csv(input_folder + "funds_stocks_2019.csv")
    df = utils_date.char_to_date(df)  # convert all dates to np datetime64
    df.set_index('Date', inplace=True)

    # Drop the null columns - but do not take log returns
    clean_df = drop_null_columns(df)

    result_prices = get_descriptive_stats(data=clean_df, alpha=0.05)
    # are any of the time series stationary?
    any(result_prices['Aug Dickey-Fuller Test'])  # False
    # no, they are all time dependent

    # Example: adf_test(clean_df_returns.iloc[:,0])

    # Now look at log returns
    clean_df_returns = log_daily_returns(df)

    clean_df_log_rets = drop_null_columns(clean_df_returns)

    result_returns = get_descriptive_stats(data=clean_df_log_rets, alpha=0.05)
    # are any of the time series stationary?
    all(result_returns['Aug Dickey-Fuller Test'])  # True, so they are stationary returns
