"""
Created on: 30 June 2019

Investigate stationarity of time series (example of security data), analytics on log-returns
Provide summary statistics on the data
Introduce tests (such as Augmented Dickey Fuller) to check stationarity of time series
Inspiration from: https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller

from securityAnalysis.utils_finance import calculate_return_df

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
plt.style.use('seaborn')


def test_stationarity_adf(time_series: np.array) -> None:
    """
    Wrapper on adfuller method from statsmodels package, to perform Dickey-Fuller test for
    Stationarity

    Parameter:
        time_series: time series containing non-null values which to perform stationarity test on

    Returns
        None: Print statement of ['Test Statistic', 'p-value', '# lags', '# observations', and
        critical values for alpha 1, 5 and 10%

    NOTE:
        Test statistic: t
        Critical value, c
        Null hypothesis, H_0
        If t < c, reject H_0 --> time series is stationary
        If t > c, fail to reject H_0 --> time series is non-stationary (has some drift with time)
    """
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(time_series, autolag='AIC')
    df_output = pd.Series(df_test[0:4],
                          index=['Test Statistic', 'p-value', '#Lags Used',
                                 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print(df_output)


def get_aug_dickey_fuller_result(time_series: np.array, alpha: int = 5) -> bool:
    """
    Method to perform Augmented Dickey Fuller Test for stationarity on time_series, at a
    given level of significance alpha

    Parameters:
        time_series: 1-D array of time series data to be tested for stationarity
        alpha: chosen level of significance, must be one of 1,5 or 10%

    Returns:
        bool: True if stationary data (t-statistic less than critical value at significance level
        alpha, reject H_0), False for non-stationary data
    """
    assert alpha in [1, 5, 10], "Choose appropriate alpha significance: [1, 5 or 10%]"
    print(f"Performing augmented Dickey Fuller test at significance level alpha: {alpha}")

    df_test = adfuller(time_series, autolag='AIC')
    test_stats = {
        'test_statistic': df_test[0],
        'p-values': df_test[4]
    }
    is_stationary = test_stats['test_statistic'] < test_stats['p-values'][f"{str(alpha)}%"]

    return is_stationary


def get_descriptive_stats(data: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Compute descriptive, high level stats (p-values given for two tailed tests),
    incuding skewness and kurtosis, specifying alpha (for tests of skewness and kurtosis)

    Args:
        data: Clean dataframe with no NaNs
        alpha: level of significance for the two-tailed test. must lie between 0 and 1

    Returns
        dict of results for descriptive level statistics
    """
    assert 0 < alpha < 1, f"Alpha level of {alpha} is not valid, must lie between 0 and 1"
    print("Getting descriptive level stats for dataframe...")

    result_df = pd.DataFrame(columns=['Size', 'Mean', 'Std Dev', 'Skewness', 'Excess Kurtosis'])

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

    result_df['Excess Kurtosis'] = kurtosis(data)  # if high excess kurtosis --> thick tails
    result_df['Excess Kurtosis t-statistic'] = \
        result_df['Excess Kurtosis'].values / np.sqrt(24 / result_df['Size'].values)
    result_df['Excess Kurtosis p-value'] = \
        2 * (1 - stats.t.cdf(result_df['Excess Kurtosis t-statistic'], df=1))
    kurt_h0_title = f"Kurtosis reject H_0 at {str(100 * alpha)}% sig level"
    kurt_h0_values = result_df['Excess Kurtosis p-value'].values < alpha
    result_df['Excess Kurtosis accept H_0'] = kurt_h0_values
    result_df.rename(columns={'Excess Kurtosis accept H_0': kurt_h0_title}, inplace=True)

    adf_results = []
    for i in data.columns:
        adf_results.append(get_aug_dickey_fuller_result(data.loc[:, i]))

    result_df['Aug Dickey-Fuller Test'] = adf_results
    result_dict = result_df.T.to_dict()

    return result_dict


if __name__ == '__main__':

    # examples of running the above methods

    # ----------------
    # real market data
    # ----------------
    import yfinance
    price_series = yfinance.download(tickers='GOOGL', start="2010-01-01")['Adj Close'] # google data
    price_df = pd.DataFrame(price_series)

    # -------------------
    # random data example
    # -------------------
    import datetime
    date_rng = pd.date_range(datetime.datetime.now().strftime("%Y-%m-%d"), periods=500).to_list()
    random_returns = pd.Series(np.random.randn(500), index=date_rng)
    price_series = random_returns.cumsum()
    price_df = pd.DataFrame(price_series)

    # run analysis
    returns_df = calculate_return_df(data=price_df,
                                     is_relative_return=True)

    # # could also look at log returns of the data and see if the time series is stationary
    # log_returns_df = calculate_return_df(data=price_df,
    #                                      is_log_return=True)


    # test for stationarity (using Augmented Dickey Fuller test) for one timeseries
    test_stationarity_adf(time_series=price_series)

    # augmented dickey fuller result
    get_aug_dickey_fuller_result(time_series=price_series)

    # more descriptive statistics on skewness, kurtosis, as well as # observations, max, min, mean,
    # standard deviation etc
    get_descriptive_stats(data=returns_df,
                          alpha=0.05)
