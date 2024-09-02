"""
Created on: 30 June 2019

Module for investigating the stationarity of time series data, specifically focusing
on security data.
Provides summary statistics, log-return analytics, and tests for stationarity using
methods like the Augmented Dickey-Fuller test.

Inspiration from: https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller

from securityAnalysis.utils_finance import calculate_security_returns

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)

# Set plot style
plt.style.use('seaborn')


def test_stationarity_adf(time_series: np.array) -> None:
    """
    Perform the Augmented Dickey-Fuller test for stationarity on a given time series.
    (Wrapper on adfuller method from statsmodels package)

    Parameters:
        time_series (np.ndarray): Time series data to test for stationarity.

    Returns:
        None: Prints the test results, including the test statistic, p-value,
        number of lags, number of observations, and critical values for alpha
        1%, 5%, and 10%.

    NOTE:
        Test statistic: t
        Critical value, c
        Null hypothesis, H_0
        If t < c, reject H_0 --> time series is stationary
        If t > c, fail to reject H_0 --> time series is non-stationary
            (has some drift with time)
    """
    print('Results of Dickey-Fuller Test:')
    adf_result = adfuller(time_series, autolag='AIC')
    df_output = pd.Series(
        adf_result[0:4],
        index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in adf_result[4].items():
        df_output[f'Critical Value ({key})'] = value
    print(df_output)


def get_aug_dickey_fuller_result(time_series: np.array, alpha: int = 5) -> bool:
    """
    Perform the Augmented Dickey-Fuller Test for stationarity on a time series.

    Parameters:
        time_series (np.ndarray): 1-D array of time series data.
        alpha (int): Significance level (1, 5, or 10%).

    Returns:
        bool: True if the series is stationary (rejects the null hypothesis),
        False otherwise.
    """
    assert alpha in [1, 5, 10], "Choose appropriate alpha significance: [1, 5 or 10%]"
    print(f"Performing augmented Dickey Fuller test at significance level alpha: "
          f"{alpha}")

    adf_result = adfuller(time_series, autolag='AIC')

    test_statistic = adf_result[0],
    critical_value = adf_result[4][f'{alpha}%']

    return test_statistic < critical_value


def get_descriptive_stats(data: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Compute descriptive, high level stats (p-values given for two tailed tests),
    incuding skewness and kurtosis, specifying alpha (for tests of skewness and kurtosis)

    Args:
        data (pd.DataFrame): DataFrame with no NaNs containing the data to analyze.
        alpha (float): Significance level for the two-tailed test.
            Must be between 0 and 1.

    Returns
        dict of results for descriptive level statistics
    """
    assert 0 < alpha < 1, f"Alpha level of {alpha} is not valid, must lie between 0 " \
                          f"and 1"
    print("Getting descriptive level stats for dataframe...")

    result_df = pd.DataFrame(
        columns=['Size', 'Mean', 'Std Dev', 'Skewness', 'Excess Kurtosis']
    )

    result_df['Size'] = data.count()
    result_df['Mean'] = data.mean()
    result_df['Std Dev'] = np.std(data)
    result_df['Min'] = np.min(data)
    result_df['Max'] = np.max(data)

    result_df['Skewness'] = skew(data)
    result_df['Skewness t-statistic'] = \
        result_df['Skewness'].values / np.sqrt(6 / result_df['Size'].values)
    result_df['Skewness p-value'] = \
        2 * (1 - stats.t.cdf(result_df['Skewness t-statistic'], df=1))
    # so, one can reject h_0 (skewness of log returns = 0) for a
    # p-value of less than alpha
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
    # real market data
    import yfinance
    import datetime

    price_series_stock = yfinance.download(tickers='GOOGL', start="2010-01-01")['Adj Close'] # google data
    price_df = pd.DataFrame(price_series_stock)

    # random data example
    date_rng = pd.date_range(
        datetime.datetime.now().strftime("%Y-%m-%d"), periods=500
    ).to_list()
    random_returns = pd.Series(np.random.randn(500), index=date_rng)
    price_series_random = random_returns.cumsum()

    # run analysis
    returns_df = calculate_security_returns(
        data=pd.DataFrame(price_series_random),
        is_relative_return=True
    )

    # # could also look at log returns of the data and see if the time series is stationary
    # log_returns_df = calculate_security_returns(data=price_df,
    #                                      is_log_return=True)


    # Test for stationarity
    test_stationarity_adf(time_series=price_series_stockrandom)

    # augmented dickey fuller result
    is_stationary = get_aug_dickey_fuller_result(time_series=price_series_stockrandom)
    print(f"Is the time series stationary?: {is_stationary}")

    # more descriptive statistics on skewness, kurtosis,
    # as well as # observations, max, min, mean,
    # standard deviation etc
    descriptive_stats = get_descriptive_stats(
        data=returns_df,
        alpha=0.05
    )
    print(descriptive_stats)
