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
from scipy.stats import kurtosis, skew, t
from statsmodels.tsa.stattools import adfuller

from securityAnalysis.utils_finance import calculate_security_returns

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
plt.style.use('seaborn')


def test_stationarity_adf(time_series: np.ndarray) -> None:
    """
    Perform the Dickey-Fuller test for stationarity on a time series.

    Args:
        time_series (np.ndarray): A time series array containing non-null values.

    Returns:
        None. Outputs the test results including the Test Statistic, p-value, number of lags used,
        number of observations, and critical values for the 1%, 5%, and 10% levels.
    """
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(time_series, autolag='AIC')
    df_output = pd.Series(df_test[:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output[f'Critical Value ({key})'] = value
    print(df_output)


def get_aug_dickey_fuller_result(time_series: np.ndarray, alpha: int = 5) -> bool:
    """
    Perform the Augmented Dickey Fuller Test for stationarity on a time series.

    Args:
        time_series (np.ndarray): A 1-D array of time series data.
        alpha (int): The level of significance (1, 5, or 10%).

    Returns:
        bool: True if data is stationary; False otherwise.

    Raises:
        ValueError: If alpha is not one of 1, 5, or 10%.
    """
    if alpha not in [1, 5, 10]:
        raise ValueError("Alpha significance level must be one of 1, 5, or 10%.")

    print(f"Performing Augmented Dickey Fuller test at alpha level: {alpha}%")
    df_test = adfuller(time_series, autolag='AIC')
    test_statistic, critical_values = df_test[0], df_test[4]
    is_stationary = test_statistic < critical_values[f"{alpha}%"]
    return is_stationary


def calculate_basic_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic descriptive statistics for a DataFrame.

    Args:
        data: Clean DataFrame with no NaNs

    Returns:
        DataFrame with basic statistics columns.
    """
    stats_df = pd.DataFrame()
    stats_df['Size'] = data.count()
    stats_df['Mean'] = data.mean()
    stats_df['Std Dev'] = np.std(data, ddof=1)
    stats_df['Min'] = np.min(data)
    stats_df['Max'] = np.max(data)
    return stats_df


def test_skewness_kurtosis(data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Test for skewness and kurtosis.

    Args:
        data: Clean DataFrame with no NaNs
        alpha: Level of significance for the two-tailed test.

    Returns:
        DataFrame with skewness and kurtosis test results.
    """
    test_df = pd.DataFrame()
    test_df['Skewness'] = skew(data)
    test_df['Excess Kurtosis'] = kurtosis(data)

    # Calculate t-statistic and p-value for Skewness
    test_df['Skewness t-statistic'] = test_df['Skewness'] / np.sqrt(6 / data.count())
    test_df['Skewness p-value'] = (
            2 * (1 - t.cdf(np.abs(test_df['Skewness t-statistic']), df=data.count() - 1)))
    skew_h0_title = f"Skewness reject H_0 at {100 * alpha}% sig level"
    test_df[skew_h0_title] = test_df['Skewness p-value'] < alpha

    # Calculate t-statistic and p-value for Excess Kurtosis
    test_df['Excess Kurtosis t-statistic'] = test_df['Excess Kurtosis'] / np.sqrt(24 / data.count())
    test_df['Excess Kurtosis p-value'] = (
            2 * (1 - t.cdf(np.abs(test_df['Excess Kurtosis t-statistic']), df=data.count() - 1)))
    kurt_h0_title = f"Kurtosis reject H_0 at {100 * alpha}% sig level"
    test_df[kurt_h0_title] = test_df['Excess Kurtosis p-value'] < alpha

    return test_df


def run_adfuller_test(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run the Augmented Dickey-Fuller test on each column of a DataFrame.

    Args:
        data: Clean DataFrame with no NaNs

    Returns:
        DataFrame with Augmented Dickey-Fuller test results.
    """
    adf_df = pd.DataFrame()
    adf_df['Aug Dickey-Fuller Test'] = [adfuller(column)[1] for _, column in data.iteritems()]
    return adf_df


def get_descriptive_stats(data: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Compute descriptive, high-level stats, including skewness and kurtosis,
    specifying alpha (for tests of skewness and kurtosis).

    Args:
        data: Clean DataFrame with no NaNs.
        alpha: Level of significance for the two-tailed test. Must lie between 0 and 1.

    Returns:
        Dictionary of results for descriptive level statistics.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha level of {alpha} is not valid, must lie between 0 and 1")
    print("Getting descriptive level stats for dataframe...")

    # Basic descriptive statistics
    result_df = calculate_basic_stats(data)

    # Skewness and Kurtosis tests
    test_df = test_skewness_kurtosis(data, alpha)
    result_df = result_df.join(test_df)

    # Augmented Dickey-Fuller tests
    adf_df = run_adfuller_test(data)
    result_df = result_df.join(adf_df)

    # Convert the DataFrame to a dictionary
    result_dict = result_df.T.to_dict()

    return result_dict


if __name__ == '__main__':
    import yfinance as yf
    # Real market data example
    # Fetch historical adjusted closing prices for Alphabet (GOOGL)
    price_series = yf.download(tickers='GOOGL', start="2010-01-01")['Adj Close']
    price_df = pd.DataFrame(price_series)

    # Random data example
    # Create a DataFrame with random returns for demonstration purposes
    date_rng = pd.date_range(start=pd.Timestamp.now().strftime("%Y-%m-%d"), periods=500)
    random_returns = pd.Series(np.random.randn(500), index=date_rng)
    random_price_series = random_returns.cumsum()
    random_price_df = pd.DataFrame(random_price_series)
    price_series = random_price_series
    price_df = random_price_df

    # Calculate returns from price data (assumed to be implemented elsewhere)
    returns_df = calculate_security_returns(data=price_df, is_relative_return=True)

    # (Optional) Calculate log returns and test for stationarity
    # log_returns_df = calculate_security_returns(data=price_df, is_log_return=True)
    # test_stationarity_adf(time_series=log_returns_df)

    # Test for stationarity using the Augmented Dickey-Fuller test for a single time series
    test_stationarity_adf(time_series=price_series)

    # Retrieve the Augmented Dickey-Fuller test result
    get_aug_dickey_fuller_result(time_series=price_series)

    # Get more descriptive statistics including skewness, kurtosis, and other basic stats
    descriptive_stats = get_descriptive_stats(data=returns_df, alpha=0.05)

    # Print the descriptive statistics
    print(descriptive_stats)
