"""
Created: 17 June 2020
Utils specific for financial security data
"""
import numpy as np
import pandas as pd

from decorators import deprecated
from utils_date import excel_date_to_np


# array methods
def calculate_relative_return_arr(a: np.array) -> np.array:
    """Calculate relative return of an array"""
    return a[1:] / a[:-1]


# dataframe methods
def calculate_return_df(data: pd.DataFrame,
                        is_relative_return: bool = True,
                        is_log_return: bool = False,
                        is_absolute_return: bool = False):
    """Method to calculate different types of return from dataframe

    Parameters
        df: Dataframe containing columns as securities, with all columns as float
        is_relative_return
        is_log_return
        is_absolute_return

    Returns
        pd.DataFrame:  a dataframe with returns shifted as instructed
    """
    data = data.select_dtypes(exclude=['string', 'object'])

    if is_log_return:
        print("log returns")
        return_df = data.apply(lambda x: np.log(x / x.shift(1)))[1:]
    elif is_relative_return:
        print("relative_returns")
        return_df = data.apply(lambda x: (x / x.shift(1)) - 1)[1:]
    elif is_absolute_return:
        print("absolute_returns")
        return_df = data.apply(lambda x: x - x.shift(1))[1:]
    else:
        print("not a valid return type")

    return return_df


def calculate_annualised_return_df(data: pd.DataFrame) -> pd.Series:
    """
    Calculate annualised return (assuming input data is daily).
    For example, see unit test: test_utils_finance

    Parameters:
        data: Input dataframe with numeric columns filtered for analysis

    Returns:
        pd.Series: Annualised return for input_df, labels are input columns
    """
    daily_rtn = calculate_return_df(data=data, is_relative_return=True)
    ann_rtn = np.mean(daily_rtn) * 252  # num business days in a year
    return ann_rtn


def calculate_annual_volatility_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annualised return (assuming input data is daily).
    For example, see unit test: test_utils_finance

    Parameters:
        data: Input dataframe with numeric columns filtered for analysis

    Returns:
        pd.Series: Annualised volatility for input_df, labels are input columns
    """
    daily_rtn = calculate_return_df(data=data, is_relative_return=True)
    ann_vol = np.std(daily_rtn) * np.sqrt(252)  # num business days in a year
    return ann_vol


def return_info_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """Annual return from securities data(frame)"""
    daily_rtn = data.pct_change(1).iloc[1:, ]
    annual_rtn = np.mean(daily_rtn) * 252
    ann_vol = np.std(daily_rtn) * np.sqrt(252)
    info_ratio = np.divide(annual_rtn, ann_vol)
    return info_ratio


def return_sharpe_ratio(data: pd.DataFrame, risk_free: float = 0) -> np.ndarray:
    """Function to give annualised Sharpe Ratio measure from input data,
    user input risk free rate

    Args:
        data
        risk_free: Risk free rate, as a decimal, so RFR of 6% = 0.06

    Returns:
        np.ndarray
    """
    print(f"Risk free rate set as {risk_free}")

    annual_rtn = calculate_annualised_return_df(data=data)
    annual_vol = calculate_annual_volatility_df(data=data)
    sharpe_ratio = np.divide(annual_rtn - risk_free, annual_vol)
    return sharpe_ratio


def return_sortino_ratio(data: pd.DataFrame,
                         target_return: float,
                         risk_free: float,
                         rtn_period: int = 1) -> np.ndarray:
    """Method to calculate Sortino Ratio (gives a better measure of downside volatility, thus risk.
    Unlike the Sharpe Ratio it does not penalise upside volatility.

    Args:
        data: Original dataframe of input data
        target_return: Target return (for the return period)
        risk_free: Risk free rate, annualised
        rtn_period: Specify the return period (number of days) for the ratio.

    Returns:
        ndarray: sortino ratio
    """

    period_return = data.pct_change(rtn_period).iloc[1:, ]
    downside_return = np.array(period_return.values - target_return)

    inner_bit = np.minimum(np.zeros(shape=downside_return.shape[1]), downside_return)

    tdd_sum = np.sum(np.square(inner_bit), axis=0)
    target_downside_dev = np.sqrt(tdd_sum / len(period_return))

    sortino = (period_return.mean() - risk_free) / target_downside_dev

    return sortino


@deprecated
def clean_bloomberg_security_data(input_file):
    """
    Params:
        inputFile: csv
            Data starts on the third row in format date | float.
            e.g.
            TICKER      | (empty)    | (empty) | ...
            "Date"      | "PX_LAST"  | (empty) | ...
            DD/MM/YYYY  | float      | (empty) | ...
    Read csv file with two columns from bloomberg one with the date, and the other with the price.

    Returns:
         data (dataframe): Melted dataframe
    """

    a = pd.read_csv(input_file, header=None)
    a = a.copy()
    product = a.iloc[0, 0]

    data = a.iloc[2:, :]
    data = data.copy()
    data['product'] = product
    data.columns = ['date', 'price', 'product']
    data = data[['product', 'date', 'price']]
    return data


@deprecated
def return_melted_df(input_file):
    """
    Read csv file with Bloomberg data (in format below with or without blank columns) and create
    melted pivot format inputFile
    bb ticker | (empty)         | bb ticker | (empty)
    Date      | "PX_LAST"       | Date      | "PX_LAST"
    dd/mm/yyyy| float           | dd/mm/yyyy| float

    Returns:
    Contract    | Date      | Price
    xxxx        | dd/mm/yy  | ##.##
    """

    x = pd.read_csv(input_file, header=None, parse_dates=True)
    x.dropna(axis=1, how='all', inplace=True)

    if any(pd.DataFrame(x.iloc[1, :]).drop_duplicates() == ['Date', "PX_LAST"]):
        x = x.copy(True)
        if x.shape[1] % 2 == 0:
            df = pd.DataFrame()
            for i in range(0, x.shape[1], 2):
                product = x.iloc[0, i]  # extract name of product/security
                data = x.iloc[2:, i:i + 2]
                data.dropna(inplace=True)
                data.reset_index(drop=True, inplace=True)

                data['product'] = product

                data.columns = ['date', 'price', 'product']
                # data['date'] = np.datetime64(data['date'])
                dates = np.array(data['date'])

                # create a mask to ensure that all entries have the correct date format,
                # not just Excel serial numbers
                res = []
                for date in dates:
                    res.append(len(date))
                res = np.array(res)
                mask = (res != 10)
                corrected_dates = pd.to_datetime(
                    excel_date_to_np(np.array(dates[mask], dtype='int32'))
                ).strftime('%Y/%m/%d')
                dates[mask] = corrected_dates
                data['date'] = dates

                data = data[['product', 'date', 'price']]
                df = df.append(data)
                return df
        else:
            print("Dataframe is not in the correct format")
    else:
        raise TypeError("The dataframe is not in the format expected with columns: [Date, PX_LAST]")


if __name__ == '__main__':
    pass
