# Created 17 June 2020. Transplated from utils_generic.py

import numpy as np
import pandas as pd

from utils_date import excel_date_to_np


def log_daily_returns(data):
    """Give log daily returns"""
    log_daily_return = data.apply(lambda x: np.log(x) - np.log(x.shift(1)))[1:]
    return log_daily_return


def daily_return(data):
    """Function to generate daily returns given input data (in dataframe, dtypes float, no time data)

    Example:
        >>> daily_return(data=pd.DataFrame([1,2,3,4]))
    """
    return data.pct_change(1).iloc[1:, ]


def annual_return(data):
    """Annual return from securities data(frame)"""
    daily_rtn = data.pct_change(1).iloc[1:, ]
    ann_rtn = np.mean(daily_rtn) * 252
    return ann_rtn


def annual_vol(data):
    """Annual return from securities data(frame)"""
    daily_rtn = data.pct_change(1).iloc[1:, ]
    ann_vol = np.std(daily_rtn) * np.sqrt(252)
    return ann_vol


def calc_info_ratio(data):
    """Annual return from securities data(frame)"""
    daily_rtn = data.pct_change(1).iloc[1:, ]
    annual_rtn = np.mean(daily_rtn) * 252
    ann_vol = np.std(daily_rtn) * np.sqrt(252)
    info_ratio = np.divide(annual_rtn, ann_vol)
    return info_ratio


def calc_sharpe_ratio(data, risk_free):
    """Function to give annualised Sharpe Ratio measure from input data, as well as risk free rate

    Args:
        data (dataframe)
        risk_free (float): Risk free rate, as a decimal, so RFR of 6% = 0.06
    """
    daily_rtn = data.pct_change(1).iloc[1:, ]
    annual_rtn = np.mean(daily_rtn) * 252
    ann_vol = np.std(daily_rtn) * np.sqrt(252)
    sharpe_ratio = np.divide(annual_rtn - risk_free, ann_vol)
    return sharpe_ratio


def calc_sortino_ratio(data, target_return, risk_free, rtn_period=1):
    """Method to calculate Sortino Ratio (gives a better measure of downside volatility, thus risk.
    Unlike the Sharpe Ratio it does not penalise upside volatility.

    Args:
        data (dataframe): Original dataframe of input data
        target_return (float): Target return (for the return period)
        risk_free (float): Risk free rate, usually annualised
        rtn_period (float, default 1): Specify the return period (number of days) for the ratio.

    Returns:
        sortino (ndarray)
    """

    prd_return = data.pct_change(rtn_period).iloc[1:, ]
    downside_return = np.array(prd_return.values - target_return)

    inner_bit = np.minimum(np.zeros(shape=downside_return.shape[1]), downside_return)

    tdd_sum = np.sum(np.square(inner_bit), axis=0)
    target_downside_dev = np.sqrt(tdd_sum / len(prd_return))

    sortino = (prd_return.mean() - risk_free) / target_downside_dev

    return sortino


def return_clean_df(input_file):
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

    # if a.iloc[1, 1] == "PX_LAST":
    #     measure = "price"
    # else:
    #     measure = "[to populate]"

    data = a.iloc[2:, :]
    data = data.copy()
    data['product'] = product
    data.columns = ['date', 'price', 'product']
    data = data[['product', 'date', 'price']]
    return data


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

                # create a mask to ensure that all entries have the correct date format, not just Excel serial numbers
                res = []
                for i in dates:
                    res.append(len(i))
                res = np.array(res)
                mask = (res != 10)
                datesToCorrect = pd.to_datetime(excel_date_to_np(np.array(dates[mask], dtype='int32'))).strftime(
                    '%Y/%m/%d')
                dates[mask] = datesToCorrect
                data['date'] = dates

                data = data[['product', 'date', 'price']]
                df = df.append(data)
                return df
        else:
            print("Dataframe is not in the correct format")
    else:
        raise TypeError("The dataframe is not in the format expected with columns: [Date, PX_LAST]")