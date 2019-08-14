"""
Created on 07 Aug 2019

Script to give holding period return and overall return for a stock by specifying units
purchased, and output a summary table
"""

import numpy as np
import pandas as pd
import os
import utils
import datetime
import json

import random
np.random.seed(100)

today = datetime.datetime.now().strftime("%Y%m%d")
to_dt64 = lambda x: np.datetime64(x, "D")

def calculate_performance(data, security, dates_transacted, units, verbose=True):
    """
    Function to give performance measures for a security

    Params
    ------
        data: df
            Security data input
        dates_transacted: list
            List of dates when the stock was purchased/sold
        units: list
            List of units purchased/sold corresponding to the dates_transacted
        verbose: bool, default False
            To print commentary throughout the script

    Returns
    -------
        table: dataframe
            Detailed output of transactions
        summary_table: dataframe
            Summary of transactions for security
    """
    #TODO: extend this to handle multiple securities

    sec_slice = data.loc[:, security]

    if verbose:
        print("Checking valid dates... \n")

    assert all([x in sec_slice.index for x in dates_transacted]), "Dates transacted are not valid/in the " \
                                                                  "lookback period"

    final_price = sec_slice.iloc[-1] #.values[0]
    final_date = sec_slice.index[-1]

    assert len(dates_transacted) == len(units), "Ensure full mapping of number of units" \
                                                "transacted on each particular date"

    prices = []
    for k in dates_transacted:
      prices.append(sec_slice.loc[k]) #.values[0])


    if verbose:
        print("Assembling summary table... \n")

    table = pd.DataFrame()
    table['Date Purchased'] = dates_transacted
    table['Price'] =  prices
    table['Units Purchased'] = units
    table['Value Purchased'] = np.array(prices) * np.array(units)
    table['Total Units'] = np.cumsum(table['Units Purchased'])
    table['Portfolio Value'] = table['Price'] * table['Total Units']
    table['Final Date'] = final_date
    table['Final Price'] = final_price
    table['Final Value'] = np.array(final_price) * np.array(table['Total Units'].values)
    table['Price Change'] = table['Final Price'].values/table['Price'].values - 1


    # need to convert input dates purchased to np.datetime64 to match the date type in the
    # table
    holding_d = np.array([to_dt64(x) for x in table['Final Date']]) - \
                            np.array([to_dt64(x) for x in dates_transacted])

    table['Holding Days'] = [np.timedelta64(x, "D").astype(int) for x in holding_d]
    table['Annualised Return'] = 365.25 * (table['Price Change'].values/table['Holding Days'].values)

    summary_table = pd.DataFrame()
    total_units = table['Total Units'].values[-1]

    summary_table['Total Units'] = total_units
    summary_table['Value Purchased'] = np.sum(table['Value Purchased'].values)
    summary_table['Final Value'] = np.product(final_price*total_units)

    summary = {'Final Date': final_date, 'Total Units': total_units,
               'Value Purchased': np.sum(table['Value Purchased'].values),
               'Final Value': np.product(final_price*total_units)}

    if verbose:
        print("Produced summary and detailed return data")

    summary['Total Return'] = (summary['Final Value']/summary['Value Purchased'])-1
    summary_table = pd.DataFrame(list(summary.items()))

    return table, summary_table

if __name__ == "__main__":

    wkdir = r"C:\Users\Philip\Documents\python\input\security_data"

    data = utils.prep_fund_data(df_path=os.path.join(wkdir, "price_data.csv")) #formatted data for dt64 types etc.

    security = 'MSCI ACWI'

    rand_int_date = np.random.randint(low=0, high=len(data.index), size=2)

    dates_transacted = [data.index[rand_int_date[i]] for i in range(2)]
    units = np.random.randint(low=1, high= 15, size=2)

    #securities = data.columns

        tab, summary = calculate_performance(data=data,
                                         security='MSCI ACWI',
                                         dates_transacted=dates_transacted,
                                         units=units)

    # % timeit - n 5 calculate_performance(data=data, \
    #                       security='MSCI ACWI', \
    #                       dates_transacted=dates_transacted, \
    #                       units=units)
    # 109 ms ± 1.98 ms per loop (mean ± std. dev. of 7 runs, 5 loops each)

    # presentable format
    summary = summary.T
    summary.columns = summary.iloc[0]
    summary = summary[1:]

    summary.set_index("Final Date", inplace=True)
    summary[['Value Purchased', 'Final Value']] = \
        summary[['Value Purchased', 'Final Value']].applymap("£{0:,.2f}".format)


    # Get data in melted format
    data_melted  = pd.melt(data.reset_index(), id_vars=['Date'], value_vars =data.columns)
    data_melted.rename(columns={'variable':'security', 'value':'close_px'}, inplace=True)
    data_melted.dropna(axis=0, inplace=True)

    data_melted.to_json(os.path.join(wkdir, "price_data.json"))

    with open(os.path.join(wkdir, "price_data.json")) as rand_df:
         data_load= json.load(rand_df)