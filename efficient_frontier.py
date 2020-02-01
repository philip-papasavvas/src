"""
Author: Philip
Date created: 05 May 2019

Script to look at producing an efficient frontier for a given portfolio of securities, thus portfolio optimisation
Used the following for inspiration on https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
"""
# import build in modules
import os
import pandas as pd
import numpy as np
import datetime as dt
from re import sub, search

# third party imports
import matplotlib.pyplot as plt

# local imports
import utils
from __init__ import get_data_path

date_to_str = lambda d: d.astype(str).replace('-', '')
plt.style.use('seaborn')


def plot_raw_data(df, dir, file_name, y_label, legend_loc="best", to_close=False):
    """Function to do basic plotting of securities data

    Args:
        df (dataframe): Multiple columns of securities data with index as date
        dir (str): Directory in which to save raw data
        file_name (str): Name of file to go before ".png" extension
        y_label (str): Labelling for y axis
        legend_loc (str, default 'best'): Available options: 'best', 'upper left', 'upper right', 'lower left',
            'lower right', 'upper center', 'lower center', 'center left', 'center right', 'center'
        to_close (bool, default False): Whether to close the plot after plotting
    """
    plt.figure(figsize=(13, 7))
    for i in df.columns.values:
        plt.plot(df.index, df[i], lw=0.5, alpha=0.8, label=i)
    plt.legend(loc=legend_loc, fontsize=10)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(dir, file_name + ".png"))
    if to_close:
        plt.close()


# RANDOM PORTFOLIO GENERATION

def port_perform_annual(mean_returns, cov, weights):
    """
    Calculate annual portfolio performance using given mean returns and weight allocation per security

    Args:
        mean_returns (dataframe): Dataframe of mean returns of the securities
        cov (dataframe): Covariance matrix of returns
        weights (array): Weights (in an array) to allocate to each security

    Returns:
        returns (df): Annualised return of the portfolio
        std (df): Volatility of returns of portfolio

    Example: see below __main__ for an example
    """
    returns = np.sum(mean_returns*weights)* 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    return returns, std


def random_portfolios(n_pts, mean_returns, cov, risk_free):
    """
    Function to return portfolio performance metrics: annual standard deviation (volatility), return and Sharpe ratio

    Args:
        n_pts (int): Number of different portfolios to try with random allocation to each stock
        mean_returns (dataframe): Mean returns of the securities
        cov (dataframe): Covariance matrix of returns
        risk_free (float): Risk free rate of return from investing. 0.015 = 1.5%

    Returns:
        results (tuple): (std dev, ret, Sharpe) for each portfolio
        weights_lst (list): weight of each stock in portfolio
    """

    results = np.zeros((n_pts,3))
    weights_lst = []
    for i in range(n_pts):
        weights = np.random.random(mean_returns.shape[0]) # random numbers between 0 and 1
        weights /= np.sum(weights)
        weights_lst.append(weights)

        port_st_dev, port_ret = port_perform_annual(weights=weights, mean_returns=mean_returns, cov=cov)

        results[i,0] = port_st_dev
        results[i,1] = port_ret
        results[i,2] = (port_ret - risk_free)/port_st_dev

    return results, weights_lst


def display_simulated_frontier_random(mean_returns, cov, n_ports, risk_free_rate, wk_dir=None, \
                                      to_save_results=False, to_save_plots=False):
    """"
    Plot the annualised return and volatility for different portfolios,
    plot the efficient frontier, and identify the most optimal portfolio for investment (judging)
    by the highest Sharpe Ratio, and also portfolio with the lowest risk (standard deviation)

    Args:
        mean_returns (dataframe): Returns of the stocks
        cov (dataframe): Covariance matrix of returns
        risk_free_rate (float): Risk free rate of return from investing 0.015 = 1.5%
        n_ports (int): Number of different portfolios to try with random allocation to each stock
        to_save_results (bool, default False): Save results for all portfolios tested in specified working directory
        to_save_plots (bool, default False): Save plots in specified working directory

    Returns:
        None.
        Print statement with allocation of securities recommended by weight in the portfolio

        if save_results:
            Plot of the efficient frontier for the given input portfolio, with starred points
            for highest Sharpe Ratio and lowest volatility (on the efficient frontier) portfolios
    """
    if (to_save_plots or to_save_results) and wk_dir is None:
        raise AttributeError
        print("User has opted to save results but has not specified working directory")

    results, weights = random_portfolios(n_ports, mean_returns, cov, risk_free_rate)

    # Maximum Sharpe ratio locations
    max_Sharpe_loc = np.argmax(results[:,2])
    sharpe_max = results[max_Sharpe_loc,2]
    ret_ms, std_ms = results[max_Sharpe_loc, 1], results[max_Sharpe_loc, 0] #return, st_dev of portfolio
    max_Sharpe_allocation = pd.DataFrame(weights[max_Sharpe_loc], index= df.columns, columns=['allocation'])
    max_Sharpe_allocation['allocation'] = np.round(max_Sharpe_allocation['allocation']*100,2)

    # Minimum volatility locations
    min_vol_loc = np.argmin(results[:,0])
    sharpe_min_vol = results[min_vol_loc,2]
    ret_mv, std_mv = results[min_vol_loc, 1], results[min_vol_loc, 0]  # return, std of portfolio
    min_vol_allocation = pd.DataFrame(weights[min_vol_loc], index=df.columns, columns=['allocation'])
    min_vol_allocation['allocation'] = np.round(min_vol_allocation['allocation'] * 100, 2)

    if to_save_results:
        perf_measures = pd.DataFrame(results * (100, 100, 1),
                                     columns=['Annualised Volatility %', 'Annualised Return %', 'Sharpe Ratio']
                                     )
        perf_measures.index.name = 'Portfolio number'
        alloc = pd.DataFrame(weights * 100, columns=[i + "_%" for i in df.columns])
        summary = pd.concat([perf_measures, alloc], axis=1)
        summary.sort_values(by=['Sharpe Ratio'], ascending=False, inplace=True)

        summary.to_csv(os.path.join(wk_dir, str(n_ports) + "_PortfolioOptimisation.csv"))

    print(f"-"*50 + "\n", "Maximum Sharpe Ratio Portfolio Allocation: \n \n",
          f"\t Sharpe Ratio: \t {round(sharpe_max, 4)} \n",
          f"\t Annualised Return: \t {round(ret_ms,4)}",
          f" \n \t Annualised Volatility: \t {round(std_ms, 4)} \n \n", max_Sharpe_allocation.T)
    print(f"-"*50 + "\n" "Minimum volatility Portfolio Allocation: \n",
          f"\t Sharpe Ratio: \t {round(sharpe_min_vol,4)}, \n",
          f"\t Annualised Return: \t {round(ret_mv,4)}",
          f" \n \t Annualised Volatility: \t {round(std_mv, 4)} \n \n", min_vol_allocation.T)

    # Plot the efficient frontier for the portfolios tested
    plt.figure(figsize=(10,7))
    plt.scatter(results[:,0], results[:,1], c = results[:,2], cmap='YlGnBu', marker = "x", s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(std_ms, ret_ms, marker="*", color="r", s=500, label='Maximum Sharpe Ratio')
    plt.scatter(std_mv, ret_mv, marker="*", color="b", s=500, label='Minimum Volatility')
    plt.xlabel("Annualised volatility")
    plt.ylabel("Annualised returns")
    plt.legend()
    plt.title("Simulated Portfolio Optimisation using Efficient Frontier")

    if to_save_plots:
        plt.savefig(os.path.join(wkdir, "_".join(["EfficientFrontier", n_ports, risk_free_rate + ".png"])))


if __name__ == "main":

    run_date_str = dt.datetime.now().strftime("%Y-%m-%d")
    run_date_dt = np.datetime64(run_date_str)

    wkdir = r"/Users/philip_p/python/projects"
    output_dir = os.path.join(wkdir, "output", "efficient_frontier")
    today_output_dir = os.path.join(output_dir, run_date_str)

    if not os.path.exists(today_output_dir):
        os.mkdir(today_output_dir)

    df = pd.read_csv(get_data_path("example_data.csv"), index_col='Date')
    df = utils.char_to_date(df)
    df.dropna(axis=0, inplace=True)

    # EXPLORATORY DATA ANALYSIS
    # --------------
    # plot the prices - using plot_raw_data method defined above

    # plot_raw_data(df= df, file_name="price_plot", y_label="Price (p)", to_close=True)

    # # Now look at 1 day returns (should follow something resembling a normal distribution)
    # returns = df.pct_change()
    # plot_raw_data(df= returns, file_name="returns_plot", y_label="Daily Return (%)", to_close=True)

    # Explore the methods
    daily_return = df.pct_change()
    mean_return = daily_return.mean()
    covariance_return = daily_return.cov()
    risk_free = 0.015

    # Annualised performance of a portfolio with given weights for securities in dataframe
    sample_returns, sample_cov = port_perform_annual(mean_returns=mean_return, cov=covariance_return, \
                                                     weights=np.repeat(0.25,4))

    # Porfolio return metrics
    results, optimal_weights = random_portfolios(n_pts=20, mean_returns=mean_return, cov=covariance_return, risk_free=0.015)

    # Plot efficient frontier for portfolios (annualised return versus volatility) - bullet shape,
    # highlighting most efficient portfolio (risk/reward and lowest volatility)
    res = display_simulated_frontier_random(mean_returns= mean_return, cov= covariance_return, n_ports=50, \
                                            risk_free_rate=0.015, wk_dir=today_output_dir, to_save_results=True, to_save_plots=True)

    # For a given dataframe of security price data (sec_df) with n columns
    # returns = sec_df.pct_change(), mean_returns = returns.mean()
    # cov = returns.cov()
    from small_projects.random_walks import random_price
    a, b = pd.Series(random_price(start=500, tick=2, walks=499)), pd.Series(
        random_price(start=1000, tick=1, walks=499))
    df = pd.DataFrame(data=pd.concat([a, b], axis=1))  # index=pd.date_range(start="2000-01-01", periods=500)
    returns = df.pct_change()
    mean_returns, cov = returns.mean(), returns.cov()
    returns, cov_mtrx = port_perform_annual(mean_returns=mean_returns, cov=cov, weights=np.repeat(0.5, 2))
