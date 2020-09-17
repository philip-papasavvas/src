"""
Created 5 May 2019

Produce efficient frontier for a given portfolio of securities, thus portfolio optimisation
Inspiration:
https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
"""

import datetime as dt
import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils_date
from get_paths import get_data_path

plt.style.use('seaborn')


def plot_raw_data(input_data: pd.DataFrame, output_dir: str, file_name: str, y_label: str,
                  legend_loc: str = "best", to_close: bool = False) -> None:
    """Function to do basic plotting of securities data

    Args:
        input_data: Multiple columns of securities data with index as date
        output_dir: Directory in which to save raw data
        file_name
        y_label: Labelling for y axis
        legend_loc: One from ['best', 'upper left', 'upper right', 'lower left', 'lower right',
        'upper center', 'lower center', 'center left', 'center right', 'center']
        to_close: Close plot after plotting

    Returns:
        None
    """

    plt.figure(figsize=(13, 7))
    for i in input_data.columns.values:
        plt.plot(input_data.index, input_data[i], lw=0.5, alpha=0.8, label=i)
    plt.legend(loc=legend_loc, fontsize=10)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(output_dir, file_name + ".png"))
    if to_close:
        plt.close()


# RANDOM PORTFOLIO GENERATION

def port_perform_annual(mean_return_df: pd.DataFrame, covariance_returns: pd.DataFrame,
                        weights: np.array) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate annual portfolio performance using given mean returns and weight allocation per stock

    Args:
        mean_return_df: Dataframe of mean returns of the securities
        covariance_returns: Covariance matrix of returns
        weights: Weights for security allocation

    Returns:
        pd.DataFrame: Annualised return of the portfolio
        pd.DataFrame Volatility of returns of portfolio
    """

    annualised_portfolio_returns = np.sum(mean_return_df * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(covariance_returns, weights))) * np.sqrt(252)

    return annualised_portfolio_returns, std


def random_portfolios(num_portfolios: int, mean_return_df: pd.DataFrame,
                      covariance_returns: pd.DataFrame,
                      risk_free_rate: float) -> Tuple[np.ndarray, List[float]]:
    """
    Calculation portfolio performance metrics: annual standard deviation (volatility), return
    and Sharpe ratio

    Args:
        num_portfolios: Number of different portfolios to try with random allocation to each stock
        mean_return_df: Mean returns of the securities
        covariance_returns: Covariance matrix of returns
        risk_free_rate: Risk free rate of return from investing. 0.015 = 1.5%

    Returns:
        tuple: (std dev, ret, Sharpe) for each portfolio
        list: weight of each stock in portfolio
    """

    portfolio_results = np.zeros((num_portfolios, 3))
    weights_lst = []
    for i in range(num_portfolios):
        weights = np.random.random(mean_return_df.shape[0])  # random numbers between 0 and 1
        weights /= np.sum(weights)
        weights_lst.append(weights)

        port_st_dev, portfolio_return = \
            port_perform_annual(weights=weights,
                                mean_return_df=mean_return_df,
                                covariance_returns=covariance_returns)

        portfolio_results[i, 0] = port_st_dev
        portfolio_results[i, 1] = portfolio_return
        portfolio_results[i, 2] = (portfolio_return - risk_free_rate) / port_st_dev

    return portfolio_results, weights_lst


def display_simulated_frontier_random(
        mean_return_df: pd.DataFrame, covariance_returns: pd.DataFrame, num_portfolios: int,
        risk_free_rate: float, wk_dir: str = None, to_save_results: bool = False,
        to_save_plots: bool = False) -> None:
    """"
    Plot the annualised return and volatility for different portfolios,
    plot the efficient frontier, and identify the most optimal portfolio for investment (judging)
    by the highest Sharpe Ratio, and also portfolio with the lowest risk (standard deviation)

    Args:
        mean_return_df: Security mean returns
        covariance_returns: Covariance matrix of returns
        risk_free_rate: Risk free rate of return from investing 0.015 = 1.5%
        wk_dir: Working directory in which to save results if to_save_results is True
        num_portfolios: Number of different portfolios to try with random allocation to each stock
        to_save_results: Save results for all portfolios tested in specified working directory
        to_save_plots: If to save plots in specified working directory

    Returns:
        None.
        Print statement with allocation of securities recommended by weight in the portfolio

        if save_results:
            Plot of the efficient frontier for the given input portfolio, with starred points
            for highest Sharpe Ratio and lowest volatility (on the efficient frontier) portfolios

    """
    if (to_save_plots or to_save_results) and wk_dir is None:
        print("User has opted to save results but has not specified working directory")
        raise AttributeError

    portfolio_metrics, weights = random_portfolios(num_portfolios=num_portfolios,
                                                   mean_return_df=mean_return_df,
                                                   covariance_returns=covariance_returns,
                                                   risk_free_rate=risk_free_rate)

    # Maximum Sharpe ratio locations
    max_sharpe_loc = np.argmax(portfolio_metrics[:, 2])
    sharpe_max = portfolio_metrics[max_sharpe_loc, 2]
    # return, st_dev of portfolio
    return_max_sharpe = portfolio_metrics[max_sharpe_loc, 1]
    st_dev_max_sharpe = portfolio_metrics[max_sharpe_loc, 0]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_loc],
                                         index=example_df.columns,
                                         columns=['allocation'])
    max_sharpe_allocation['allocation'] = np.round(max_sharpe_allocation['allocation'] * 100, 2)

    # Minimum volatility locations
    min_vol_loc = np.argmin(portfolio_metrics[:, 0])
    sharpe_min_vol = portfolio_metrics[min_vol_loc, 2]
    return_min_vol = portfolio_metrics[min_vol_loc, 1]
    st_dev_min_vol = portfolio_metrics[min_vol_loc, 0]
    min_vol_allocation = pd.DataFrame(weights[min_vol_loc],
                                      index=example_df.columns,
                                      columns=['allocation'])
    min_vol_allocation['allocation'] = np.round(min_vol_allocation['allocation'] * 100, 2)

    if to_save_results:
        perf_measures = pd.DataFrame(
            data=portfolio_metrics * (100, 100, 1),
            columns=['Annualised Volatility %', 'Annualised Return %', 'Sharpe Ratio']
        )
        perf_measures.index.name = 'Portfolio number'
        alloc = pd.DataFrame(weights * 100, columns=[i + "_%" for i in example_df.columns])
        summary = pd.concat([perf_measures, alloc], axis=1)
        summary.sort_values(by=['Sharpe Ratio'], ascending=False, inplace=True)

        summary.to_csv(f"{wk_dir}/{str(num_portfolios)}_PortfolioOptimisation.csv")

    print(f"-" * 50 + "\n", "Maximum Sharpe Ratio Portfolio Allocation: \n \n",
          f"\t Sharpe Ratio: \t {round(sharpe_max, 4)} \n",
          f"\t Annualised Return: \t {round(return_max_sharpe, 4)}",
          f" \n \t Annualised Volatility: \t {round(st_dev_max_sharpe, 4)} \n \n",
          max_sharpe_allocation.T)
    print(f"-" * 50 + "\n" "Minimum volatility Portfolio Allocation: \n",
          f"\t Sharpe Ratio: \t {round(sharpe_min_vol, 4)}, \n",
          f"\t Annualised Return: \t {round(return_min_vol, 4)}",
          f" \n \t Annualised Volatility: \t {round(st_dev_min_vol, 4)} \n \n",
          min_vol_allocation.T)

    # Plot the efficient frontier for the portfolios tested
    plt.figure(figsize=(10, 7))
    plt.scatter(x=portfolio_metrics[:, 0],
                y=portfolio_metrics[:, 1],
                c=portfolio_metrics[:, 2],
                cmap='YlGnBu',
                marker="x",
                s=10,
                alpha=0.3)
    plt.colorbar()
    plt.scatter(x=st_dev_max_sharpe, y=return_max_sharpe, marker="*",
                color="r", s=500, label='Maximum Sharpe Ratio')
    plt.scatter(x=st_dev_min_vol, y=return_min_vol, marker="*",
                color="b", s=500, label='Minimum Volatility')
    plt.xlabel("Annualised volatility")
    plt.ylabel("Annualised returns")
    plt.legend()
    plt.title("Simulated Portfolio Optimisation using Efficient Frontier")

    if to_save_plots:
        plt.savefig(
            WORK_DIR + f"/EfficientFrontier_{num_portfolios}_{risk_free_rate}.png")


if __name__ == "main":

    RUN_DATE_STR = dt.datetime.now().strftime("%Y-%m-%d")
    RUN_DATE_DT = np.datetime64(RUN_DATE_STR)

    WORK_DIR = r"/Users/philip_p/python/projects"
    OUTPUT_DIR = os.path.join(WORK_DIR, "output", "efficient_frontier")
    TODAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, RUN_DATE_STR)

    if not os.path.exists(TODAY_OUTPUT_DIR):
        os.mkdir(TODAY_OUTPUT_DIR)

    example_df = pd.read_csv(get_data_path("example_data.csv"), index_col='Date')
    example_df = utils_date.char_to_date(example_df)
    example_df.dropna(axis=0, inplace=True)

    # EXPLORATORY DATA ANALYSIS
    # --------------
    # plot the prices - using plot_raw_data method defined above

    # plot_raw_data(df= df, file_name="price_plot", y_label="Price (p)", to_close=True)

    # # Now look at 1 day returns (should follow something resembling a normal distribution)
    # returns = df.pct_change()
    # plot_raw_data(df=returns, file_name="returns_plot",
    #               y_label="Daily Return (%)", to_close=True)

    # Explore the methods
    daily_return = example_df.pct_change()
    mean_return = daily_return.mean()
    covariance_return = daily_return.cov()
    RISK_FREE = 0.015

    # Annualised performance of a portfolio with given weights for securities in dataframe
    sample_returns, sample_cov = port_perform_annual(mean_return_df=mean_return,
                                                     covariance_returns=covariance_return,
                                                     weights=np.repeat(0.25, 4))

    # Porfolio return metrics
    results, optimal_weights = random_portfolios(num_portfolios=20,
                                                 mean_return_df=mean_return,
                                                 covariance_returns=covariance_return,
                                                 risk_free_rate=0.015)

    # Plot efficient frontier for portfolios (annualised return versus volatility) - bullet shape,
    # highlighting most efficient portfolio (risk/reward and lowest volatility)
    display_simulated_frontier_random(mean_return_df=mean_return,
                                      covariance_returns=covariance_return,
                                      num_portfolios=50,
                                      risk_free_rate=0.015,
                                      wk_dir=TODAY_OUTPUT_DIR,
                                      to_save_results=True,
                                      to_save_plots=True)

    # For a given dataframe of security price data (sec_df) with n columns
    # returns = sec_df.pct_change(), mean_returns = returns.mean()
    # cov = returns.cov()
    from small_projects.random_walks import random_price

    a, b = pd.Series(random_price(start=500, tick=2, num_walks=499)), pd.Series(
        random_price(start=1000, tick=1, num_walks=499))
    example_df = pd.DataFrame(data=pd.concat([a, b], axis=1))
    # index=pd.date_range(start="2000-01-01", periods=500)
    returns = example_df.pct_change()
    mean_returns, cov = returns.mean(), returns.cov()
    returns, cov_mtrx = port_perform_annual(mean_return_df=mean_returns,
                                            covariance_returns=cov,
                                            weights=np.repeat(0.5, 2))
