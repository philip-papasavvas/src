"""
Author: Philip
Date created: 05/05/19

Script to look at producing an efficient frontier for a given portfolio of securities, thus portfolio optimisation
Based (heavily) on https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
from re import sub, search
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
from pp_utils import char_to_date

dateToStr = lambda d: d.astype(str).replace('-', '')

plt.style.use('seaborn')

runDate = np.datetime64(dt.datetime.now().strftime("%Y-%m-%d"))
runDateStr = dateToStr(runDate)

dateToStr = lambda d: d.astype(str).replace('-', '')

wkdir = "C://Users//Philip//Documents//python//"
inputFolder = wkdir + "input/"
outputDir = wkdir + "output/"
outputFolder = outputDir + runDateStr + "/"

if not os.path.exists(outputFolder):
   os.mkdir(outputFolder)

df = pd.read_csv(inputFolder + "example_data.csv")
df = char_to_date(df)
df.dropna(axis=0, inplace=True)
df.set_index('Date', inplace=True)

# -------------------------------------
#Random stock portfolio generation - decide what proportion of each stock in portfolio
# -------------------------------------
#TODO: sort out sharpe ratio calculations as they seem to be wrong, name for efficient frontier png when saved

#Using methods from article
def port_annual_perf(weights, mean_returns, cov):
    """
    Function to calculate returns and volatility of a given portfolio

    Params:
        weights: list
        mean_returns: dataframe
        cov: dataframe
            Covariance matrix of mean_returns

    Returns:
        returns: annualised return of the portfolio
        std: volatility of returns of portfolio
    """

    returns = np.sum(mean_returns*weights)* 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    return std, returns

# Example
# df = pd.read_csv(inputFolder + "example_data.csv", index_col="Date", parse_dates=True)
# df.dropna(axis=0, inplace=True)
#
# returns = df.pct_change()
# mean_returns = returns.mean()
# cov = returns.cov()
#
# a = port_annual_perf(weights = np.repeat(0.25,4), mean_returns= mean_returns, cov=cov)


def random_portfolios(nPortfolios, mean_returns, cov, risk_free):
    """
    Function to return portfolio (std dev, return, Sharpe ratio)

    Params:
        nPortfolios: int
            Number of different portfolios to try with random allocation to each stock
        mean_returns: dataframe
            Returns of the stocks
        cov: dataframe
            Covariance matrix of returns
        risk_free: float
            Risk free rate of return from investing

    Returns:
        results: tuple (std dev, ret, Sharpe) for each portfolio
        weights_lst: weight of each stock in portfolio
    """

    results = np.zeros((nPortfolios,3))
    weights_lst = []
    for i in range(nPortfolios):
        weights = np.random.random(mean_returns.shape[0]) #four random numbers between 0 and 1
        weights /= np.sum(weights)
        weights_lst.append(weights)

        port_st_dev, port_ret = port_annual_perf(weights = weights, mean_returns = mean_returns, cov=cov)

        results[i,0] = port_st_dev
        results[i,1] = port_ret
        results[i,2] = (port_ret - risk_free)/port_st_dev

    return results, weights_lst

# Example
# df = pd.read_csv(inputFolder + "example_data.csv", index_col="Date", parse_dates=True)
# df.dropna(axis=0, inplace=True)
# daily_return = df.pct_change()
# mean_return = daily_return.mean()
# covariance_return = daily_return.cov()
# risk_free = 0.015
# fundamentals, wts = random_portfolios(nPortfolios=100, mean_returns=mean_return, cov=covariance_return, risk_free=0.015)


def display_simulated_frontier_random(mean_returns, cov, nPortfolios, risk_free, saveResults= False, savePlots=False):
    """"
    Plotting function to plot the annualised return and volatility for each different portfolio,
    plot the efficient frontier, and identify the most optimal portfolio for investment (judging)
    by the highest Sharpe Ratio, and also the least risky investment

    Params:
        mean_returns: dataframe
            Returns of the stocks
        cov: dataframe
            Covariance matrix of returns
        risk_free: float
            Risk free rate of return from investing
        nPortfolios: int
            Number of different portfolios to try with random allocation to each stock
        saveResults: bool, default False
            Save results for all portfolios tested in output folder
        savePlots: bool, default False
            Save plots in output folder

    Returns:
        fig: figure
            Plot of the efficient frontier for the given input portfolio, with starred points
            for highest Sharpe Ratio and lowest volatility (on the efficient frontier) portfolios
        allocation: lst
            Allocation of weights to securities in the portfolio
    """

    results, weights = random_portfolios(nPortfolios, mean_returns, cov, risk_free)

    # Maximum Sharpe ratio locations
    max_Sharpe_loc = np.argmax(results[:,2])
    sharpe_max = results[max_Sharpe_loc,2]
    ret_ms, std_ms = results[max_Sharpe_loc, 1], results[max_Sharpe_loc, 0] #return, std of portfolio
    max_Sharpe_allocation = pd.DataFrame(weights[max_Sharpe_loc], index= df.columns, columns = ['allocation'])
    max_Sharpe_allocation['allocation'] = np.round(max_Sharpe_allocation['allocation']*100,2)

    # Minimum volatility locations
    min_vol_loc = np.argmin(results[:,0])
    sharpe_min_vol = results[min_vol_loc,2]
    ret_mv, std_mv = results[min_vol_loc, 1], results[min_vol_loc, 0]  # return, std of portfolio
    min_vol_allocation = pd.DataFrame(weights[min_vol_loc], index=df.columns, columns=['allocation'])
    min_vol_allocation['allocation'] = np.round(min_vol_allocation['allocation'] * 100, 2)

    print("-"*50 + "\n", "Maximum Sharpe Ratio Portfolio Allocation: \n \n",
          "\t Sharpe Ratio: \t" + str(round(sharpe_max, 4)), "\n",
          "\t Annualised Return: \t" + str(round(ret_ms,4)),
          " \n \t Annualised Volatility: \t"  + str(round(std_ms, 4)), "\n \n",
          max_Sharpe_allocation.T)
    print("-" * 50 + "\n" "Minimum volatility Portfolio Allocation: \n",
          "\t Sharpe Ratio: \t" + str(round(sharpe_min_vol,4)), "\n",
          "\t Annualised Return: \t" + str(round(ret_mv,4)),
          " \n \t Annualised Volatility: \t"  + str(round(std_mv, 4)), "\n \n",
          min_vol_allocation.T)

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
    if savePlots == True:
        plt.savefig(outputFolder + "_EfficientFrontier_" + str(nPortfolios) + "_" + str(risk_free) + ".png")

    if saveResults == True:
        # output = pd.ExcelWriter(outputFolder + "_Portfolio Optimisation - Efficient Frontier.xlsx")
        # perf_measures = pd.DataFrame(results, columns = ['Annualised Volatility', 'Annualised Return', 'Sharpe Ratio'])
        # perf_measures.index.name = 'Portfolio number'
        # perf_measures.to_excel(output, "Portfolio_Performance")
        #
        # alloc = pd.DataFrame(weights, columns = [i + "_%" for i in df.columns)
        # alloc.to_excel(output, "Portfolio allocations")
        # output.save()

        perf_measures = pd.DataFrame(results*(100,100,1), columns=['Annualised Volatility %', 'Annualised Return %', 'Sharpe Ratio'])
        perf_measures.index.name = 'Portfolio number'
        alloc = pd.DataFrame(weights*100, columns=[i + "_%" for i in df.columns])
        summary = pd.concat([perf_measures, alloc], axis=1)
        summary.sort_values(by=['Sharpe Ratio'], ascending=False, inplace=True)

        summary.to_csv(outputFolder + str(nPortfolios) + "_Portfolio_Optimisation.csv")

if __name__ == "__main__":
    # --------------

    # EXPLORATORY DATA ANALYSIS
    # --------------
    # plot the prices
    plt.figure(figsize=(13, 7))
    for i in df.columns.values:
        plt.plot(df.index, df[i], lw=3, alpha=0.8, label=i)
    plt.legend(loc='upper left', fontsize=10)
    plt.ylabel("Price (p)")
    plt.savefig(outputFolder + "price_plot.png")
    plt.close()

    returns = df.pct_change()
    plt.figure(figsize=(13, 7))
    for i in returns.columns.values:
        plt.plot(returns.index, returns[i], lw=1, alpha=0.8, label=i)
    plt.legend(loc='upper left', fontsize=10)
    plt.ylabel("Daily Returns (%)")
    plt.savefig(outputFolder + "returns_plot.png")
    plt.close()

    # Example
    df = pd.read_csv(inputFolder + "example_data.csv", index_col="Date", parse_dates=True)
    df.dropna(axis=0, inplace=True)
    daily_return = df.pct_change()
    mean_return = daily_return.mean()
    covariance_return = daily_return.cov()
    # risk_free = 0.015

    res = display_simulated_frontier_random(mean_returns= mean_return, cov= covariance_return, nPortfolios=50000,
                                            risk_free=0.015, saveResults=True, savePlots=True)
