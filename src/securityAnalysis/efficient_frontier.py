"""
Created on: 5 May 2019

Efficient frontier for a given portfolio of securities, thus portfolio optimisation
Inspiration:
https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
"""

import numpy as np
import pandas as pd

if __name__ == "main":

    # For a given dataframe of security price data (sec_df) with n columns
    # returns = sec_df.pct_change(), mean_returns = returns.mean()
    # cov = returns.cov()
    from src.small_projects.random_walks import random_price

    a, b = pd.Series(random_price(start=500, tick=2, num_walks=499)), pd.Series(
        random_price(start=1000, tick=1, num_walks=499))
    example_df = pd.DataFrame(data=pd.concat([a, b], axis=1))
    # index=pd.date_range(start="2000-01-01", periods=500)
    returns = example_df.pct_change()
    mean_returns, cov = returns.mean(), returns.cov()

    annualised_portfolio_returns = np.sum(mean_returns * np.repeat(0.5, 2)) * 252
    std = np.sqrt(np.dot(np.repeat(0.5, 2).T, np.dot(cov, np.repeat(0.5, 2)))) * np.sqrt(252)

