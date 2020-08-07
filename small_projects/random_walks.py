"""
Created on: 30 Jun 2019
Created by: Philip.P

Look at the random walk process and prove non-stationarity through tests,
also show how the random walk process works
"""

import random
from typing import Tuple, List


def random_walk(n: int) -> Tuple:
    """Return coordinates after n block random walk"""
    x, y = 0, 0
    for i in range(n):
        (dx, dy) = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        x += dx
        y += dy
    return x, y


def random_price(start: int, tick: float, num_walks: int) -> List:
    """
    Quick function to generate brownian random motion price data

    Args:
        start: Start price for the random security data
        tick: Tick size for price change
        num_walks: Number of floats to move by

    Returns
        price_lst: List of prices with starting price and consequent walks
    """
    price_lst = [start]
    for k in range(num_walks):
        start += random.choice([tick, (-tick)])
        price_lst.append(start)
    return price_lst


if __name__ == "__main__":

    for i in range(25):
        walk = random_walk(10)
        print(walk, "Distance from home is", (abs(walk[0]) + abs(walk[1])))
    # What is the longest walk you can take so that you will end up only
    # distance of 4 from home

    # number_walks = 50000
    #
    # for walk_length in range(1,31):
    #     no_transport = 0 # number of walks 4 or fewer from home
    #     for i in range(number_walks):
    #         (x,y) = random_walk(walk_length)
    #         distance = abs(x) + abs(y)
    #         if distance <= 4:
    #             no_transport += 1
    #     no_transport_pc = float(no_transport) / number_walks
    #
    #     print(f"Walk size: {walk_length}, % no transport: {100*no_transport_pc}")
