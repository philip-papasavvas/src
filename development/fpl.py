"""
Created on: 07 Aug 2019

Script to download FPL player data and run analyses
"""

import numpy as np
import pandas as pd
import json
import os

with open(r"C:\Users\Philip\PycharmProjects\PythonSkills\data\fpl_static_20190807.json") as json_file:
    data = json.load(json_file)

a = pd.read_csv(r"C:\Users\Philip\PycharmProjects\PythonSkills\data\fpl_players_201819.csv")