import numpy as np
import pandas as pd
import os

import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode,iplot
# init_notebook_mode(connected=True)
# %matplotlib inline

basePath = "C:\\Users\\ppapasav\\Desktop\\Python-Data-Science-and-Machine-Learning-Bootcamp"

os.chdir("C:\\Users\\ppapasav\\Desktop\\")
df = pd.read_csv(basePath +'911.csv')