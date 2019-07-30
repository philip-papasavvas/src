"""
Created on 20 Jul 2019
author: Philip

Create an instance of plotly to view financial data in HTML format, not on
graphs as I have done before.
Change lookback period, toggle for price and total return

"""
import numpy as np
import pandas as pd

from plotly import __version__
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0

import cufflinks as cf

df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
#df.head()
#
df_2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
#df_2.head()

cf.go_offline()

plotly.offline.plot(df.iplot(kind='scatter',x='A',y='B'), filename='file.html')

