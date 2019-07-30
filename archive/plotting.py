import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

wkdir = "C://Users//Philip//Documents//python/"
inputDir = wkdir + "input/"
df = pd.read_csv(inputDir + 'example_data.csv', parse_dates=True, index_col='Date')

dailyReturn = df.pct_change(1).iloc[1:, ]
annualReturn = np.mean(dailyReturn) * 252
annualVolatility = np.std(dailyReturn) * np.sqrt(252)

# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

# Plot on that set of axes
axes.plot(df.index, df.iloc[:,1], 'b')
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('Set Title')

fig, axes = plt.subplots(nrows=4, ncols=1)

for ax in axes:
    ax.plot(df.index, df.iloc[:,ax])
    axes.set_ylabel(df.columns[ax])
    axes.set_xlabel('Date')

