"""
Created on 02/04/19

Linear regression on USA housing dataset
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

wkdir = "C://Users//Philip//PyCharmProjects"

#load the data
housing = pd.read_csv(wkdir + "/data" + "/USA_housing.csv")

#examine the dataset
housing.head()
housing.info()
housing.describe()

#housing.columns
# Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#        'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
#       dtype='object')


#Exploratory data analysis
outputDir = wkdir + "/data/output/"

pairplot = sns.pairplot(housing)
pairplot.figure.savefig(outputDir + "housing_pairplot.png")

distplot = sns.distplot(housing['Price'])
dplot = distplot.get_figure()
dplot.savefig(outputDir + "housing_distplot.png")
# prices look quite normally distributed

hmap = sns.heatmap(housing.corr(), cmap="plasma")
#hmap.savefig(outputDir + "housing_corr-heatmap.png")


# Train the linear regression model, split into the training and test sets.
# Within the train set you have the X array containing features to train on
# and the y array which is the target variable, in this case the Price of the
# house

# The address column is text and therefore cannot be used by the linear
# regression model
# housing.columns
X = housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', \
             'Avg. Area Number of Bedrooms', 'Area Population']]
y = housing['Price']

# Split the data into training and testing set.
# Train the model on the training set, then use the test set to
# evaluate the model

from sklearn.model_selection import train_test_split

X_train, X_test, \
    y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# Create and train the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Evaluate the model
pd.DataFrame(lm.coef_,X.columns,columns = ['Coefficient'])
#coefficients show increase in price of house per unit of column,
# in this case income of $1 more is $21.6 increase in house price


# Test model predictions
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.suptitle("Predicted price v actual house price")
plt.xlabel("Test_prices")
plt.ylabel("Predicted prices")
plt.savefig(outputDir + "housing_predicted-prices.png")