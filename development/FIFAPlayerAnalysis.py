"""
Created 26/11/18
@author: Philip P
Exploratory data analysis

"""
import utils
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import plotly
import os

player_data = "C://Users//Philip//Documents//python//input//Kaggle_FIFA17_players.csv"

df = pd.read_csv(player_data)
df = df.copy(True)
df['Height'] = pd.to_numeric(df['Height'].str.strip(' cm'))
df['Weight'] = pd.to_numeric(df['Weight'].str.strip(' kg'))

subset = df[['Name', 'Age', 'Height', 'Weight', 'Rating', 'Work_Rate', 'Club_Position']]
subset = subset.copy(True)

sns.distplot(subset['Rating'])
sns.jointplot(x = 'Work_Rate', y = 'Rating', data=subset, kind = 'hex')

sub = st[['Name', 'Age', 'Height', 'Weight', 'Rating', 'Work_Rate', 'Club_Position']]
st = df[df['Club_Position'] == 'ST']

sns.jointplot(x='Height', y='Rating', data=st, kind= 'reg')

defence = df[(df['Club_Position'] == 'RCB') & (df['Club_Position'] == 'LCB') & (df['Club_Position'] == 'RCB')]
defence = df[df['Rating'] >70]
sns.jointplot(x='Height', y='Rating', data=defence, kind= 'hex')

sns.distplot(defence['Rating'])