"""
Author: @Philip.P
Created: 24/11/2018
"""
import os
import numpy as np
import pandas as pd
import random
import seaborn as sns

os.chdir("C:\\Users\\ppapasav\\Documents\\Python")

# NUMPY Basics
"""
np.zeros(10)
np.ones(10) #np.ones(10)*5
np.arange(10,51) #numbers 10 to 50

#even numbers 10 to 50
np.linspace(10,50,21) #lin space is (start,stop, #points)
#np.linspace(0,1,20)
np.arange(10,50,2)
a = np.arange(10,50,2)
a>20 #returns a bool
a[a>20] #only return values more than 20
#or in list form [i for i in range(10,51) if i%2 == 0]

np.arange(9).reshape(3,3) #3x3 matrix 1 to 8
np.eye(3) #identity matrix of dimension N=3

np.random.rand(1) #rand num between 0 and 1
np.random.randn(25) #multiple random numbers, normal distrib


mat = np.arange(1,26).reshape(5,5) #1 to 25 5x5 matrix
# array([[ 1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10],
#        [11, 12, 13, 14, 15],
#        [16, 17, 18, 19, 20],
#        [21, 22, 23, 24, 25]])
#start slicing
mat[2:5,1:] #row then column
mat[3,4] #20
mat[:3,1] #array([ 2,  7, 12])
mat[4,] #array([21, 22, 23, 24, 25])
mat[3:,]
#array([[16, 17, 18, 19, 20],
#       [21, 22, 23, 24, 25]])
mat.sum() #sum all values in mat
sum(mat) #sum columns

#array (arr) functions --> np.sqrt(arr), np.exp(arr), arr**3
#np.log(arr), np.sin(arr)

np.random.RandomState(seed=1)
#random seeds giving same random numbers
"""

# PANDAS basics
"""
"""
#series can have axis labels whereas an array cannot
#dataframes are a bunch of series objects stuck together 
labels = ['a','b','c']
my_list = [10,20,30]
arr = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}
pd.Series(data=my_list)
pd.Series(data=my_list,index=labels) #same as pd.Series(my_list,labels)

ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])
ser1['USA']

df = pd.DataFrame(np.random.randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
# df['W'] is the same as df.W (with the latter being SQL syntax)
df.drop('W', axis=1, inplace=False) #inplace=False (default), axis=1 to signal columns
df.drop('E', axis=0, inplace=False) #drop row
#df.loc['label']
#df.iloc['position'] 
# df[(df['W']>1) & (df['Y'] > 0)]
# df.reset_index() 
newkind = "CA NY WY OR CO".split()
df['states'] = newkind
df.set_index('states', inplace=True)

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
byCo = df.groupby('Company')
byCo.mean() # byCo.describe(), also .min(), .std(), .count()

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()
df['col2'].unique() #df['col2'].nunique()

#merge, join, concatenate
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

# concatenation for gluing dataframes - axes dimensions should match
pd.concat([df1,df2]) #try pd.concat([df1,df2], axis=1)

#merging putting DFs together on common axes
#adding a new line to check commit

#join is to combine columns of potentially different indexed DFs



#exercise
wkdir = "C:\\Users\\ppapasav\\Documents\\python\\data\\"
banks = pd.read_csv(wkdir + 'banklist.csv') #failed US bank data
banks.columns #columns
# ['Bank Name', 'City', 'ST', 'CERT', 'Acquiring Institution',
#        'Closing Date', 'Updated Date']
len(banks.ST.unique())  #length of list of unique states
banks['ST'].value_counts().head(5) # top 5 states by count
banks['Acquiring Institution'].value_counts().head(5)  # top 5 by acquiring institution
banks[banks['Acquiring Institution'] == 'Columbia State Bank']

banks[banks['ST']=="CA"]['City'].value_counts().head(1)
# most common city in CA for banks to fail in is Los Angeles

banks['Bank Name'].str.count('Bank').value_counts()
# 14 banks don't have bank in their name, 2 have it twice
sum(banks['Bank Name'].apply(lambda name: 'Bank' not in name))

sum(banks['Bank Name'].apply(lambda name: name[0].upper() == 'S'))
# bank names beginning with an 'S'

len(banks[banks['CERT'] > 20000]) #417 banks have CERT values above 20 000

#number of words in bank title
banks['Bank Name'].apply(lambda name: len(name.split()))
[len(i.split()) for i in banks['Bank Name']]

#filter by the number of banks which have two words in title, return bool
banks['Bank Name'].apply(lambda name: len(name.split()) ==2)

#filter by number that have closed in 2018
banks[banks['Closing Date'].apply(lambda date: date[-2:] == '17')]
"""


