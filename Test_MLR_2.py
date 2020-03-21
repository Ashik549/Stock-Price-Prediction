# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:15:15 2018

@author: Md. Ashikur Rahman
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import statistics
import matplotlib.pyplot as plt
# we will look at stock prices over the past year
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',10)
pd.set_option('display.max_colwidth',10)
pd.set_option('display.width',None)


# lets get the apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
 
#s="AAPL"
 #apple = quandl.get("WIKI/" + s, start_date=start, end_date=end)
apple=pd.read_csv('AAPL.csv')
 
type(apple)
pd.core.frame.DataFrame
apple.head()
 
#pure


#plt.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
#apple["Adj Close"].plot(grid = True) # Plot the adjusted closing price of AAPL
#pure

microsoft=pd.read_csv('MSFT.csv')
google=pd.read_csv('GOOG.csv')
facebook = pd.read_csv('FB.csv')
twitter = pd.read_csv ('TWTR.csv')
snap_inc = pd.read_csv ('SNAP.csv')
 
# Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
stocks = pd.DataFrame ({"AAPL": apple ["Adj Close"],
                        "MSFT": microsoft ["Adj Close"],
                        "GOOG": google ["Adj Close"],
                        "FB": facebook ["Adj Close"],
                        "TWTR": twitter ["Adj Close"],
                        "SNAP": snap_inc ["Adj Close"]})
stocks.fillna (-99999, inplace = True)
 
X = stocks.iloc[:, :-1].values
Y = stocks.iloc[:, 5].values

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state = 0)

#preprocessing
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler ()
X_train = sc_X.fit_transform (X_train)
X_test = sc_X.transform (X_test)

#OLS
'''import statsmodels.formula.api as sm
X = np.append (arr = np.ones ((252, 1)).astype (int), values = X, axis = 1)
X_opt = X [:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit ()
regressor_OLS.summary ()
X_opt = X [:, [0, 1, 3, 4]]
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit ()
regressor_OLS.summary ()
X_opt = X [:, [1, 3, 4]]
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit ()
regressor_OLS.summary ()

#fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression ()
regressor.fit (X_train, Y_train)
accuracy = regressor.score (X_test, Y_test)

#predicting the test set result
y_pred = regressor.predict (X_test)