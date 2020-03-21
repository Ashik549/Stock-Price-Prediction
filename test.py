# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:52:43 2018

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
 
# Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
stocks = pd.DataFrame({"AAPL": apple["Adj Close"],
                      "MSFT": microsoft["Adj Close"],
                      "GOOG": google["Adj Close"]})
 
X = stocks.iloc[:,:-1].values
Y = stocks.iloc[:,2].values

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.25, random_state = 0)

'''#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler ()
X_train = sc.fit_transform (X_train)
X_test = sc.transform (X_test)
'''
#Fitting Random Forest Classification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier (n_estimators = 400, criterion = 'entropy', random_state = 0)
classifier.fit (X_train, Y_train)

#Predicting the test set results
Y_pred = classifier.predict (X_test)

'''#fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression ()
regressor.fit (X_train, Y_train)'''
'''#Fitting Polynomial Regression to the database
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures (degree = 2)
X_poly = poly_reg.fit_transform (X)
#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X_poly, Y, test_size = 0.2, random_state = 0)
lin_reg_2 = LinearRegression ()
lin_reg_2.fit (X_train, Y_train)
accuracy = lin_reg_2.score (X_test, Y_test)
'''
#predicting the test set result
#y_pred = regressor.predict (X_test)
