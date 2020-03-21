# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:26:27 2018

@author: Md. Ashikur Rahman
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B.T) - Y) ** 2)/(2 * m)
    return J

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B.T)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

def multiple_regression (X, Y):
    B = np.array ([0, 0, 0, 0, 0, 0]) #initial coef
    alpha = 0.0001

    inital_cost = cost_function(X_train, Y_train, B)

    newB, cost_history = gradient_descent(X_train, Y_train, B, alpha, 100000)
    return newB


# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


'''# we will look at stock prices over the past year
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',10)
pd.set_option('display.max_colwidth',10)
pd.set_option('display.width',None)
'''
apple=pd.read_csv('AAPL.csv')
 
type(apple)
pd.core.frame.DataFrame
apple.head()
 
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

forecast_out = int (math.ceil (0.005 * len (stocks)))
stocks ['TWTR'] = stocks ['TWTR'].shift (-forecast_out)

X = np.array (stocks.drop (['TWTR'], 1))
X = preprocessing.scale (X)
m = len (stocks)
X0 = np.ones (m, dtype = np.float64)
X = np.insert (X, 0, X0, axis = 1)
X_lately = X [-forecast_out:]
X = X [:-forecast_out]
stocks.dropna (inplace = True)
Y = np.array (stocks ['TWTR'])



#OLS
import statsmodels.formula.api as sm
X_opt = X
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit ()
regressor_OLS.summary ()


#Final
X_train, X_test, Y_train, Y_test = model_selection.train_test_split (X_opt, Y, test_size = 0.2, random_state = 4)

#X_train = X_train.T
coef = multiple_regression (X_train, Y_train)

Y_pred = X_test.dot(coef)
print(rmse(Y_test, Y_pred))
print(r2_score(Y_test, Y_pred))
