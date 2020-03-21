# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:52:14 2018

@author: Md. Ashikur Rahman
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import preprocessing, model_selection
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
    B = np.array([0, 0, 0, 0, 0, 0]) #initial coef
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


var = int (input ('Number of Stocks: '))
var = int (var)
print (var)
    
d = {}
for i in range(var):
    name = input('CSV file name: ')
    data = pd.read_csv(name)
    name = name [:-4]
    d[name] = data['Adj Close']

stocks = pd.concat(d.values(), axis=1, keys=d)

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
token = 1
columns = [] #columns to the nxt step of X_opt
for i in range (var):
    columns.append(i)
sl = float (input ('Enter significance level: '))

while token == 1:
    token = 0
    regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit ()
    col = [] #columns after optimization holder
    for i in columns:
        if(regressor_OLS.pvalues[i] > sl):
            token = 1
        else:
            col.append(i)
    columns = col
    X_opt = X [:, columns]
        
#Final
X_train, X_test, Y_train, Y_test = model_selection.train_test_split (X_opt, Y, test_size = 0.2, random_state = 4)

#X_train = X_train.T
coef = multiple_regression (X_train, Y_train)

Y_pred = X_test.dot(coef)
print(rmse(Y_test, Y_pred))
print(r2_score(Y_test, Y_pred))
