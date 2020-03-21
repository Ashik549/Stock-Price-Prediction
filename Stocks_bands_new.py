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
from pandas_datareader import data as web

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

def multiple_regression (X, Y, var):
    x = []
    for i in range (var):
        x.append(i)
    B = np.array(x)
    #initial coef
    alpha = 0.0001

    inital_cost = cost_function(X_train, Y_train, B)

    newB, cost_history = gradient_descent(X_train, Y_train, B, alpha, 200000)
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


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas #as a numpy array

def standard_deviation(tf, prices, date):
    sd = []
    sddate = []
    x = tf
    
    while x <= len(prices):
        array2consider = prices[x - tf:x]
        standev = array2consider.std()
        sd.append(standev)
        sddate.append(date[x])
        
        x += 1
        
    return sddate, sd

def bollinger_bands(mult, tff, closep, length, date):
    bdate = []
    topBand = []
    botBand = []
    midBand = []
    x = tff
    
    while x < length:
        curSMA = moving_average(closep[x-tff:x], tff) [-1]
        
        d, curSD = standard_deviation(tff,closep[0:tff], date)
        curSD = curSD[-1]
        
        TB = curSMA + (curSD * mult)
        BB = curSMA - (curSD * mult)
        D = date[x]
        
        bdate.append(D)
        topBand.append(TB)
        botBand.append(BB)
        midBand.append(curSMA)
        x += 1
    
    return bdate, topBand, botBand, midBand

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

#import adj. close price
def get_adj_close(ticker, start, end):
    start = start
    end = end
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Adj Close']
    return pd.DataFrame(info)


var = int (input ('Number of Stocks: '))
d = {}
'''
for i in range(var):
    name = input('CSV file name: ')
    data = pd.read_csv(name)
    name = name [:-4]
    d[name] = data['Adj Close']
'''

for i in range(var):
    name = input('Stock name: ')
    d[name] = get_adj_close(name, '1/1/2014', '12/31/2018')
    
stocks = pd.concat(d.values(), axis=1, keys=d)
stocks.fillna (-99999, inplace = True)

label = input ('Which stock you want to predict: ')
forecast_out = 2
stocks [label] = stocks [label].shift (-forecast_out)

X = np.array (stocks.drop ([label], 1))
X = preprocessing.scale (X)
m = len (stocks)
X0 = np.ones (m, dtype = np.float64)
X = np.insert (X, 0, X0, axis = 1)
X_lately = X [-forecast_out:]
X = X [:-forecast_out]
stocks.dropna (inplace = True)
Y = np.array (stocks [label])
Y_len = Y.shape
Y = Y.reshape ((Y_len[0],))


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
    regressor_OLS.summary()
    col = [] #columns after optimization holder
    for i in range (var):
        if(regressor_OLS.pvalues[i] > sl):
            token = 1
            var -= 1
        else:
            col.append(i)
    columns = col
    X_opt = X [:, columns]
        
#Final

date = stocks.index
maxi = -9
rand = 0
for i in range (100):
    X_train, X_te, Y_train, Y_te = model_selection.train_test_split (X_opt, Y, test_size = 0.2, random_state = i)
    X_tr, X_test, Y_tr, Y_test, d_tr, d_test = model_selection.train_test_split (X_opt, Y, date, test_size = 0.2, shuffle = False)

    #X_train = X_train.T
    coef = multiple_regression (X_train, Y_train, var)
        
    Y_pred = X_test.dot(coef)
    
    temp = r2_score(Y_test, Y_pred)
    if (temp > maxi):
        maxi = temp
        rand = i
        fin_coef = coef

Y_pred = X_test.dot(fin_coef)
print(rmse(Y_test, Y_pred))
print(r2_score(Y_test, Y_pred))
print(rand)
    
#Plotting
plt.plot(Y_pred)
plt.plot(Y_test)
    
'''
label_length = len(Y_test)
Y_test_bol = []
Y_pred_bol = []

for i in range (label_length):
    Y_test_bol.append(Y_test[i])
    Y_pred_bol.append(Y_pred[i])
'''

label_len = len(Y_test)
d, tb, bb, mb = bollinger_bands (2, 20, Y_test, label_len, date_test)

band_data = pd.DataFrame ({'Date': d, 'Adj Close:': Y_test, 'Top band': tb, 'Bottom band': bb, '30 Day MA': mb})
band_data.set_index('Date')

band_data[['Adj Close', '30 Day MA', 'Top Band', 'Bottom Band']].plot(figsize=(12,6))
plt.title('30 Day Bollinger Band for Facebook')
plt.ylabel('Price (USD)')
plt.show();