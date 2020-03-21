
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 18:44:49 2018

@author: mohammed Fahad
"""
import pandas as pd
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import statistics

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
import matplotlib.pyplot as plt

#plt.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
#apple["Adj Close"].plot(grid = True) # Plot the adjusted closing price of AAPL
#pure

microsoft=pd.read_csv('MSFT.csv')
google=pd.read_csv('GOOG.csv')
 
# Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
stocks = pd.DataFrame({"AAPL": apple["Adj Close"],
                      "MSFT": microsoft["Adj Close"],
                      "GOOG": google["Adj Close"]})
 
stocks.head()
stocks.plot(secondary_y = ["AAPL", "MSFT"],grid = True)

#Let's use NumPy's log function, though math's log function would work just as well

 
stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()

stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
#correlation
stock_corr=stocks.corr()
print(stock_corr.head())
