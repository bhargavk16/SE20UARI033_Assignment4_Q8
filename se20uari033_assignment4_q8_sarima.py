# ipynb link github repo: https://github.com/bhargavk16/SE20UARI033_Assignment4_Q8/blob/main/se20uari033_sarima_sunspot.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
data = pd.read_csv('/content/Sunspots.csv')
data.head(2)
data.columns
data.columns
data['Date']=pd.to_datetime(data['Date']) #Converting date into datetime object
data_new = data.set_index(data['Date']) #Setting the date column as index
data_new1 = data_new.drop(labels =['Date'],axis = 1) #Deleting the data column
fig = plt.figure(figsize = (10,5))
data_new1['Monthly Mean Total Sunspot Number'].plot(style = 'k.')
data_new1['2019'].resample('M').mean().plot(kind='bar')
data_q = data_new1.resample('q').mean()
data_q.head()


def adfuller_test(data):

  if isinstance(data, pd.Series):
    data = data.values

  # Reshape the 2-dimensional array to a 1-dimensional array.
    data = data.reshape(-1)

  #Augmented Dickey-Fuller test.
    result = adfuller(data)

    labels = ['ADF Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
      print(label + ':' + str(value))

  # Interpret the results.
    if result[1] <= 0.05:
      print("Data is stationary")
    else:
      print("Data is non-stationary ")
adfuller_test(data_q)
data_q.plot()
model=sm.tsa.statespace.SARIMAX(data_q['Monthly Mean Total Sunspot Number'],order=(2, 0, 2),seasonal_order=(2,0,2,6)) #seasonal_order is (p,d,q,seasonal_value) In this case I'm considering it as 6
results=model.fit()
results.summary()
data_q['forecast']=results.predict(start=1000,end=1084,dynamic=True)
data_q[['Monthly Mean Total Sunspot Number','forecast']].plot(figsize=(12,8))
pred = data_q[data_q.forecast.notna()]
pred[['Monthly Mean Total Sunspot Number','forecast']].plot(figsize=(12,8))
