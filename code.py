import calendar
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

df=pd.read_csv('prices.csv')
company_sym = input('Enter company symbol : ')
df=df.loc[df['symbol'] == company_sym]
print(df.tail())

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import math

forecast_col = 'close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X=np.array(df.drop(['label','symbol','date'], axis=1))
X = preprocessing.scale(X,axis=0)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
forecast_set = clf.predict(X_lately)
print(forecast_set)
df['Forecast'] = np.nan
last_date = df.iloc[-1].date
last_date=dt.strptime(last_date, '%Y-%m-%d')
last_date = calendar.timegm(last_date.utctimetuple())
last_unix = last_date
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = dt.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
#print(df.tail())
df['close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
