import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


start = '2010-01-01'
end = '2023-12-31'
yf.pdr_override()
df = pdr.get_data_yahoo('AAPL', start, end)
df=df.reset_index()
df=df.drop(['Date','Adj Close'] ,axis=1)

ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

#Split Data
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler=MinMaxScaler(feature_range=(0,1))
data_train_array=scaler.fit_transform(data_train)

x_train=[]
y_train=[]

for i in range(100,data_train_array.shape[0]):
    x_train.append(data_train_array[i-100])
    y_train.append(data_train_array[i, 0])


x_train, y_train=np.array(x_train), np.array(y_train)

model =Sequential()
model.add(LSTM(units=50, activation= 'relu', return_sequences= True, 
               input_shape= (x_train.shape[1] ,1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation= 'relu', return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation= 'relu', return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation= 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)
model.save('keras_model.h5')

past100day=data_train.tail(100)
final_df= pd.concat([past100day,data_test], ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-1: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions :

y_predicted = model.predict(x_test)

scale_factor = 1/0.02123255
y_predicted =y_predicted * scale_factor
y_test = y_test * scale_factor

