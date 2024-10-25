import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-12-31'
yf.pdr_override()

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_yahoo(user_input, start, end)

#Describing Data .....
st.subheader('Data from 2010-2023')
st.write(df.describe())

#Vusializing ....

st.subheader('Closing Price Vs Time Chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close ,'b', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100, 'r',label=' 100 Days Movivg Average')
plt.plot(df.Close ,'b' , label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r' ,label=' 100 Days Movivg Average')
plt.plot(ma200, 'g',label=' 200 Days Movivg Average')
plt.plot(df.Close ,'b' , label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


#Split Data Into Training & Testing ...


data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array=scaler.fit_transform(data_train)


#Load Model ...

model = load_model('keras_model.h5')

#Testing ....

past100day=data_train.tail(100)
final_df= pd.concat([past100day,data_test], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-1: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted =y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph ....
st.subheader('Prediction VS Original')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label='Original Price' )
plt.plot(y_predicted , 'r',  label='Predicted Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)