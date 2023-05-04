import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

start = '2012-12-31'
end = '2023-01-01'

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock ticker', 'SBIN.NS')
df = yf.download(user_input, start , end)

# df = df.reset_index()
# df = df.drop(['Date','Adj Close'], axis=1)

st.subheader('Data from 2013 - 2022')
st.write(df.describe())

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

"---"

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.title('Closing Price vs Time Chart')
plt.plot(data_training,'g',label='Training Data')
plt.plot(data_testing,'b', label='Testing Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

"---"

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.title('Data that is going to be predicted')
plt.plot(data_testing,'b', label='Testing Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

"---"

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

sc = MinMaxScaler(feature_range=(0,1))
data_training_arr = sc.fit_transform(data_training)

past_100_days = data_training.tail(100)
final = past_100_days.append(data_testing, ignore_index=True)
input_data = sc.fit_transform(final)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

mi = final['Close'].min()
ma = final['Close'].max()
y_test_og = y_test*(ma-mi)+mi
y_testing = pd.DataFrame(y_test_og,columns=['Close Price'],index=data_testing.index)

gru = load_model('4layerGRU.h5')
y_gru = gru.predict(x_test)
y_gru_og = y_gru*(ma-mi)+mi
y_gru_pred = pd.DataFrame(y_gru_og,columns=['Predicted Price'],index=data_testing.index)

st.subheader('Predictions vs Original (Traditional GRU)')
fig2 = plt.figure(figsize=(12,6))
plt.title('Predictions of Traditional GRU model')
plt.plot(y_testing, 'b', label="Original price")
plt.plot(y_gru_pred, 'brown', label="Predicted price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
mse1 = mean_squared_error(y_test,y_gru)
rmse1 = sqrt(mse1)
st.text(f"The root mean squared error value is {rmse1}")

"---"

lstm = load_model('4layerLSTM.h5')
y_lstm = lstm.predict(x_test)
y_lstm_og = y_lstm*(ma-mi)+mi
y_lstm_pred = pd.DataFrame(y_lstm_og,columns=['Predicted Price'],index=data_testing.index)

st.subheader('Predictions vs Original (Traditional LSTM)')
fig2 = plt.figure(figsize=(12,6))
plt.title('Predictions of Traditional LSTM model')
plt.plot(y_testing, 'b', label="Original price")
plt.plot(y_lstm_pred, 'brown', label="Predicted price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
mse2 = mean_squared_error(y_test,y_lstm)
rmse2 = sqrt(mse2)
st.text(f"The root mean squared error value is {rmse2}")

"---"

hyb = load_model('4layerLSTM-GRU.h5')
y_hyb = hyb.predict(x_test)
y_hyb_og = y_hyb*(ma-mi)+mi
y_hyb_pred = pd.DataFrame(y_hyb_og,columns=['Predicted Price'],index=data_testing.index)

st.subheader('Predictions vs Original (Hybrid LSTM-GRU)')
fig2 = plt.figure(figsize=(12,6))
plt.title('Predictions of the Hybrid LSTM-GRU model')
plt.plot(y_testing, 'b', label="Original price")
plt.plot(y_hyb_pred, 'brown', label="Predicted price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
mse3 = mean_squared_error(y_test,y_hyb)
rmse3 = sqrt(mse3)
st.text(f"The root mean squared error value is {rmse3}")
"---"
st.text("As we can see the hybrid model has produced the most precise results with minimal error rate")