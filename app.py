from flask import Flask, render_template, url_for, redirect, make_response, jsonify
from datetime import datetime
from flask import request
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas_datareader import data as api 
from pandas_datareader._utils import RemoteDataError 
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras import Sequential
from datetime import datetime,timedelta
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()
from flask_cors import CORS,cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods = ['GET','POST'])
def dataPredict():
        data = request.json

        res = make_response(str(getPrediction(data['name'])), 200)
        return res




def getPrediction(cryptoName):
        data = cg.get_coin_market_chart_range_by_id(id = cryptoName, vs_currency='usd',from_timestamp= (datetime.now() - timedelta(days=720)).timestamp(),to_timestamp=datetime.now().timestamp())
        df = pd.DataFrame(data)
        datetime.fromtimestamp(float(df['prices'][0][0])/1000.0)
        df['timestamp'] = df['prices'].apply(lambda x : datetime.fromtimestamp(float(x[0])/1000.0))
        df['prices'] = df['prices'].apply(lambda x : x[1])
        df['market_caps'] = df['market_caps'].apply(lambda x : x[1])
        df['total_volumes'] = df['total_volumes'].apply(lambda x : x[1])
        df.drop(['market_caps'],axis=1,inplace=True)
        df['timestamp'] = df['timestamp'].apply(lambda x : x.date())
        df['next_day_price'] = df['prices'].shift(periods=-1)
        df = df.dropna()
        df.index = df['timestamp']
        scaler = MinMaxScaler(feature_range= (-1,1))
        scaler_2 = MinMaxScaler(feature_range= (-1,1))
        scaler_3 = MinMaxScaler(feature_range= (-1,1))
        df_scaled = df.copy(deep=True)
        df_scaled['prices'] = scaler.fit_transform(df['prices'].values.reshape(-1,1))
        df_scaled['prices'] = np.reshape(df_scaled['prices'],len(df_scaled['prices']))
        df_scaled['total_volumes'] = scaler_2.fit_transform(df['total_volumes'].values.reshape(-1,1))
        df_scaled['total_volumes'] = np.reshape(df_scaled['total_volumes'],len(df_scaled['total_volumes']))
        df_scaled['next_day_price'] = scaler_3.fit_transform(df['next_day_price'].values.reshape(-1,1))
        df_scaled['next_day_price'] = np.reshape(df_scaled['next_day_price'],len(df_scaled['next_day_price']))
        df_scaled = df.copy(deep=True)
        df = df.drop('timestamp',axis=1)
        X = df_scaled[['prices']][:-1]
        Y = df_scaled[['next_day_price']][:-1]
        model=Sequential()
        model.add(LSTM(units=200, activation='relu', return_sequences=True, input_shape=(1,1)))
        model.add(LSTM(units=50, activation='relu'))
        model.add(Dense(units=10, activation='relu'))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X.values.reshape(X.shape[0],X.shape[1],1),Y,epochs=25,batch_size=64,verbose=1)
        y_pred=model.predict(df_scaled[['prices']].tail(1).values.reshape(1,1,1))
        print(y_pred)
        #y_pred = scaler_3.inverse_transform(y_pred)
        return y_pred
        


if __name__ == "__main__":
    app.run(debug = True)
