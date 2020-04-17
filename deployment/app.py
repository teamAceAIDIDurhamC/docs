import streamlit as st
import pandas as pd 
from yahoo_fin import stock_info as si
import pandas_datareader.data as web
import trendln
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression

from pyFTS.data import Enrollments
from pyFTS.partitioners import Grid
from pyFTS.models import chen
from pyFTS.common import FLR


import datetime

from datetime import date

import NLP

today = date.today()

today_date = today.strftime("%d/%m/%Y")

# Long term function that uses the target date and live price
def long_term_function(text):
    st.subheader('Predict Long Term Investment of Stock')
    user_input = st.text_input("Which US stock are you interested in for long term? Please enter its stock symbol, for example, AAPL, IBM, AMZN, etc.")
    #Input widgets
    FutTarDate = st.date_input('Enter the target sold date for the stock (dd/mm/yyyy)(for example: 2025/02/01):')
    livePrice = st.number_input("Enter the current price of the stock: ")

    #Button definition that will process the long term prediciton using the previous values
    if st.button('Process Long Term'):
        #Using last 4 years data
        try:
            #st.write("Stock recommended: ", NLP.stock_recommend(user_input))
            data = web.get_data_yahoo(user_input, '1/1/2016', today_date)
            #data1 = data.tail()
            st.subheader('Data of the Stock of the Latest 5 Trading Days')
            st.table(data.tail())
            #data_show = data1.to_string(index = False)
            data.reset_index(inplace=True, drop=False)
            
            #data1 = data.reset_index(drop=True)
            # Widget to show the dataframe

            st.subheader('Long-Term Technical Analysis of the Stock (4 Years to Date)')
            plt.figure(figsize=(15,10))
            fig = trendln.plot_support_resistance(data.Close)
            st.pyplot(fig)
            
            st.subheader('Actual Price vs AI Predicted Price of the Stock (4 Years to Date)')
            process(data, FutTarDate, livePrice)
        except:
            st.text('Please enter a valid stock symbol and try again!')
            
        
# Same processing part
def process(data, FutTarDate, livePrice):
    data['Year'] = pd.DatetimeIndex(data['Date']).year
    data['Month'] = pd.DatetimeIndex(data['Date']).month
    data['Day'] = pd.DatetimeIndex(data['Date']).day
    data['logClose']=np.log(data['Close']+1)
    x = data.drop(columns = ['Date', 'logClose', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])
    y = data['logClose']

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.35, shuffle=False)

    lr = LinearRegression(n_jobs=-1)

    lr.fit(x_train, y_train)
    forest_reg = RandomForestRegressor(n_estimators=1, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)
    forest_reg.fit(x_train, y_train)
    
    #VizPricePredict(forest_reg, 'RFR', data, x)
    
    VizPricePredictComb(lr, forest_reg, 0.8, 0.2, data, x) 
    print(FutTarDate)
    targetDate = pd.DataFrame({'Year': [FutTarDate.year], 'Month': [FutTarDate.month], 'Day': [FutTarDate.day]})
    PredictPrice = whole_data_training_comb(lr, forest_reg, 0.85, 0.15,x, y, targetDate)

    returns = ((PredictPrice - livePrice)/livePrice)*100

    #Widgets to show results
    st.subheader('AI Prediction for Investment on the Stock')
    st.write('The future price on the target date is prediected as: ' + str("%.2f" %PredictPrice)  )
    st.write('If you invest today, the future returns on the chosen stock till the target date is: ' +  str("%.2f" %returns) + '%!')


#Functions
def whole_data_training_comb(model1, model2, w1, w2,x,y, targetDate):
    model1.fit(x,y)
    model2.fit(x,y)
    PredictPrice = w1*(np.exp(model1.predict(targetDate))-1)+w2*(np.exp(model2.predict(targetDate))-1)
    return PredictPrice
        
def VizPricePredict(model, name, data, x):
    PredictPrice = np.exp(model.predict(x))-1    
    plt.figure(figsize = (18,9))
    plt.plot(data['Adj Close'], color='black', label='Actual Stock Price')
    plt.plot(PredictPrice, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction-'+name, fontsize=23)
    plt.xlabel('Time Unit', fontsize=18)
    plt.ylabel('Stock Price', fontsize=18)
    plt.legend(prop={'size': 18})
    st.pyplot()

def VizPricePredictComb(model1, model2, w1, w2, data, x):
    PredictPrice = w1*(np.exp(model1.predict(x))-1)+w2*(np.exp(model2.predict(x))-1)    
    plt.figure(figsize = (18,9))
    plt.plot(data['Adj Close'], color='black', label='Actual Stock Price')
    plt.plot(PredictPrice, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction Using Combined AI Models', fontsize=23)
    plt.xlabel('Time Unit', fontsize=18)
    plt.ylabel('Stock Price', fontsize=18)
    plt.legend(prop={'size': 18})
    st.pyplot()


def short_term_function(text):
    st.subheader('Predict Stock Price of the Next Trading Day')
    user_input = st.text_input("Which US stock are you interested in for short term? Please enter its stock symbol, for example, AAPL, IBM, AMZN, etc.")
    # Widget with text
    #st.subheader('Predict the Stock Price of the Next Trading Day')
    # Button widget for shor term processing
    if st.button('Process Short Term'):
        # Using the data till today, this can be improved to get the actual date programatically
        try:
            data = web.get_data_yahoo(user_input, '3/3/2018', today_date)
        
            # The following is the same as the jupyter notebook
            data.reset_index(inplace=True, drop=False)
            data['counter'] = range(len(data))
            data = data.drop(columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
            
            
            #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])
            #plt.plot(data['counter'],data['Adj Close'])
            #plt.title('Actual Price of the Stock-2 Years to Date', fontsize=13)
            #plt.xlabel('Time Unit', fontsize=10)
            #plt.ylabel('Stock Price', fontsize=10)
            #st.pyplot()
            
            data = data['Adj Close'].values

            fs = Grid.GridPartitioner(data=data, npart=10)

            fuzzyfied = fs.fuzzyfy(data, method='maximum', mode='sets')
            patterns = FLR.generate_non_recurrent_flrs(fuzzyfied)
            model = chen.ConventionalFTS(partitioner=fs)
            model.fit(data)

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[18,8])

            forecasts = model.predict(data)
            forecasts.insert(0,None)

            orig, = plt.plot(data, label="Actual Price")
            pred, = plt.plot(forecasts, label="Predicted Price")
            plt.title('Actual Price vs AI Predicted Price of Stock-2 Years to Date', fontsize=18)
            plt.xlabel('Time Unit', fontsize=13)
            plt.ylabel('Stock Price', fontsize=13)
            plt.legend(handles=[orig, pred])
            st.pyplot()
            prediction = model.predict([data[len(data)-1]])[0]

            st.write('The stock price of the next trading day predicted by the AI is: ' +  str("%.2f" %prediction) + '!')
        except:
            st.text('Please enter a valid stock symbol and try again')

def stock_recommend_function(text):
    st.subheader('AI Recommender for Stock')
    user_input2 = st.text_input("Please enter any keywords related to the company you want to invest in:")
    # Button widget for shor term processing
    if st.button('Recommend'):
        st.write("Top 5 Matching Stocks Recommended by the AI: ")
        st.table(NLP.stock_recommend(user_input2))
        
#Sidebar widget with two options

st.sidebar.header("Options:")
long_term = st.sidebar.checkbox("Long Term Investment ", value=True)
short_term = st.sidebar.checkbox("Short Term Investment", value=False)
stock_recommend = st.sidebar.checkbox("Recommender for Stock", value=False)

#Selectbox widget
st.title("AI Assistant for Stock Investing")
st.write("*Please select what you want on the left!* :sunglasses:")
st.write("Today's date: ", today_date)
#user_input = st.text_input("Which US stock are you interested in? Please enter its stock symbol, for example, AAPL, IBM, AMZN, etc.")

#st.write("Stock recommended: ", NLP.stock_recommend(user_input))

#Based on the sidebar selection it will call those functions
if long_term:
    long_term_function('')

if short_term:
    short_term_function('')
    
if stock_recommend:
    stock_recommend_function('')
