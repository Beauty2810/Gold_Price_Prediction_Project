import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
import pickle
from pickle import dump
from pickle import load

# Reading the data
data=pd.read_csv("Gold_data.csv")
data['date'] = pd.to_datetime(data['date'])
data.set_index('date',inplace=True)


# Loading the model we already created
loaded_model=load(open('model.sav','rb'))

# Defining a funtion to forecast prices
def forecast_price(x):
    predictions=loaded_model.forecast(x)
    pred_df=pd.DataFrame({"Prediction":predictions})
    
    # Original Time Series
    #st.write("Original Time Series")
    #st.line_chart(data['price'])
    
    # Predicted values graph
    st.write("Predicted Prices")
    st.line_chart(pred_df['Prediction'])
    
    # Both together
    st.write("Combined Plot")
    combine=pd.concat([data,pred_df])
    st.line_chart(combine)
    
    return pred_df
    


def main():
    
    # giving a title
    st.title('Gold Price Prediction')
    
    # getting the input data from the user
    period=st.slider("Forecast Period",1,365,30)
    
    # Creating empty list to store the forecasted prices
    result=[]
    
    # Creating button for Forecast
    if st.button('FORECAST'):
        result= forecast_price(period)
    #st.success(result)
    st.write(result)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    