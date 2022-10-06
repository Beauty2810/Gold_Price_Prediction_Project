![shutterstock_1797061261-1](https://user-images.githubusercontent.com/108981162/194382309-a09812ad-99f4-41bb-a68b-bb53a8e2907f.jpg)

# Gold Price Prediction

This project focuses on time series analysis using different Forecasting Algorithms to forecast the Gold Prices.


## Dataset
Data for this project is collected from January 1th 2016 to December 21st 2021. The data has  2182 rows in total and 2 columns in total. The columns are Date and Price. The datatype for Date column is object and that of Price column is float64. The data collected here is on daily basis.

## Exploratory Data Analysis
We did some data analysis by plotting different types of graphs with the help of matplotlib and seaborn. We also performed some basic preprocessing such as checking and handling missing values, checking for outliers and handling them if any. Also we checked for stationarity of the time series with the help of Augmented Dickey Fuller test. Our data was non-stationary so we peformed different transformations to make the data stationary for further use. 
## Model Building
We tried 6 different parametric and non-parametric time series analysis and forecasting techniques, Simple Exponential Smoothing, Double Exponential Smoothing, Holt Winter's Method, ARIMA, SARIMA and FB Prophet. We did the hyperparameter tuning for each model and used the parameters for which the model is performing the best. The models are evaluated based on different evaluation scores like AIC, MAPE, RMSE, RMPSE etc.

## Observation and Conclusion

Based on various evaluation scores we observed that, Holt Winter’s Model is performing best for the data, giving minimum MAPE (0.1314) and AIC scores compared to other models. Finally, we forecasted the Gold Prices for the next 30 days and deployed the model on streamlit.

We can safely conclude that the price of gold in the world market and the regional Indian market are very volatile and depend a lot of external factors which cannot be modelled so easily.

