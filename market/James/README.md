# Corporación Favorita Grocery Sales Forecasting

## Overview
The objective of this [challenge](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) is to build a model that more accurately forecasts product sales for the Ecuadorian-based grocery retailer. By doing so, they can better please customers by having just enough of the right products at the right time.

## Approach
There is a significant time series component to the data. To tackle this, we are exploring several types of machine learning models. As a baseline, we use a moving average model that repeatedly calculates the average sales over a short range of time. This removes the time series component from the data but also loses model expressiveness. We are experimenting with Time Series Analysis using LSTM (Long Short Term Memory) networks as well. Using these networks will allow us to fit our model to trends in the data. We are also looking into linear regression as a purely supervised approach.

### Timeline

### Useful literature
